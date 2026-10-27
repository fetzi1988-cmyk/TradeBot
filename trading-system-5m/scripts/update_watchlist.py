#!/usr/bin/env python3
"""
Stündliche dynamische Pairlist-Aktualisierung.
Wählt Top-40 USDC-Pairs aus ALLEN Binance-USDC-Pairs nach Volumen.
Filter: Stablecoins, Gold, Fiat, Unicode-Garbage, Extrempumps (>40% 24h).
"""
import ccxt
import yaml
from pathlib import Path
from datetime import datetime, timezone

PAIRS_FILE = Path("/root/trading-system/config/pairs.yaml")

# ZURUECKGENOMMEN 27.07.2026 - Wert war vom 22.-27.07. auf 1_500_000 erhoeht
# (Cut 40 -> 20 Pairs). BEGRUENDUNG DAMALS WAR METHODISCH FEHLERHAFT: die
# "monotone Treppe ueber sieben Stufen" (Top-10/15/18/20/25/30/alle) bestand
# aus VERSCHACHTELTEN Mengen - Top-15 enthaelt alle Top-10-Pairs plus fuenf.
# Jeder Schritt aendert nur wenige Pairs, die Kennzahl bewegt sich dadurch
# fast zwangslaeufig glatt. Das waren nie sieben unabhaengige Bestaetigungen,
# sondern eine Messung, redundant wiederholt.
# GEGENPRUEFUNG mit DISJUNKTEN Quartilen (27.07.) zeigte kein sauberes
# Muster: momentum WR ueber die vier Liquiditaets-Quartile 47.6/69.4/43.2/
# 55.6% - springt, keine Treppe. LIVE-ERGEBNIS bestaetigte das: WR fiel nach
# dem Cut von 51.8% auf 42.6% (n=61), Hard-Stop-Rate praktisch unveraendert.
# Zusatzbefund: Volatilitaet (entry_atr_pct) trennt bei momentum UND
# rsi_cross besser als Liquiditaet - und zwar in ENTGEGENGESETZTE Richtung
# (momentum will Bewegung, rsi_cross will Ruhe). Ein globaler Pair-Cut kann
# beide nicht gleichzeitig bedienen. Nicht erneut global anheben ohne neue,
# disjunkt gepruefte Evidenz.
MIN_VOLUME   = 500_000   # mind. 500k USDC 24h Volumen
MAX_PAIRS    = 40
MAX_CHANGE   = 40.0      # max 40% 24h-Änderung (abs) — filtert Extrempumps/Dumps

# Einheitliche Blacklist (synchron mit update_pairs.py)
EXCLUDE = {
    # Stablecoins & Fiat
    "USD1/USDC", "EUR/USDC", "FDUSD/USDC", "USDE/USDC", "EURI/USDC",
    "AUD/USDC",  "USDT/USDC", "BFUSD/USDC",
    # Gold & Commodities
    "PAXG/USDC", "XAUT/USDC",
    # Bekannte Problemfälle
    "TON/USDC",  "BONK/USDC", "PENGU/USDC",
    "WLFI/USDC", "ROBO/USDC", "ENJ/USDC",  "DASH/USDC",
    "GIGGLE/USDC", "0G/USDC", "2Z/USDC",   "U/USDC",
}

# Core-Pairs die immer drin bleiben (auch bei kurzfristig niedrigem Volumen)
CORE_PAIRS = {"BTC/USDC", "ETH/USDC", "SOL/USDC", "BNB/USDC"}

exchange = ccxt.binance({"enableRateLimit": True})
exchange.load_markets()

usdc_markets = [s for s in exchange.markets if s.endswith("/USDC")]

results = []
skipped_pump  = []
skipped_ascii = []
skipped_excl  = []

for p in usdc_markets:
    # Filter 1: Nur ASCII (fängt 币安人生 und ähnliche)
    if not p.isascii():
        skipped_ascii.append(p)
        continue

    # Filter 2: Blacklist
    if p in EXCLUDE:
        skipped_excl.append(p)
        continue

    try:
        ticker = exchange.fetch_ticker(p)
        vol    = float(ticker.get("quoteVolume",  0) or 0)
        change = abs(float(ticker.get("percentage", 0) or 0))

        if vol < MIN_VOLUME:
            continue

        # Kein Extrempump-Filter — per-Coin-Stop übernimmt das

        results.append((p, vol))

    except Exception:
        pass

# Nach Volumen sortieren, Top-40 nehmen
results.sort(key=lambda x: x[1], reverse=True)
top_pairs = [p for p, _ in results[:MAX_PAIRS]]

# Core-Pairs garantieren
for cp in CORE_PAIRS:
    if cp not in top_pairs:
        top_pairs.insert(0, cp)
top_pairs = top_pairs[:MAX_PAIRS]

# pairs.yaml schreiben
current   = yaml.safe_load(PAIRS_FILE.read_text()) or {}
old_pairs = set(current.get("pairs", []))
new_pairs = set(top_pairs)

current["pairs"] = top_pairs
PAIRS_FILE.write_text(yaml.safe_dump(current, sort_keys=False, allow_unicode=False))

# Log
ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
added   = new_pairs - old_pairs
removed = old_pairs - new_pairs
print(f"[{ts}] Pairs: {len(top_pairs)} | +{len(added)} -{len(removed)}")
if added:
    print(f"  Neu:      {', '.join(sorted(added))}")
if removed:
    print(f"  Entfernt: {', '.join(sorted(removed))}")
if skipped_pump:
    print(f"  Pump-Filter ({len(skipped_pump)}): {', '.join(f'{p}({c:.0f}%)' for p,c in skipped_pump)}")
if skipped_ascii:
    print(f"  Unicode-Filter: {len(skipped_ascii)} Pairs")
