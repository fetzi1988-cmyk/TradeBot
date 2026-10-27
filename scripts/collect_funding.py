#!/usr/bin/env python3
"""
FUNDING-ARB — historische Funding-Raten (Binance USDS-M Perpetuals) sammeln.

Idee dahinter: delta-neutrale Funding-Arbitrage = Spot LONG + Perp SHORT in gleicher
Notional. Preisrisiko hebt sich (naeherungsweise) auf, uebrig bleibt die Funding-Rate,
die der Short bei positivem Funding KASSIERT (und bei negativem Funding ZAHLT).
Dieses Skript holt nur die Rohdaten; die Auswertung macht core/backtest_funding.py.

EINHEIT (kritisch, bitte nie vermischen):
  funding_rate ist ein BRUCH pro Funding-Intervall, KEIN Prozentwert.
  Binance-Standard = 8h-Intervall (3 Abrechnungen pro Tag).
    0.0001  = 0.01 % pro 8h  ->  0.0001 * 3 * 365 = 0.1095 = 10.95 % pro Jahr
  Manche Perps haben 4h- oder 1h-Intervalle; die Rate gilt dann je Abrechnung.
  Gespeichert wird die Rate UNVERAENDERT so, wie Binance/ccxt sie liefert (Bruch).

Universum = Schnittmenge aus
  (a) Pairs, die das System schon kennt (vorhandene *_1h.csv ODER *_1h_features.csv
      in data/candles/, gleiche Key-Logik wie collect_1h_history.py, z.B. "AAVE_USDC"),
  (b) Pairs, zu denen es auf Binance USDS-M einen aktiven linearen Perp gibt.
      Perps sind oft als "AAVE/USDT:USDT" notiert (nicht "AAVE/USDC") -> wird robust
      aufgeloest (gleiche Quote zuerst, dann USDT/USDC; "1000"-Praefix-Kontrakte wie
      1000PEPE als Fallback). Pairs ohne Perp werden uebersprungen.

Ausgabe je Pair:  data/funding/<KEY>_funding.csv
  Spalten: timestamp (ms, int, UTC) | funding_rate (float, BRUCH pro Intervall) | datetime (ISO-UTC)
Zusatz:    data/funding/_symbol_map.csv (KEY -> verwendetes Perp-Symbol, zur Kontrolle)

Danach:  .venv/bin/python3 /root/TradeBot/core/backtest_funding.py data/funding

Aufruf auf dem Server (im trading-system-Verzeichnis):
    cd /root/trading-system
    .venv/bin/python3 /root/TradeBot/scripts/collect_funding.py          # Standard: ~1500 Intervalle (8h) ~ 500 Tage
    .venv/bin/python3 /root/TradeBot/scripts/collect_funding.py 2000     # mehr Tiefe
    .venv/bin/python3 /root/TradeBot/scripts/collect_funding.py 1500 20  # nur erste 20 Pairs (Test)
"""
import sys, time, glob, os
import ccxt
import pandas as pd

CANDLES = "data/candles"
OUT_DIR = "data/funding"
TF = "1h"                          # nur zur Pair-Erkennung (welche *_1h*.csv gibt es)
FUND_INTERVAL_MS = 8 * 60 * 60 * 1000   # Standard-Funding-Intervall 8h (nur fuer Zielzeitraum)
PAGE_LIMIT = 1000                  # Binance liefert max. 1000 Eintraege pro Request
MAX_PAGES = 50                     # Sicherung gegen Endlosschleife
MIN_EVENTS = 30                    # weniger Funding-Abrechnungen -> Pair als Fehler werten
SANITY_MAX_ABS_RATE = 0.05         # |Rate| > 0.05 (= 5 % pro Intervall) waere verdaechtig
                                   # (Binance-Caps liegen weit darunter) -> Einheiten-Warnung
SANITY_MAX_MEDIAN_ABS = 0.002      # Median |Rate| > 0.002 (0.2 % pro 8h) als typischer Wert unrealistisch
                                   # (typisch 0.0001) -> Warnung (Prozent statt Bruch?)
STABLE_QUOTES = ("USDT", "USDC", "FDUSD")   # nur Stablecoin-Quotes sind sinnvoll absicherbar
PERP_QUOTES = ("USDT", "USDC")     # Quoten, in denen Binance USDS-M Perps anbietet
BASE_PREFIXES = ("", "1000", "1000000")     # z.B. PEPE -> 1000PEPE (Kontraktgroesse; Rate ist
                                   # ein Bruch und damit unabhaengig vom Multiplikator)


def pair_list():
    """Pair-Keys wie in collect_1h_history.py, z.B. 'AAVE_USDC'."""
    pairs = set()
    for pat in (f"*_{TF}.csv", f"*_{TF}_features.csv"):
        for fp in glob.glob(os.path.join(CANDLES, pat)):
            key = os.path.basename(fp).replace(f"_{TF}_features.csv", "").replace(f"_{TF}.csv", "")
            pairs.add(key)
    return sorted(pairs)


def resolve_perp(markets, key):
    """Loest Pair-Key ('AAVE_USDC') auf ein existierendes Binance-USDS-M-Perp-Symbol auf.
    Gibt (symbol, grund) zurueck; symbol=None wenn kein Perp gefunden (grund erklaert warum).
    Reihenfolge: einfache Basis ueber alle Quoten (gleiche Quote wie Spot zuerst),
    erst danach 1000er-Praefix-Kontrakte."""
    if "_" not in key:
        return None, "Key ohne Quote"
    base, quote = key.split("_", 1)
    if quote not in STABLE_QUOTES:
        return None, f"Quote {quote} nicht stabil (kein sauberer Hedge)"
    quotes = [quote] + [q for q in PERP_QUOTES if q != quote]
    for prefix in BASE_PREFIXES:
        for q in quotes:
            sym = f"{prefix}{base}/{q}:{q}"          # ccxt-Notation linearer Perp
            m = markets.get(sym)
            if m and m.get("swap") and m.get("linear") is not False and m.get("active") is not False:
                return sym, "ok"
    return None, "kein aktiver Perp"


def fetch_funding_deep(ex, symbol, target_events):
    """Paginiert vorwaerts ab (jetzt - target_events*8h) bis jetzt (since wird immer weiter
    nach vorn geschoben, die Historie wird so rueckwaerts bis zum Ziel abgedeckt).
    Rueckgabe: Liste von (timestamp_ms:int, funding_rate:float BRUCH pro Intervall)."""
    all_rows = []
    now = ex.milliseconds()
    cursor = now - target_events * FUND_INTERVAL_MS
    for _ in range(MAX_PAGES):
        batch = None
        for versuch in range(3):                      # einfache Retries bei Netz/Rate-Limit
            try:
                batch = ex.fetch_funding_rate_history(symbol, since=cursor, limit=PAGE_LIMIT)
                break
            except ccxt.NetworkError as e:
                print(f"    Netz-/Limit-Fehler bei {symbol} (Versuch {versuch+1}/3): {e}")
                time.sleep(2.0 * (versuch + 1))
            except Exception as e:                    # z.B. BadSymbol -> kein Retry
                print(f"    fetch-Fehler bei {symbol}: {e}")
                break
        if not batch:
            break
        for r in batch:
            ts = r.get("timestamp")
            fr = r.get("fundingRate")
            if ts is None or fr is None:
                continue
            all_rows.append((int(ts), float(fr)))     # fr: BRUCH pro Intervall (0.0001 = 0.01 %)
        last_ts = int(batch[-1]["timestamp"])
        if last_ts < cursor:                          # kein Fortschritt
            break
        cursor = last_ts + 1
        if cursor >= now:
            break
        time.sleep(ex.rateLimit / 1000.0)
    # dedupe (nach Timestamp) + sortieren
    seen = {}
    for ts, fr in all_rows:
        seen[ts] = fr
    return [(ts, seen[ts]) for ts in sorted(seen)]


def median_interval_h(rows):
    """Typisches Funding-Intervall in Stunden (nur Info; 8h ist Standard)."""
    if len(rows) < 3:
        return float("nan")
    ts = pd.Series([r[0] for r in rows])
    return float(ts.diff().median()) / 3_600_000.0


def main(target_events=1500, limit_pairs=None):
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    try:
        markets = ex.load_markets()
    except Exception as e:
        print(f"Markets laden fehlgeschlagen: {e}")
        sys.exit(1)

    pairs = pair_list()
    if not pairs:
        print(f"Keine *_{TF}.csv / *_{TF}_features.csv in {CANDLES} gefunden "
              f"(im trading-system-Verzeichnis ausfuehren?).")
        sys.exit(1)

    # Universum: nur Pairs mit Perp
    universe = []                                     # (key, perp_symbol)
    ohne_perp = []
    for key in pairs:
        sym, grund = resolve_perp(markets, key)
        if sym is None:
            ohne_perp.append((key, grund))
        else:
            universe.append((key, sym))
    if limit_pairs:
        universe = universe[:limit_pairs]

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{len(pairs)} bekannte Pairs: {len(universe)} im Lauf (mit Perp"
          f"{', limit_pairs aktiv' if limit_pairs else ''}), {len(ohne_perp)} ohne Perp (uebersprungen).")
    print(f"Sammle Funding-Historie, Ziel ~{target_events} Intervalle (8h) ~ "
          f"{target_events * 8 / 24:.0f} Tage/Pair. Einheit: Bruch pro Intervall.")

    ok = 0; fehler = []; warn = []; nicht_8h = []; symbol_map = []
    for i, (key, symbol) in enumerate(universe, 1):
        rows = fetch_funding_deep(ex, symbol, target_events)
        if len(rows) < MIN_EVENTS:
            fehler.append((key, f"{symbol}: nur {len(rows)} Funding-Eintraege"))
            continue
        df = pd.DataFrame(rows, columns=["timestamp", "funding_rate"])
        df["timestamp"] = df["timestamp"].astype("int64")         # ms seit Epoch, UTC
        df["funding_rate"] = df["funding_rate"].astype("float64") # BRUCH pro Intervall
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        max_abs = float(df["funding_rate"].abs().max())
        if max_abs > SANITY_MAX_ABS_RATE:
            warn.append((key, f"max |Rate| = {max_abs:.4f} (= {max_abs*100:.2f} % pro Intervall) "
                              f"-> Einheiten pruefen!"))
        med_abs = float(df["funding_rate"].abs().median())
        if med_abs > SANITY_MAX_MEDIAN_ABS:
            warn.append((key, f"Median |Rate| = {med_abs:.4f} (= {med_abs*100:.2f} % pro Intervall) "
                              f"-> Einheiten pruefen!"))
        med_h = median_interval_h(rows)
        if med_h == med_h and abs(med_h - 8.0) > 0.5:             # NaN-sicher
            nicht_8h.append((key, med_h))
        df.to_csv(os.path.join(OUT_DIR, f"{key}_funding.csv"), index=False)
        symbol_map.append((key, symbol, len(rows), round(med_h, 2)))
        ok += 1
        if i % 20 == 0 or i == len(universe):
            print(f"  [{i}/{len(universe)}] {key} ({symbol}): {len(rows)} Eintraege, "
                  f"Intervall ~{med_h:.1f}h")

    if symbol_map:
        pd.DataFrame(symbol_map, columns=["key", "perp_symbol", "n_events", "median_interval_h"]) \
          .to_csv(os.path.join(OUT_DIR, "_symbol_map.csv"), index=False)

    print(f"\nFertig: {ok}/{len(universe)} Pairs mit Funding-Historie nach {OUT_DIR}/ geschrieben.")
    print(f"Ohne Perp uebersprungen: {len(ohne_perp)}")
    for k, g in ohne_perp[:10]:
        print(f"  {k}: {g}")
    if fehler:
        print(f"Fehler/zu wenig Daten bei {len(fehler)}:")
        for k, e in fehler[:10]:
            print(f"  {k}: {e}")
    if warn:
        print(f"Einheiten-Warnungen bei {len(warn)}:")
        for k, e in warn[:10]:
            print(f"  {k}: {e}")
    if nicht_8h:
        print(f"Hinweis: {len(nicht_8h)} Pairs mit Nicht-8h-Intervall (Rate gilt je Abrechnung; "
              f"der Backtest annualisiert zeitbasiert, nicht ueber 3/Tag):")
        for k, h in nicht_8h[:10]:
            print(f"  {k}: ~{h:.1f}h")
    print("\nNaechster Schritt:")
    print("  .venv/bin/python3 /root/TradeBot/core/backtest_funding.py data/funding")


if __name__ == "__main__":
    te = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(te, lim)
