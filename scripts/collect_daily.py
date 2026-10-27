#!/usr/bin/env python3
"""
Sauberer TAGES-Daten-Sammler (1d) fuer das Regime-Halten.

Die 1h-Daten sind ein Flickenteppich (BTC nur ~13 Tage, ~6 Bars/Tag, Luecken).
Fuer einen fairen Regime-Test brauchen wir saubere, tiefe TAGESDATEN:
  - BTC/USDT  (tiefer Taktgeber fuer das Markt-Regime, Jahre an Historie)
  - alle Pairs, die das System schon kennt (aus data/candles/, als *_USDC-Spot)

Speichert nach data/daily/<KEY>_1d.csv mit
  timestamp(ms) | open | high | low | close | volume | datetime(ISO-UTC)

Aufruf auf dem Server (im trading-system):
  .venv/bin/python3 /root/TradeBot/scripts/collect_daily.py
  .venv/bin/python3 /root/TradeBot/scripts/collect_daily.py 1500   # Ziel-Tage
"""
import sys, os, glob, time
import ccxt
import pandas as pd

OUT = "data/daily"
SRC = "data/candles"
TF = "1d"
TF_MS = 24 * 60 * 60 * 1000


def known_keys():
    keys = set()
    for pat in ("*_1h.csv", "*_1h_features.csv", "*_5m.csv", "*_5m_features.csv"):
        for fp in glob.glob(os.path.join(SRC, pat)):
            b = os.path.basename(fp)
            for suf in ("_1h_features.csv", "_1h.csv", "_5m_features.csv", "_5m.csv"):
                if b.endswith(suf):
                    keys.add(b[:-len(suf)]); break
    return sorted(keys)


def fetch_deep(ex, symbol, target):
    rows, now = [], ex.milliseconds()
    cursor = now - target * TF_MS
    while True:
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe=TF, since=cursor, limit=1000)
        except Exception as e:
            print(f"    fetch-Fehler {symbol}: {e}"); break
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0]
        if last <= cursor:
            break
        cursor = last + TF_MS
        if cursor >= now or len(rows) >= target + 50:
            break
        time.sleep(ex.rateLimit / 1000.0)
    seen = {r[0]: r for r in rows}
    return [seen[k] for k in sorted(seen)]


def save(rows, key):
    if len(rows) < 30:
        return False
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.to_csv(os.path.join(OUT, f"{key}_{TF}.csv"), index=False)
    return True


def main(target=1500, whitelist=None):
    os.makedirs(OUT, exist_ok=True)
    ex = ccxt.binance({"enableRateLimit": True})
    # whitelist (Komma-Liste von Keys) -> nur diese; sonst alle bekannten Pairs.
    # BTC/USDT kommt immer als tiefer Regime-Taktgeber dazu.
    keys = whitelist if whitelist else known_keys()
    print(f"Sammle {TF}-Historie fuer {len(keys)} Pairs + BTC/USDT, Ziel ~{target} Tage")
    ok, err = 0, []
    # erst der Taktgeber
    if save(fetch_deep(ex, "BTC/USDT", target), "BTC_USDT"):
        ok += 1; print("  BTC_USDT: Taktgeber gespeichert")
    else:
        err.append("BTC_USDT")
    for i, key in enumerate(keys, 1):
        if key == "BTC_USDT":
            continue
        sym = key.replace("_", "/")
        rows = fetch_deep(ex, sym, target)
        if save(rows, key):
            ok += 1
        else:
            err.append(key)
        if i % 25 == 0 or i == len(keys):
            print(f"  [{i}/{len(keys)}] {key}: {len(rows)} Tage")
    print(f"\nFertig: {ok} Pairs mit Tagesdaten in {OUT}.")
    if err:
        print(f"Zu wenig/Fehler bei {len(err)}: {', '.join(err[:12])}{' ...' if len(err) > 12 else ''}")
    print("\nNaechster Schritt:")
    print("  .venv/bin/python3 /root/TradeBot/core/regime_hold.py data/daily 1d 150")


if __name__ == "__main__":
    tb = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    wl = [k.strip() for k in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    main(tb, wl)
