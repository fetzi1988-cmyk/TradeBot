#!/usr/bin/env python3
"""
ENDVISION-1h — tiefe 1h-Historie sammeln.

Holt fuer alle Pairs, die das System schon kennt (vorhandene *_1h.csv ODER
*_1h_features.csv in data/candles/), moeglichst viele 1h-Kerzen von Binance
per Pagination rueckwaerts und ueberschreibt *_1h.csv mit der tiefen Historie.

Danach:  .venv/bin/python3 scripts/build_features_tf.py --tf 1h
um die *_1h_features.csv mit der tiefen Historie neu zu erzeugen.

Aufruf auf dem Server (im trading-system-Verzeichnis):
    .venv/bin/python3 scripts/collect_1h_history.py          # Standard: ~3000 Kerzen
    .venv/bin/python3 scripts/collect_1h_history.py 5000     # mehr Tiefe
    .venv/bin/python3 scripts/collect_1h_history.py 3000 20  # nur erste 20 Pairs (Test)
"""
import sys, time, glob, os
import ccxt
import pandas as pd

CANDLES = "data/candles"
TF = "1h"
TF_MS = 60 * 60 * 1000


def pair_list():
    pairs = set()
    for pat in (f"*_{TF}.csv", f"*_{TF}_features.csv"):
        for fp in glob.glob(os.path.join(CANDLES, pat)):
            key = os.path.basename(fp).replace(f"_{TF}_features.csv", "").replace(f"_{TF}.csv", "")
            pairs.add(key)
    # key z.B. "AAVE_USDC" -> ccxt-Symbol "AAVE/USDC"
    return sorted(pairs)


def fetch_deep(ex, symbol, target_bars):
    """Paginiert rueckwaerts, bis target_bars erreicht oder keine Daten mehr."""
    all_rows = []
    now = ex.milliseconds()
    since = now - target_bars * TF_MS
    cursor = since
    while True:
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe=TF, since=cursor, limit=1000)
        except Exception as e:
            print(f"    fetch-Fehler bei {symbol}: {e}")
            break
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:          # kein Fortschritt
            break
        cursor = last_ts + TF_MS
        if cursor >= now or len(all_rows) >= target_bars + 50:
            break
        time.sleep(ex.rateLimit / 1000.0)
    # dedupe + sortieren
    seen = {}
    for r in all_rows:
        seen[r[0]] = r
    rows = [seen[k] for k in sorted(seen)]
    return rows


def main(target_bars=3000, limit_pairs=None):
    ex = ccxt.binance({"enableRateLimit": True})
    pairs = pair_list()
    if limit_pairs:
        pairs = pairs[:limit_pairs]
    print(f"Sammle {TF}-Historie fuer {len(pairs)} Pairs, Ziel ~{target_bars} Kerzen/Pair")
    ok = 0; fehler = []
    for i, key in enumerate(pairs, 1):
        symbol = key.replace("_", "/")
        rows = fetch_deep(ex, symbol, target_bars)
        if len(rows) < 100:
            fehler.append((key, f"nur {len(rows)} Kerzen"))
            continue
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # datetime-Spalte, die feature_engine.build_features erwartet (ISO-UTC)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.to_csv(os.path.join(CANDLES, f"{key}_{TF}.csv"), index=False)
        ok += 1
        if i % 20 == 0 or i == len(pairs):
            print(f"  [{i}/{len(pairs)}] {key}: {len(rows)} Kerzen")
    print(f"\nFertig: {ok}/{len(pairs)} Pairs mit tiefer Historie geschrieben.")
    if fehler:
        print(f"Fehler/zu wenig bei {len(fehler)}:")
        for k, e in fehler[:10]:
            print(f"  {k}: {e}")
    print("\nNaechster Schritt:")
    print("  .venv/bin/python3 scripts/build_features_tf.py --tf 1h")


if __name__ == "__main__":
    tb = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(tb, lim)
