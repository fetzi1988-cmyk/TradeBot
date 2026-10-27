#!/usr/bin/env python3
"""
Ergaenzt die 'datetime'-Spalte in vorhandenen *_1h.csv, falls sie fehlt.
Noetig, weil feature_engine.build_features eine datetime-Spalte erwartet,
der erste collect_1h_history.py sie aber noch nicht geschrieben hat.
So muss NICHT neu von Binance gesammelt werden.

Aufruf im trading-system-Verzeichnis:
    .venv/bin/python3 /root/TradeBot/scripts/repair_datetime.py data/candles
"""
import glob, os, sys
import pandas as pd

CANDLES = sys.argv[1] if len(sys.argv) > 1 else "data/candles"
TF = "1h"

fixed = 0; already = 0; err = 0
for fp in glob.glob(os.path.join(CANDLES, f"*_{TF}.csv")):
    if fp.endswith("_features.csv"):
        continue
    try:
        d = pd.read_csv(fp)
        if "datetime" in d.columns and d["datetime"].notna().all():
            already += 1
            continue
        if "timestamp" not in d.columns:
            print(f"  kein timestamp, uebersprungen: {os.path.basename(fp)}")
            err += 1
            continue
        d["datetime"] = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
        d.to_csv(fp, index=False)
        fixed += 1
    except Exception as e:
        print(f"  Fehler {os.path.basename(fp)}: {e}")
        err += 1

print(f"datetime ergaenzt: {fixed}   schon ok: {already}   Fehler: {err}")
print("Naechster Schritt:  .venv/bin/python3 scripts/build_features_tf.py --tf 1h")
