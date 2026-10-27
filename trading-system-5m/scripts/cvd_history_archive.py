"""
CVD History Archiver - persistiert die rollierende 10-Bucket-History aus
data/cvd/*.json in eine wachsende Archiv-Tabelle. Ohne dieses Script gibt es
KEIN Langzeit-Archiv von echtem buy_vol/sell_vol - die Live-JSON haelt nur
die letzten 10 Eintraege (~50min), Aelteres faellt raus, bevor es genutzt
werden kann. Ziel: nach ein paar Tagen genug Historie fuer einen echten
laengeren VPIN (z.B. 30 Buckets/2.5h, analog zum bestehenden synthetischen
vpin) und fuer Backtests wie bei rsi_cross/momentum/squeeze_sniper heute.

Cron: alle 5 Minuten. JSON-Fenster ist ~50min, 5-Min-Cadence gibt reichlich
Puffer, jeden Bucket sicher genau einmal zu erwischen.
"""
import pandas as pd
import json
import glob
from pathlib import Path

ARCHIVE_PATH = Path("data/research/cvd_history_archive.csv")
COLS = ["pair", "ts", "cvd", "buy_vol", "sell_vol", "trades"]

def load_archive():
    if ARCHIVE_PATH.exists():
        return pd.read_csv(ARCHIVE_PATH)
    return pd.DataFrame(columns=COLS)

def main():
    archive = load_archive()
    existing_keys = set(zip(archive["pair"], archive["ts"])) if len(archive) else set()
    new_rows = []

    for fp in glob.glob("data/cvd/*_cvd.json"):
        pair = fp.split("/")[-1].replace("_cvd.json", "").replace("_", "/")
        try:
            data = json.loads(Path(fp).read_text())
            for entry in data.get("history", []):
                key = (pair, entry["ts"])
                if key in existing_keys:
                    continue
                new_rows.append({
                    "pair": pair, "ts": entry["ts"],
                    "cvd": entry.get("cvd", 0.0),
                    "buy_vol": entry.get("buy_vol", 0.0),
                    "sell_vol": entry.get("sell_vol", 0.0),
                    "trades": entry.get("trades", 0),
                })
        except Exception:
            continue

    if new_rows:
        archive = pd.concat([archive, pd.DataFrame(new_rows)], ignore_index=True)
        archive.to_csv(ARCHIVE_PATH, index=False)

    print(f"Archiv-Groesse gesamt: {len(archive)}  neu diesen Lauf: {len(new_rows)}")

if __name__ == "__main__":
    main()
