import pandas as pd
import numpy as np
import glob
import yaml
from pathlib import Path

LOG_PATH = Path("data/research/cvd_divergence_log.csv")
COLS = ["pair", "signal_time", "price_chg_6", "cvd_chg_6", "entry_price", "fwd30"]

def load_active_pairs():
    # Scope-Fix 30.06.2026: nur aktive Watchlist-Pairs durchsuchen, nicht alle
    # Datei-Leichen aus frueheren Rotationen in data/candles/ - spart Zeit
    # und verhindert Verwechslung mit dem Indikator-Sweep-Datenmuell-Fund.
    data = yaml.safe_load(open("config/pairs.yaml").read())
    pairs = data if isinstance(data, list) else data.get("pairs", [])
    return set(pairs)

def load_log():
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH, parse_dates=["signal_time"])
    return pd.DataFrame(columns=COLS)

def main():
    log = load_log()
    existing_keys = set(zip(log["pair"], log["signal_time"].astype(str))) if len(log) else set()
    new_rows = []
    active = load_active_pairs()

    for fp in glob.glob("data/candles/*_5m_features.csv"):
        pair = fp.split("/")[-1].replace("_5m_features.csv", "").replace("_", "/")
        if pair not in active:
            continue
        try:
            df = pd.read_csv(fp)
            if "cvd_trend_6" not in df.columns or len(df) < 30:
                continue
            tcol = df.columns[0]
            df["dt"] = pd.to_datetime(df[tcol], unit="ms", utc=True, errors="coerce")
            if df["dt"].isna().all():
                df["dt"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
            _now = pd.Timestamp.now(tz="UTC")
            _last_dt = df["dt"].max()
            if pd.isna(_last_dt) or (_now - _last_dt).total_seconds() / 3600 > 6:
                continue
            df["price_chg_6"] = df["close"].pct_change(6) * 100
            df["cvd_chg_6"] = df["cvd_trend_6"].diff(6)
            df["fwd30"] = (df["close"].shift(-6) / df["close"] - 1) * 100

            qualifying = df[df["price_chg_6"] > 0.3].dropna(subset=["price_chg_6", "cvd_chg_6"])
            for _, row in qualifying.iterrows():
                key = (pair, str(row["dt"]))
                if key in existing_keys:
                    continue
                new_rows.append({
                    "pair": pair, "signal_time": row["dt"],
                    "price_chg_6": row["price_chg_6"], "cvd_chg_6": row["cvd_chg_6"],
                    "entry_price": row["close"], "fwd30": row["fwd30"] if pd.notna(row["fwd30"]) else None,
                })
        except Exception:
            continue

    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)

    # Nachtraegliches Auffuellen: alte Eintraege ohne fwd30, falls inzwischen Daten da sind
    missing = log[log["fwd30"].isna()]
    for idx, row in missing.iterrows():
        fp = f"data/candles/{row['pair'].replace('/','_')}_5m_features.csv"
        try:
            df = pd.read_csv(fp)
            tcol = df.columns[0]
            df["dt"] = pd.to_datetime(df[tcol], unit="ms", utc=True, errors="coerce")
            if df["dt"].isna().all():
                df["dt"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
            match = df[df["dt"] == row["signal_time"]]
            if len(match):
                pos = match.index[0]
                if pos + 6 < len(df):
                    fwd = (df["close"].iloc[pos+6] / df["close"].iloc[pos] - 1) * 100
                    log.at[idx, "fwd30"] = fwd
        except Exception:
            continue

    log.to_csv(LOG_PATH, index=False)
    resolved = log["fwd30"].notna().sum()
    print(f"Log-Groesse gesamt: {len(log)}  davon mit fwd30 aufgeloest: {resolved}  neu heute: {len(new_rows)}")

if __name__ == "__main__":
    main()
