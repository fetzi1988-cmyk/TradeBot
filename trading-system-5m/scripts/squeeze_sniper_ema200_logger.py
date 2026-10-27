import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import glob
import yaml
from pathlib import Path

LOG_PATH = Path("data/research/squeeze_sniper_ema200_log.csv")
COLS = ["pair", "group", "signal_time", "entry_price", "atr_pct_at_signal", "pnl", "exit_reason"]

MAX_CANDLES = 60  # ~5h Vorlauf fuer die Simulation
RESOLVE_BUFFER_HOURS = 5.5  # erst nach dieser Wartezeit gilt "end_data" als endgueltig
ROLLOFF_HOURS = 40  # danach gilt eine unauffindbare Zeile als endgueltig verloren (Fenster ~1.45 Tage = ~35h)


def to_bool(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1"])


def load_active_pairs():
    data = yaml.safe_load(open("config/pairs.yaml").read())
    pairs = data if isinstance(data, list) else data.get("pairs", [])
    return set(pairs)


def load_log():
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH, parse_dates=["signal_time"])
    return pd.DataFrame(columns=COLS)


def parse_dt_column(df):
    tcol = df.columns[0]
    if tcol == "timestamp":
        s = pd.to_numeric(df[tcol], errors="coerce")
        unit = "ms" if s.dropna().median() > 10_000_000_000 else "s"
        df["dt"] = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
    else:
        df["dt"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    if df["dt"].isna().all() and "datetime" in df.columns:
        df["dt"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    return df


def prep_columns(df):
    for c in ["open", "high", "low", "close", "vwap", "ema_9", "ema_200", "vol_ratio", "atr_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["cvd_accumulation", "vpin_toxic", "ofi_bullish", "dif_cross_up_dea", "obv_up"]:
        df[c] = to_bool(df[c]) if c in df.columns else False
    return df


def simulate_sniper(df, idx):
    if idx + 1 >= len(df):
        return None
    e = df.iloc[idx + 1]
    entry_time, entry = e["dt"], float(e["open"])
    if entry <= 0:
        return None
    path = df.iloc[idx + 1: idx + 1 + MAX_CANDLES]
    if path.empty:
        return None

    atr_pct = float(df.loc[idx, "atr_pct"]) if pd.notna(df.loc[idx, "atr_pct"]) else 0.008
    dynamic_stop = max(-8.0, min(-0.30, -(atr_pct * 120)))
    tp_sniper = max(2.00, min(8.00, atr_pct * 500))
    break_even_active = False

    for _, row in path.iterrows():
        hold = (row["dt"] - entry_time).total_seconds() / 60
        lo = (row["low"] / entry - 1) * 100
        cl = (row["close"] / entry - 1) * 100

        if cl <= dynamic_stop:
            return cl, "hard_stop"
        if cl >= 1.0 and not break_even_active:
            break_even_active = True
        if break_even_active and lo <= 0.20:
            return 0.20, "break_even_stop"
        if cl >= tp_sniper:
            return cl, "sniper_take_profit"
        ema9 = float(row.get("ema_9", 0.0) or 0.0)
        if ema9 > 0 and row["close"] < ema9 and cl > 0.50:
            return cl, "sniper_trail_stop"
        vwap = float(row.get("vwap", 0.0) or 0.0)
        if vwap > 0 and row["close"] < vwap and hold >= 15 and cl < 0.0:
            return cl, "sniper_vwap_lost"
        if hold >= 45 and cl < 0.0:
            return cl, "sniper_timeout"

    return (path.iloc[-1]["close"] / entry - 1) * 100, "end_data"


def main():
    log = load_log()
    existing_keys = set(zip(log["pair"], log["group"], log["signal_time"].astype(str))) if len(log) else set()
    new_rows = []
    active = load_active_pairs()

    for fp in glob.glob("data/candles/*_5m_features.csv"):
        pair = fp.split("/")[-1].replace("_5m_features.csv", "").replace("_", "/")
        if pair not in active:
            continue
        try:
            df = pd.read_csv(fp)
            required = {"cvd_accumulation", "vpin_toxic", "ofi_bullish", "dif_cross_up_dea",
                        "obv_up", "vol_ratio", "ema_200", "close", "atr_pct"}
            if not required.issubset(df.columns) or len(df) < 30:
                continue
            df = parse_dt_column(df)
            df = prep_columns(df)
            _now = pd.Timestamp.now(tz="UTC")
            _last_dt = df["dt"].max()
            if pd.isna(_last_dt) or (_now - _last_dt).total_seconds() / 3600 > 6:
                continue

            six = (
                df["cvd_accumulation"] & df["vpin_toxic"] & (~df["ofi_bullish"])
                & df["dif_cross_up_dea"] & df["obv_up"] & (df["vol_ratio"] > 1.80)
            )
            groups = {
                "allowed": six & (df["close"] > df["ema_200"]),
                "near_miss": six & (df["close"] <= df["ema_200"]),
            }
            for group_name, cond in groups.items():
                qualifying = df[cond.fillna(False)]
                for _, row in qualifying.iterrows():
                    key = (pair, group_name, str(row["dt"]))
                    if key in existing_keys:
                        continue
                    new_rows.append({
                        "pair": pair, "group": group_name, "signal_time": row["dt"],
                        "entry_price": row["close"],
                        "atr_pct_at_signal": row.get("atr_pct", None),
                        "pnl": None, "exit_reason": None,
                    })
        except Exception:
            continue

    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)

    now = pd.Timestamp.now(tz="UTC")
    missing = log[log["pnl"].isna() & (log["exit_reason"] != "unresolvable_rolled_off")]
    for idx, row in missing.iterrows():
        elapsed_hours = (now - row["signal_time"]).total_seconds() / 3600
        fp = f"data/candles/{row['pair'].replace('/', '_')}_5m_features.csv"
        try:
            df = pd.read_csv(fp)
            df = parse_dt_column(df)
            df = prep_columns(df)
            match = df[df["dt"] == row["signal_time"]]
            if len(match) == 0:
                if elapsed_hours > ROLLOFF_HOURS:
                    log.at[idx, "exit_reason"] = "unresolvable_rolled_off"
                continue
            pos = match.index[0]
            out = simulate_sniper(df, pos)
            if out is None:
                continue
            pnl, reason = out
            if reason == "end_data" and elapsed_hours < RESOLVE_BUFFER_HOURS:
                continue
            log.at[idx, "pnl"] = pnl
            log.at[idx, "exit_reason"] = reason
        except Exception:
            continue

    log.to_csv(LOG_PATH, index=False)
    resolved = log["pnl"].notna().sum()
    unresolvable = (log["exit_reason"] == "unresolvable_rolled_off").sum()
    print(f"Log-Groesse gesamt: {len(log)}  aufgeloest: {resolved}  unresolvable: {unresolvable}  neu heute: {len(new_rows)}")
    for grp in ["allowed", "near_miss"]:
        g = log[(log["group"] == grp) & log["pnl"].notna()]
        print(f"  {grp}: n={len(g)}", f"Netto-Mittel={g['pnl'].mean()-0.15:+.4f}%" if len(g) else "")


if __name__ == "__main__":
    main()
