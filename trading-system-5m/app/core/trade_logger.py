import csv
from pathlib import Path
from datetime import datetime, timezone

_CSV_PATH = Path("/root/trading-system/data/history/trades.csv")
_FIELDS = [
    "trade_id","pair","strategy","regime","execution_mode",
    "entry_time_utc","entry_price","stake_usdc",
    # Entry Features — ML Input
    "entry_rsi_14","entry_vol_ratio","entry_vpin",
    "entry_adx","entry_dif","entry_dea",
    "entry_bb_percent","entry_above_vwap","entry_bullish_wick",
    "entry_ofi","entry_atr_pct",
    "entry_adx_trend","entry_adx_range",
    "entry_l3_context","entry_btc_corr", "entry_htf_bullish","entry_htf_15m_bullish","entry_is_decorr","entry_is_counter",
    "entry_hmm_regime","entry_hmm_conf",
    # Close
    "close_time_utc","close_price","exit_reason","pnl_pct","hold_minutes",
    "hard_stop_price_at_exit","break_even_set_at_exit",
    "highest_price_at_exit","lowest_price_at_exit",
    "exit_open","exit_high","exit_low","exit_close",
    # ML Label (wird beim Close gesetzt)
    "ml_label",
]

def _ensure_file():
    _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _CSV_PATH.exists():
        with open(_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_FIELDS).writeheader()
    else:
        # Bestehende CSV: fehlende Spalten ergänzen
        with open(_CSV_PATH, "r", newline="", encoding="utf-8") as f:
            existing = f.readline().strip().split(",")
        missing = [c for c in _FIELDS if c not in existing]
        if missing:
            import tempfile, shutil
            rows = []
            with open(_CSV_PATH, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for m in missing:
                        row[m] = ""
                    rows.append(row)
            with open(_CSV_PATH, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=_FIELDS)
                w.writeheader()
                w.writerows(rows)

def log_open(trade: dict):
    _ensure_file()
    row = {f: "" for f in _FIELDS}
    row.update({
        "trade_id":          trade.get("trade_id",""),
        "pair":              trade.get("pair",""),
        "strategy":          trade.get("strategy",""),
        "regime":            trade.get("regime",""),
        "execution_mode":    trade.get("execution_mode",""),
        "entry_time_utc":    trade.get("entry_time_utc",""),
        "entry_price":       trade.get("entry_price",""),
        "stake_usdc":        trade.get("stake_usdc",""),
        "entry_rsi_14":      trade.get("entry_rsi_14",""),
        "entry_vol_ratio":   trade.get("entry_vol_ratio",""),
        "entry_vpin":        trade.get("entry_vpin",""),
        "entry_adx":         trade.get("entry_adx",""),
        "entry_dif":         trade.get("entry_dif",""),
        "entry_dea":         trade.get("entry_dea",""),
        "entry_bb_percent":  trade.get("entry_bb_percent",""),
        "entry_above_vwap":  trade.get("entry_above_vwap",""),
        "entry_bullish_wick": trade.get("entry_bullish_wick",""),
        "entry_ofi":         trade.get("entry_ofi",""),
        "entry_atr_pct":     trade.get("entry_atr_pct",""),
        "entry_adx_trend":   trade.get("entry_adx_trend",""),
        "entry_adx_range":   trade.get("entry_adx_range",""),
        "entry_l3_context":  trade.get("entry_l3_context",""),
        "entry_btc_corr":    trade.get("entry_btc_corr",""),
        "entry_htf_bullish":     trade.get("entry_htf_bullish",""),
        "entry_htf_15m_bullish": trade.get("entry_htf_15m_bullish",""),
        "entry_is_decorr":       trade.get("entry_is_decorr",""),
        "entry_is_counter":      trade.get("entry_is_counter",""),
        "entry_hmm_regime":  trade.get("entry_hmm_regime",""),
        "entry_hmm_conf":    trade.get("entry_hmm_conf",""),
    })
    with open(_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_FIELDS).writerow(row)

def log_close(trade: dict):
    _ensure_file()
    rows = []
    updated = False
    tid  = str(trade.get("trade_id",""))
    pair = str(trade.get("pair",""))
    with open(_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (str(row.get("trade_id","")) == tid
                    and str(row.get("pair","")) == pair
                    and row.get("close_time_utc","") == ""):
                pnl = float(trade.get("pnl_pct", 0.0) or 0.0)
                row["close_time_utc"] = trade.get("close_time_utc","")
                row["close_price"]    = trade.get("close_price","")
                row["exit_reason"]    = trade.get("exit_reason","")
                row["pnl_pct"]        = round(pnl, 4)
                row["hold_minutes"]   = round(float(trade.get("hold_minutes", 0.0) or 0.0), 1)
                row["hard_stop_price_at_exit"]  = trade.get("hard_stop_price_at_exit", "")
                row["break_even_set_at_exit"]   = trade.get("break_even_set_at_exit", "")
                row["highest_price_at_exit"]    = trade.get("highest_price_at_exit", "")
                row["lowest_price_at_exit"]     = trade.get("lowest_price_at_exit", "")
                row["exit_open"]   = trade.get("exit_open", "")
                row["exit_high"]   = trade.get("exit_high", "")
                row["exit_low"]    = trade.get("exit_low", "")
                row["exit_close"]  = trade.get("exit_close", "")
                row["ml_label"]       = 1 if pnl > 0.0 else 0
            rows.append(row)
    if not rows:
        return
    with open(_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)
