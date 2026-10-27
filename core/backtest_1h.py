"""
ENDVISION-1h — Backtest-Engine.

Liest *_1h_features.csv (erzeugt von build_features_tf.py --tf 1h im 5m-System),
nimmt die signal_*-Spalten als Entry-Trigger (kein Signal-Nachbau — die bewaehrte
feature_engine bleibt die Quelle), simuliert jeden Trade Bar-fuer-Bar mit dem
ATR-Exit aus portfolio_1h.py und zieht die echten Kosten ab.

Ausgabe: dieselbe Auswertung, mit der wir das 5m-System beurteilt haben —
pro Strategie net-mean/WR/G-L + Monats-Split (haelt es ueber Phasen?).

Aufruf:  python3 core/backtest_1h.py [config/config_1h.yaml]
"""
from __future__ import annotations
import sys, glob, os, warnings
warnings.filterwarnings("ignore")
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.portfolio_1h import Trade, position_size


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def backtest(cfg):
    C = float(cfg["costs"]["round_trip_pct"])
    ex = cfg["exit"]
    strategies = list(cfg["enabled_strategies"])
    cand_dir = cfg["data"]["candles_dir"]
    suffix = cfg["data"]["features_suffix"]
    min_rows = int(cfg["backtest"]["min_rows_per_pair"])
    risk = cfg["risk"]

    files = sorted(glob.glob(os.path.join(cand_dir, "*" + suffix)))
    all_trades = []
    pairs_used = 0

    for fp in files:
        pair = os.path.basename(fp).replace(suffix, "")
        try:
            d = pd.read_csv(fp)
        except Exception:
            continue
        if len(d) < min_rows:
            continue
        for c in ["close", "high", "low", "atr_pct"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        if "timestamp" in d.columns:
            d["_ts"] = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
        else:
            d["_ts"] = pd.RangeIndex(len(d))
        d = d.sort_values("_ts").reset_index(drop=True)

        sig_cols = {s: f"signal_{s}" for s in strategies if f"signal_{s}" in d.columns}
        if not sig_cols:
            continue
        pairs_used += 1

        open_trades = []
        n = len(d)
        for i in range(n):
            row = d.iloc[i]
            # 1) offene Trades mit dieser Kerze fortschreiben
            still = []
            for tr in open_trades:
                closed = tr.update_bar(row["high"], row["low"], row["close"],
                                       str(row["_ts"]), C)
                if closed:
                    all_trades.append(tr)
                else:
                    still.append(tr)
            open_trades = still

            # 2) neue Signale -> Entry (ab naechster Kerze gelebt; Entry = close hier)
            if i + 1 >= n:
                continue
            atr = row.get("atr_pct", np.nan)
            if not (isinstance(atr, (int, float)) and atr > 0):
                continue
            e = row["close"]
            if not (e > 0):
                continue
            for strat, col in sig_cols.items():
                val = row.get(col, False)
                fired = bool(val) if not pd.isna(val) else False
                if not fired:
                    continue
                if len(open_trades) >= int(risk["max_open_trades"]):
                    break
                # nur ein offener Trade pro (pair, strategy) gleichzeitig
                if any(t.pair == pair and t.strategy == strat for t in open_trades):
                    continue
                stake = position_size(float(risk["start_capital_usdc"]),
                                      float(risk["risk_per_trade_pct"]),
                                      float(ex["init_stop_atr"]), float(atr))
                tr = Trade(
                    pair=pair, strategy=strat, entry_ts=str(row["_ts"]),
                    entry_price=float(e), atr_pct=float(atr), stake_usdc=stake,
                    init_stop_atr=float(ex["init_stop_atr"]),
                    arm_mfe_atr=float(ex["arm_mfe_atr"]),
                    trail_atr=float(ex["trail_atr"]),
                    max_hold_bars=int(ex["max_hold_bars"]),
                )
                open_trades.append(tr)

        # offene Trades am Ende der Historie: zum letzten Close schliessen
        last = d.iloc[-1]
        for tr in open_trades:
            cl = (last["close"] / tr.entry_price - 1.0) * 100.0
            tr._close(cl, str(last["_ts"]), "eod_backtest", C)
            all_trades.append(tr)

    return all_trades, pairs_used


def report(trades, C):
    if not trades:
        print("Keine Trades erzeugt — Signale/Spalten pruefen.")
        return
    rows = [{
        "pair": t.pair, "strategy": t.strategy, "entry_ts": t.entry_ts,
        "net_pct": t.net_pct, "gross_pct": t.gross_pct,
        "mfe_pct": t.mfe_pct, "mae_pct": t.mae_pct,
        "exit_reason": t.exit_reason,
    } for t in trades]
    r = pd.DataFrame(rows)
    r["_dt"] = pd.to_datetime(r["entry_ts"], errors="coerce", utc=True)

    def blk(df, name):
        n = len(df); g = df["net_pct"]
        wg = df["gross_pct"][df["gross_pct"] > 0]; lg = df["gross_pct"][df["gross_pct"] <= 0]
        gl = (wg.median() / abs(lg.median())) if len(wg) and len(lg) and lg.median() != 0 else float("nan")
        rs = np.sort(g.values)[::-1]; wo5 = rs[5:].mean() if n > 5 else g.mean()
        flag = "" if n >= 30 else "  << n<30"
        mark = "  <== NETTO POSITIV" if g.mean() > 0 and n >= 30 else ""
        print(f"\n### {name}  n={n}{flag}")
        print(f"  net-mean={g.mean():+.4f}%  net-median={g.median():+.4f}%  "
              f"net-WR={(g>0).mean()*100:.1f}%  G/L(brutto)={gl:.3f}  ohneT5={wo5:+.4f}%{mark}")

    print("=" * 80)
    print(f"ENDVISION-1h BACKTEST  Trades={len(r)}  Kosten c={C}%")
    print(f"Spanne: {r['_dt'].min()} -> {r['_dt'].max()}")
    print("=" * 80)
    blk(r, "ALLE")
    for name, grp in sorted(r.groupby("strategy"), key=lambda x: -len(x[1])):
        blk(grp, str(name))

    # Monats-Split
    print("\n" + "=" * 80)
    print("MONATS-SPLIT (net-mean pro Strategie) — haelt es ueber Phasen?")
    r["m"] = r["_dt"].dt.strftime("%Y-%m")
    months = sorted(x for x in r["m"].dropna().unique())
    head = "  {:14s} | ".format("Strategie") + " | ".join(m.rjust(9) for m in months) + " |   gesamt"
    print(head)
    print("  " + "-" * (len(head)))
    for name in list(r["strategy"].value_counts().index):
        cells = []
        for m in months:
            sub = r[(r["strategy"] == name) & (r["m"] == m)]
            cells.append((f"{sub['net_pct'].mean():+.3f}" if len(sub) >= 15 else f"n{len(sub)}").rjust(9))
        tot = r[r["strategy"] == name]["net_pct"].mean()
        print(f"  {name:14s} | " + " | ".join(cells) + f" | {tot:+.4f}")
    return r


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/config_1h.yaml"
    cfg = load_config(cfg_path)
    if len(sys.argv) > 2:                      # candles_dir per CLI ueberschreibbar
        cfg["data"]["candles_dir"] = sys.argv[2]
    C = float(cfg["costs"]["round_trip_pct"])
    trades, pairs = backtest(cfg)
    print(f"\nPairs genutzt: {pairs}")
    r = report(trades, C)
    if r is not None:
        out = cfg["data"]["trades_log"].replace(".csv", "_backtest.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        r.to_csv(out, index=False)
        print(f"\nTrade-Log geschrieben: {out}")
