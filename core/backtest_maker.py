"""
ENDVISION-1h — Maker-Backtest (Limit-Entry statt Market).

Validiert den Kern-Umbau: statt am Markt zu kaufen (Taker, Spread + Slippage),
wird eine Limit-Order am Signal-Preis platziert. Vorteile: kein Spread, kein
Slippage, planbarer Fuellpreis. Preis dafuer: manche Orders werden NICHT
gefuellt (Preis laeuft weg) -> Trade verpasst.

Fokus auf die zwei Strategien mit robuster Kante im Taker-Backtest:
rsi_cross und pullback.

Modell:
  - Signal bei Kerze i -> Limit-Buy bei close[i]
  - Fill, wenn in den naechsten FILL_WINDOW Kerzen low <= limit_price
    (Preis kommt auf den Limit zurueck). Sonst: verpasst.
  - Kosten: COST_MAKER (2x Maker-Gebuehr, kein Spread/Slippage)
  - Exit: ATR-Trail wie im Taker-Backtest (portfolio_1h.Trade)

Aufruf (im trading-system, damit .venv + Daten da sind):
  .venv/bin/python3 /root/TradeBot/core/backtest_maker.py \
      /root/TradeBot/config/config_1h.yaml data/candles
"""
from __future__ import annotations
import sys, glob, os, warnings
warnings.filterwarnings("ignore")
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.portfolio_1h import Trade

COST_MAKER = 0.20      # Round-Trip: 2x 0.1% Binance-Spot-Maker, KEIN Spread/Slippage
COST_TAKER = 0.235     # Referenz, zum Vergleich
FILL_WINDOW = 2        # so viele Kerzen hat die Limit-Order Zeit, gefuellt zu werden
STRATS = ["rsi_cross", "pullback"]


def run(cfg, candles_dir):
    ex = cfg["exit"]
    files = sorted(glob.glob(os.path.join(candles_dir, "*_1h_features.csv")))
    trades = []
    signals_total = 0
    filled_total = 0

    for fp in files:
        pair = os.path.basename(fp).replace("_1h_features.csv", "")
        try:
            d = pd.read_csv(fp)
        except Exception:
            continue
        if len(d) < 60:
            continue
        for c in ["close", "high", "low", "atr_pct"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d["_ts"] = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
        d = d.sort_values("_ts").reset_index(drop=True)
        n = len(d)
        close = d["close"].values; high = d["high"].values; low = d["low"].values
        atrp = d["atr_pct"].values; ts = d["_ts"].astype(str).values

        for strat in STRATS:
            col = f"signal_{strat}"
            if col not in d.columns:
                continue
            sig = d[col].fillna(False).astype(bool).values
            for i in range(n):
                if not sig[i]:
                    continue
                if i + 1 + ex["max_hold_bars"] >= n:
                    continue
                atr = atrp[i]
                limit_price = close[i]
                if not (atr > 0 and limit_price > 0):
                    continue
                signals_total += 1

                # --- Fill-Simulation: Limit-Buy wird gefuellt, wenn Preis zurueckkommt
                fill_bar = None
                for j in range(1, FILL_WINDOW + 1):
                    if i + j >= n:
                        break
                    if low[i + j] <= limit_price:
                        fill_bar = i + j
                        break
                if fill_bar is None:
                    continue          # Order nicht gefuellt -> Trade verpasst
                filled_total += 1

                # --- Trade ab fill_bar mit ATR-Trail-Exit, Maker-Kosten
                tr = Trade(
                    pair=pair, strategy=strat, entry_ts=ts[fill_bar],
                    entry_price=float(limit_price), atr_pct=float(atr), stake_usdc=0.0,
                    init_stop_atr=float(ex["init_stop_atr"]),
                    arm_mfe_atr=float(ex["arm_mfe_atr"]),
                    trail_atr=float(ex["trail_atr"]),
                    max_hold_bars=int(ex["max_hold_bars"]),
                )
                k = fill_bar + 1
                closed = False
                while k < n and tr.bars_held < ex["max_hold_bars"]:
                    closed = tr.update_bar(high[k], low[k], close[k], ts[k], COST_MAKER)
                    if closed:
                        break
                    k += 1
                if not closed:
                    cl = (close[min(k, n - 1)] / tr.entry_price - 1.0) * 100.0
                    tr._close(cl, ts[min(k, n - 1)], "eod", COST_MAKER)
                trades.append(tr)

    return trades, signals_total, filled_total


def report(trades, sig_total, fill_total):
    rows = [{"pair": t.pair, "strategy": t.strategy, "entry_ts": t.entry_ts,
             "gross_pct": t.gross_pct, "net_pct": t.net_pct} for t in trades]
    r = pd.DataFrame(rows)
    r["_dt"] = pd.to_datetime(r["entry_ts"], errors="coerce", utc=True)
    fr = fill_total / sig_total * 100 if sig_total else 0

    print("=" * 84)
    print(f"ENDVISION-1h MAKER-BACKTEST  (Limit-Entry, Maker-Kosten {COST_MAKER}%, kein Slippage)")
    print(f"Signale={sig_total}  gefuellt={fill_total}  Fill-Rate={fr:.1f}%  Trades={len(r)}")
    print(f"Vergleich: Taker war {COST_TAKER}% + Slippage; Maker ist {COST_MAKER}% fix")
    print("=" * 84)

    def blk(df, name):
        n = len(df); g = df["net_pct"]
        rs = np.sort(g.values)[::-1]; wo5 = rs[5:].mean() if n > 5 else g.mean()
        flag = "" if n >= 30 else "  << n<30"
        mark = "  <== POSITIV & robust" if g.mean() > 0 and wo5 > 0 and n >= 30 else ""
        print(f"\n### {name}  n={n}{flag}")
        print(f"  net-mean={g.mean():+.4f}%  net-median={g.median():+.4f}%  "
              f"net-WR={(g>0).mean()*100:.1f}%  ohneT5={wo5:+.4f}%{mark}")
        # Monats-Split
        df = df.copy(); df["m"] = df["_dt"].dt.strftime("%Y-%m")
        months = sorted(x for x in df["m"].dropna().unique())
        cells = []
        for m in months:
            s = df[df["m"] == m]
            if len(s) >= 15:
                cells.append(f"{m}:{s['net_pct'].mean():+.3f}")
        if cells:
            print("  Monate (n>=15): " + "  ".join(cells))

    blk(r, "BEIDE (rsi_cross + pullback)")
    for name, grp in sorted(r.groupby("strategy"), key=lambda x: -len(x[1])):
        blk(grp, str(name))
    print("\nLesart: net-mean UND ohneT5 positiv + ueber die Monate stabil + Fill-Rate")
    print("hoch genug (>~40%) = der Maker-Umbau traegt. Dann lohnt der Live-Bau.")
    return r


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/config_1h.yaml"
    candles = sys.argv[2] if len(sys.argv) > 2 else "data/candles"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    trades, st, ft = run(cfg, candles)
    report(trades, st, ft)
