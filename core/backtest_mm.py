"""
ENDVISION-1h — Market-Making / Mean-Reversion-Scalp-Backtest (Spot-ehrlich).

Idee (nutzt die These 'Signal = wann, nicht wohin'): an einem Signal-Moment
NICHT auf Richtung wetten, sondern einen Dip kaufen (Limit unter close) und den
Bounce verkaufen (Limit ueber Entry). Beide Legs sind Maker -> guenstig. Der
Verlustfall ist Adverse Selection: nach dem Kauf laeuft es weiter runter -> Stop
(Taker). Spot kann nicht shorten, daher nur die Long-Seite.

EHRLICH modelliert:
  - Entry: Limit-Buy bei close[i]*(1-off). Fuellt, wenn low<=limit (+Penetration)
    in den naechsten W Bars. Sonst: kein Trade.
  - Danach je Bar PESSIMISTISCH: erst Stop (low<=entry*(1-stop)) als TAKER,
    dann Take-Profit (high>=entry*(1+tp)) als MAKER, sonst Zeit-Exit zum close.
  - Kosten: Maker-Leg 0.10%, Taker-Leg 0.15%.
  - KONTROLLE: derselbe Scalp an ZUFALLS-Bars. Nur wenn die Signal-Variante die
    Zufalls-Variante schlaegt, traegt das Timing der Indikatoren ueberhaupt etwas.

Aufruf im trading-system:
  .venv/bin/python3 /root/TradeBot/core/backtest_mm.py data/candles
"""
from __future__ import annotations
import sys, glob, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

CANDLES = sys.argv[1] if len(sys.argv) > 1 else "data/candles"
STRATS = ["rsi_cross", "pullback"]
W = 2                      # Fenster fuer den Entry-Fill (Bars)
MIN_PEN = 0.05             # Entry-Fill nur bei >=0.05% Penetration (keine Wick-Fantasie)
COST_MAKER = 0.10          # je Maker-Leg (Entry, TP)
COST_TAKER = 0.15          # je Taker-Leg (Stop, Zeit-Exit)

# kleine, bewusst sinnvolle Parameter-Gitter (kein wildes Sweepen):
# off=Dip-Tiefe, tp=Ziel ueber Entry, stop=max. Gegenbewegung, hold=max Bars
GRID = [
    dict(off=0.001, tp=0.003, stop=0.008, hold=8),
    dict(off=0.001, tp=0.005, stop=0.010, hold=8),
    dict(off=0.003, tp=0.005, stop=0.010, hold=8),
    dict(off=0.003, tp=0.008, stop=0.012, hold=12),
]


def load():
    P = {}
    for fp in sorted(glob.glob(os.path.join(CANDLES, "*_1h_features.csv"))):
        pair = os.path.basename(fp).replace("_1h_features.csv", "")
        try:
            d = pd.read_csv(fp)
        except Exception:
            continue
        if len(d) < 80 or "close" not in d.columns:
            continue
        for c in ["close", "high", "low"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.sort_values("timestamp").reset_index(drop=True)
        ms = 1000 if d["timestamp"].median() < 1e12 else 1
        ym = pd.to_datetime(d["timestamp"].astype("int64") * ms, unit="ms", utc=True).dt.strftime("%Y-%m").values
        trig = {s: (np.where(d[f"signal_{s}"].fillna(False).astype(bool).values)[0]
                    if f"signal_{s}" in d.columns else np.array([], int)) for s in STRATS}
        P[pair] = dict(close=d["close"].values, high=d["high"].values,
                       low=d["low"].values, ym=ym, trig=trig, n=len(d))
    return P


def scalp_one(pd_, i, par):
    """Ein Scalp ab Signal-Bar i. Gibt (net_pct, ausgang, ym) oder None (kein Fill)."""
    close, high, low, n = pd_["close"], pd_["high"], pd_["low"], pd_["n"]
    if i + W + par["hold"] + 1 >= n or close[i] <= 0:
        return None
    buy = close[i] * (1 - par["off"])
    # Entry-Fill suchen
    fb = None
    for j in range(1, W + 1):
        if low[i + j] <= buy and (buy - low[i + j]) / buy * 100 >= MIN_PEN:
            fb = i + j; break
    if fb is None:
        return None
    entry = buy
    tp = entry * (1 + par["tp"]); sl = entry * (1 - par["stop"])
    for k in range(fb + 1, min(fb + 1 + par["hold"], n)):
        if low[k] <= sl:                      # pessimistisch: Stop zuerst (Taker)
            return (-par["stop"] * 100 - COST_MAKER - COST_TAKER, "stop", pd_["ym"][k])
        if high[k] >= tp:                     # Take-Profit (Maker)
            return (par["tp"] * 100 - COST_MAKER - COST_MAKER, "tp", pd_["ym"][k])
    kk = min(fb + par["hold"], n - 1)         # Zeit-Exit zum close (Taker)
    return ((close[kk] / entry - 1) * 100 - COST_MAKER - COST_TAKER, "time", pd_["ym"][kk])


def run(P, par, mode, rng):
    rows = []
    for pair, pd_ in P.items():
        if mode == "signal":
            idx = np.unique(np.concatenate([pd_["trig"][s] for s in STRATS])) if pd_["trig"] else np.array([], int)
        else:  # random: gleich viele Zufalls-Bars wie Signale in diesem Pair
            cnt = sum(len(pd_["trig"][s]) for s in STRATS)
            hi = pd_["n"] - W - par["hold"] - 2
            idx = rng.integers(30, hi, size=cnt) if (cnt > 0 and hi > 30) else np.array([], int)
        for i in idx:
            r = scalp_one(pd_, int(i), par)
            if r:
                rows.append(r)
    if not rows:
        return None
    net = np.array([r[0] for r in rows]); outs = [r[1] for r in rows]; yms = [r[2] for r in rows]
    rs = np.sort(net)[::-1]; wo5 = rs[5:].mean() if len(net) > 5 else net.mean()
    tp_rate = outs.count("tp") / len(outs) * 100
    sl_rate = outs.count("stop") / len(outs) * 100
    mdf = pd.DataFrame({"ym": yms, "net": net})
    mon = mdf.groupby("ym")["net"].agg(["mean", "count"])
    mon = mon[mon["count"] >= 20]
    mpos = int((mon["mean"] > 0).sum()); mtot = int(len(mon))
    return dict(n=len(net), mean=net.mean(), wo5=wo5, wr=(net > 0).mean() * 100,
                tp=tp_rate, sl=sl_rate, mpos=mpos, mtot=mtot)


def main():
    P = load()
    print("=" * 92)
    print("MARKET-MAKING / MEAN-REVERSION-SCALP  (Spot: nur Long, Dip kaufen / Bounce verkaufen)")
    print(f"{len(P)} Pairs.  Maker-Leg {COST_MAKER}%  Taker-Leg {COST_TAKER}%  Fill-Fenster {W} Bars")
    print("=" * 92)
    rng = np.random.default_rng(7)
    for par in GRID:
        print(f"\n### off={par['off']*100:.1f}%  tp={par['tp']*100:.1f}%  "
              f"stop={par['stop']*100:.1f}%  hold={par['hold']}")
        sig = run(P, par, "signal", rng)
        ran = run(P, par, "random", rng)
        for name, r in [("SIGNAL", sig), ("ZUFALL", ran)]:
            if not r:
                print(f"    {name}: keine Trades"); continue
            robust = "  <== positiv & robust" if (r["mean"] > 0 and r["wo5"] > 0 and r["n"] >= 50) else ""
            print(f"    {name}: n={r['n']:5d}  net-mean={r['mean']:+.4f}%  ohneT5={r['wo5']:+.4f}%  "
                  f"WR={r['wr']:.1f}%  TP%={r['tp']:.0f}  Stop%={r['sl']:.0f}  Monate+={r['mpos']}/{r['mtot']}{robust}")
    print("\n" + "=" * 92)
    print("LESART: Eine Zeile taugt nur, wenn SIGNAL positiv & robust ist UND die")
    print("ZUFALL-Kontrolle klar schlaegt. Ist ZUFALL genauso gut -> das Signal-Timing")
    print("bringt nichts, es ist blosses Scalpen. Ist auch SIGNAL negativ -> auch MM")
    print("rettet diese Signale nicht, und wir muessen den Trigger selbst neu denken.")


if __name__ == "__main__":
    main()
