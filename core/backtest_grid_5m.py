"""
ENDVISION — 5m Range-Scalp / Grid-Zellen-Backtest mit Regime-Filter (Spot-only, Long).

Letzte ungetestete Spot-Idee: Grid/Range-Market-Making bei feiner Granularitaet.
Hypothese: in ruhigen RANGE-Phasen (nicht-trendend) gibt es intra-day-Oszillation,
die ein MAKER-Grid netto nach Gebuehren einfangen kann. Der 1h-Test hatte KEINEN
Regime-Filter und scheiterte (Random Walk). Hier ist der Filter der Kern.

Mechanik je Scalp (Maker auf beiden Beinen):
  - Trigger-Bar i: Limit-Buy bei close[i]*(1-off).
  - Fill: low<=buy innerhalb W Bars, Penetration>=MIN_PEN. Sonst kein Trade.
  - Exit pessimistisch je Bar: erst Stop (low<=entry*(1-stop)) als TAKER-Ausstieg,
    dann Take-Profit (high>=entry*(1+tp)) als MAKER, sonst Zeit-Exit zum close.
  - Kosten: Maker-Leg 0.10%. TP-Pfad=0.10+0.10. Stop/Zeit-Pfad=0.10+0.15 (Taker-Exit).

Regime-Filter = Kaufman Efficiency Ratio ueber N Bars (nur Vergangenheit):
  ER = |close[i]-close[i-N]| / sum(|diff|) der letzten N Bars.
  ER~1 = Trend, ER~0 = ranging. Range-Regime = ER <= ER_MAX.

Drei Varianten, GLEICHE Trade-Zahl je Pair (fairer Vergleich):
  GEFILTERT = nur Range-Bars (ER<=ER_MAX)
  IMMER-AN  = gleich viele, gleichmaessig ueber ALLE Bars verteilt (Filter aus)
  ZUFALL    = gleich viele Zufalls-Bars
Taugt nur, wenn GEFILTERT positiv & robust ist UND klar besser als IMMER-AN und ZUFALL.

Aufruf auf dem Server:
  .venv/bin/python3 /root/TradeBot/core/backtest_grid_5m.py data/candles 5m
"""
from __future__ import annotations
import sys, glob, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

CANDLES = sys.argv[1] if len(sys.argv) > 1 else "data/candles"
TF = sys.argv[2] if len(sys.argv) > 2 else "5m"

W = 2                 # Fill-Fenster (Bars)
MIN_PEN = 0.05        # Entry-Fill nur bei >=0.05% Penetration (keine Wick-Fantasie)
MAKER = 0.10          # je Maker-Leg (%)
TAKER = 0.15          # Taker-Exit-Bein (Stop/Zeit) (%)
ER_N = 24             # Fenster fuer die Efficiency Ratio (24*5min = 2h)
ER_MAX = 0.30         # Range-Regime, wenn ER <= ER_MAX
CAP_PER_PAIR = 1500   # max. Trigger je Pair (Laufzeit zaehmen)

GRID = [
    dict(off=0.0005, tp=0.002, stop=0.004, hold=12),
    dict(off=0.001,  tp=0.003, stop=0.006, hold=12),
    dict(off=0.001,  tp=0.004, stop=0.008, hold=24),
    dict(off=0.002,  tp=0.005, stop=0.010, hold=24),
]


def load():
    pat_feat = os.path.join(CANDLES, f"*_{TF}_features.csv")
    pat_raw = os.path.join(CANDLES, f"*_{TF}.csv")
    files = sorted(glob.glob(pat_feat)) or sorted(
        f for f in glob.glob(pat_raw) if not f.endswith(f"_{TF}_features.csv"))
    P = {}
    for fp in files:
        pair = os.path.basename(fp).replace(f"_{TF}_features.csv", "").replace(f"_{TF}.csv", "")
        try:
            d = pd.read_csv(fp)
        except Exception:
            continue
        if len(d) < ER_N + 60 or "close" not in d.columns:
            continue
        for c in ["close", "high", "low"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.sort_values("timestamp").reset_index(drop=True)
        ms = 1000 if d["timestamp"].median() < 1e12 else 1
        ym = pd.to_datetime(d["timestamp"].astype("int64") * ms, unit="ms", utc=True).dt.strftime("%Y-%m").values
        close = d["close"].values.astype(float)
        # Efficiency Ratio (nur Vergangenheit bis i)
        diff = np.abs(np.diff(close, prepend=close[0]))
        roll_vol = pd.Series(diff).rolling(ER_N).sum().values
        net_move = np.abs(close - np.concatenate([np.full(ER_N, np.nan), close[:-ER_N]]))
        with np.errstate(invalid="ignore", divide="ignore"):
            er = np.where(roll_vol > 0, net_move / roll_vol, np.nan)
        P[pair] = dict(close=close, high=d["high"].values.astype(float),
                       low=d["low"].values.astype(float), ym=ym, er=er, n=len(d))
    return P


def scalp(pd_, i, par):
    close, high, low, n = pd_["close"], pd_["high"], pd_["low"], pd_["n"]
    if i + W + par["hold"] + 1 >= n or i < ER_N or close[i] <= 0:
        return None
    buy = close[i] * (1 - par["off"])
    fb = None
    for j in range(1, W + 1):
        if low[i + j] <= buy and (buy - low[i + j]) / buy * 100 >= MIN_PEN:
            fb = i + j; break
    if fb is None:
        return None
    entry = buy; tp = entry * (1 + par["tp"]); sl = entry * (1 - par["stop"])
    for k in range(fb + 1, min(fb + 1 + par["hold"], n)):
        if low[k] <= sl:
            return (-par["stop"] * 100 - MAKER - TAKER, "stop", pd_["ym"][k])
        if high[k] >= tp:
            return (par["tp"] * 100 - MAKER - MAKER, "tp", pd_["ym"][k])
    kk = min(fb + par["hold"], n - 1)
    return ((close[kk] / entry - 1) * 100 - MAKER - TAKER, "time", pd_["ym"][kk])


def triggers(pd_, mode, rng):
    """Index-Mengen je Variante, GLEICHE Anzahl (= Range-Bar-Zahl, gedeckelt)."""
    n = pd_["n"]; lo, hi = ER_N, n - W - 30
    if hi <= lo:
        return np.array([], int)
    valid = np.arange(lo, hi)
    rng_bars = valid[pd_["er"][valid] <= ER_MAX]
    cnt = min(len(rng_bars), CAP_PER_PAIR)
    if cnt == 0:
        return np.array([], int)
    if mode == "gefiltert":
        return rng_bars if len(rng_bars) <= cnt else rng_bars[np.linspace(0, len(rng_bars) - 1, cnt).astype(int)]
    if mode == "immer":
        return valid[np.linspace(0, len(valid) - 1, cnt).astype(int)]
    return rng.choice(valid, size=cnt, replace=False)  # zufall


def run(P, par, mode, rng):
    rows = []
    for pd_ in P.values():
        for i in triggers(pd_, mode, rng):
            r = scalp(pd_, int(i), par)
            if r:
                rows.append(r)
    if not rows:
        return None
    net = np.array([r[0] for r in rows]); outs = [r[1] for r in rows]; yms = [r[2] for r in rows]
    rs = np.sort(net)[::-1]; wo5 = rs[5:].mean() if len(net) > 5 else net.mean()
    mdf = pd.DataFrame({"ym": yms, "net": net})
    g = mdf.groupby("ym")["net"].agg(["mean", "count"]); g = g[g["count"] >= 20]
    return dict(n=len(net), mean=net.mean(), wo5=wo5, wr=(net > 0).mean() * 100,
                tp=outs.count("tp") / len(outs) * 100, sl=outs.count("stop") / len(outs) * 100,
                mpos=int((g["mean"] > 0).sum()), mtot=int(len(g)))


def main():
    P = load()
    print("=" * 100)
    print(f"5m RANGE-SCALP + REGIME-FILTER  (Spot-only Long, Maker {MAKER}%/Leg, Taker-Exit {TAKER}%)")
    print(f"{len(P)} Pairs, TF={TF}.  Efficiency Ratio ueber {ER_N} Bars, Range=ER<={ER_MAX}")
    print("=" * 100)
    if not P:
        print(f"Keine {TF}-Dateien in {CANDLES} gefunden (gesucht: *_{TF}_features.csv / *_{TF}.csv).")
        return
    rng = np.random.default_rng(11)
    for par in GRID:
        print(f"\n### off={par['off']*100:.2f}%  tp={par['tp']*100:.1f}%  stop={par['stop']*100:.1f}%  hold={par['hold']}")
        for name, mode in [("GEFILTERT", "gefiltert"), ("IMMER-AN ", "immer"), ("ZUFALL   ", "zufall")]:
            r = run(P, par, mode, rng)
            if not r:
                print(f"    {name}: keine Trades"); continue
            robust = "  <== positiv & robust" if (r["mean"] > 0 and r["wo5"] > 0 and r["n"] >= 50) else ""
            print(f"    {name}: n={r['n']:6d}  net-mean={r['mean']:+.4f}%  ohneT5={r['wo5']:+.4f}%  "
                  f"WR={r['wr']:.1f}%  TP%={r['tp']:.0f}  Stop%={r['sl']:.0f}  Monate+={r['mpos']}/{r['mtot']}{robust}")
    print("\n" + "=" * 100)
    print("LESART: Nur wenn GEFILTERT positiv & robust ist (net-mean UND ohneT5 > 0, Monate stabil)")
    print("UND GEFILTERT klar IMMER-AN und ZUFALL schlaegt, traegt der Range-Ansatz. Ist GEFILTERT")
    print("~ IMMER-AN ~ ZUFALL oder negativ -> 5m ist derselbe Random Walk wie 1h, Spot-only ist aus.")


if __name__ == "__main__":
    main()
