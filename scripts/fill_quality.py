#!/usr/bin/env python3
"""
ENDVISION-1h — Fill-Qualitaets-Test fuer den Maker/Limit-Entry.

Hintergrund: Der Maker-Backtest fuellt eine Limit-Buy bei close[i], sobald in
den naechsten Kerzen low <= limit ("touch"). Mechanisch ist das korrekt: eine
Limit-Order fuellt, sobald der Markt den Preis handelt. ABER: wenn der Preis das
Limit nur mit einem 1-Tick-Docht kuesst und dort niemand verkauft, bekommt man
live keinen Fill. Die ganze Kante koennte aus solchen fragilen Wick-Fills
bestehen -> dann waere sie ein Backtest-Artefakt.

Dieser Test misst, wie tief der Preis beim Fill DURCH das Limit handelt
(Penetration = (limit - low_fill) / limit). Je tiefer, desto sicherer gab es
einen Gegenpart und desto realer ist der Fill.

Zwei Ausgaben:
  1) Verteilung der Penetration ueber alle Fills (wie fragil sind die Fills?)
  2) Kante als Funktion einer Mindest-Penetration: es werden nur Fills gezaehlt,
     bei denen der Preis mind. X% unter dem Limit handelte. Bleibt net-mean UND
     ohneT5 bis z.B. 0.10% positiv und ueber die Monate stabil -> die Kante ist
     KEIN Wick-Artefakt, sondern ueberlebt garantiert-reale Fills.
  3) (falls Volumen-Spalte vorhanden) Split nach Pair-Liquiditaet.

Aufruf im trading-system:
  .venv/bin/python3 /root/TradeBot/scripts/fill_quality.py \
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

COST_MAKER = 0.20
FILL_WINDOW = 2
STRATS = ["rsi_cross", "pullback"]
VOL_CANDIDATES = ["volume", "vol", "quote_volume", "quote_asset_volume",
                  "taker_buy_base", "base_volume", "v"]


def collect(cfg, candles_dir):
    ex = cfg["exit"]
    files = sorted(glob.glob(os.path.join(candles_dir, "*_1h_features.csv")))
    rows = []
    signals_total = 0
    vol_col_seen = None

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

        vc = next((c for c in VOL_CANDIDATES if c in d.columns), None)
        if vc:
            vol_col_seen = vc
            pair_med_vol = pd.to_numeric(d[vc], errors="coerce").median()
        else:
            pair_med_vol = np.nan

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
                atr = atrp[i]; limit_price = close[i]
                if not (atr > 0 and limit_price > 0):
                    continue
                signals_total += 1

                # touch-Fill (mechanisch korrekt), plus Penetrationstiefe
                fill_bar = None
                for j in range(1, FILL_WINDOW + 1):
                    if i + j >= n:
                        break
                    if low[i + j] <= limit_price:
                        fill_bar = i + j
                        break
                if fill_bar is None:
                    continue
                penetration = (limit_price - low[fill_bar]) / limit_price * 100.0

                tr = Trade(
                    pair=pair, strategy=strat, entry_ts=ts[fill_bar],
                    entry_price=float(limit_price), atr_pct=float(atr), stake_usdc=0.0,
                    init_stop_atr=float(ex["init_stop_atr"]),
                    arm_mfe_atr=float(ex["arm_mfe_atr"]),
                    trail_atr=float(ex["trail_atr"]),
                    max_hold_bars=int(ex["max_hold_bars"]),
                )
                k = fill_bar + 1; closed = False
                while k < n and tr.bars_held < ex["max_hold_bars"]:
                    closed = tr.update_bar(high[k], low[k], close[k], ts[k], COST_MAKER)
                    if closed:
                        break
                    k += 1
                if not closed:
                    cl = (close[min(k, n - 1)] / tr.entry_price - 1.0) * 100.0
                    tr._close(cl, ts[min(k, n - 1)], "eod", COST_MAKER)

                rows.append({
                    "pair": pair, "strategy": strat, "entry_ts": tr.entry_ts,
                    "net_pct": tr.net_pct, "penetration": penetration,
                    "pair_med_vol": pair_med_vol,
                })
    return pd.DataFrame(rows), signals_total, vol_col_seen


def edge(df):
    """net-mean, ohneT5, WR, Monate-positiv fuer einen Trade-Satz."""
    n = len(df); g = df["net_pct"]
    if n == 0:
        return dict(n=0, mean=0, wo5=0, wr=0, mpos="-", mtot=0)
    rs = np.sort(g.values)[::-1]
    wo5 = rs[5:].mean() if n > 5 else g.mean()
    dd = df.copy()
    dd["m"] = pd.to_datetime(dd["entry_ts"], errors="coerce", utc=True).dt.strftime("%Y-%m")
    mm = dd.groupby("m")["net_pct"].mean()
    mm = mm[dd.groupby("m")["net_pct"].count() >= 15]
    mpos = int((mm > 0).sum()); mtot = int(len(mm))
    return dict(n=n, mean=g.mean(), wo5=wo5, wr=(g > 0).mean() * 100, mpos=mpos, mtot=mtot)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/config_1h.yaml"
    candles = sys.argv[2] if len(sys.argv) > 2 else "data/candles"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    r, sig_total, vol_col = collect(cfg, candles)
    fr = len(r) / sig_total * 100 if sig_total else 0

    print("=" * 88)
    print("ENDVISION-1h  FILL-QUALITAETS-TEST  (touch-Fill + Penetrationstiefe)")
    print(f"Signale={sig_total}  gefuellt={len(r)}  Fill-Rate={fr:.1f}%   "
          f"Volumen-Spalte: {vol_col or 'KEINE gefunden'}")
    print("=" * 88)

    # 1) Penetrations-Verteilung
    p = r["penetration"]
    print("\n[1] Wie tief handelt der Preis beim Fill DURCH das Limit? (Penetration in %)")
    for q in [10, 25, 50, 75, 90]:
        print(f"    P{q:2d} = {np.percentile(p, q):.3f}%")
    for thr in [0.02, 0.05, 0.10]:
        share = (p < thr).mean() * 100
        print(f"    Anteil Fills mit Penetration < {thr:.2f}%  (fragile Wicks): {share:.1f}%")

    # 2) Kante je Mindest-Penetration
    print("\n[2] Kante, wenn nur Fills mit mind. X% Penetration zaehlen")
    print("    (steigendes X = nur immer sicherere, real gefuellte Trades)")
    print(f"    {'minPen':>7} | {'Fill%':>6} | {'n':>5} | {'net-mean':>9} | {'ohneT5':>9} | {'WR':>5} | Monate+")
    print("    " + "-" * 74)
    for thr in [0.00, 0.03, 0.05, 0.10, 0.15, 0.25]:
        sub = r[r["penetration"] >= thr]
        e = edge(sub)
        subfr = e["n"] / sig_total * 100 if sig_total else 0
        robust = "ok" if (e["mean"] > 0 and e["wo5"] > 0 and e["n"] >= 30) else ""
        print(f"    {thr:6.2f}% | {subfr:5.1f}% | {e['n']:5d} | {e['mean']:+8.4f}% | "
              f"{e['wo5']:+8.4f}% | {e['wr']:4.1f}% | {e['mpos']}/{e['mtot']} {robust}")

    # 2b) je Strategie bei realistischer Schwelle 0.05%
    print("\n[2b] Bei realistischer Schwelle 0.05% Penetration, je Strategie:")
    sub = r[r["penetration"] >= 0.05]
    for strat in STRATS:
        e = edge(sub[sub["strategy"] == strat])
        print(f"    {strat:10s} n={e['n']:5d}  net-mean={e['mean']:+.4f}%  "
              f"ohneT5={e['wo5']:+.4f}%  WR={e['wr']:.1f}%  Monate+={e['mpos']}/{e['mtot']}")

    # 3) Liquiditaets-Split (nur falls Volumen vorhanden)
    if vol_col and r["pair_med_vol"].notna().any():
        print(f"\n[3] Liquiditaets-Split nach Median-{vol_col} der Pairs (Fills mit Pen.>=0.05%)")
        sub = r[r["penetration"] >= 0.05].copy()
        med = sub["pair_med_vol"].median()
        hi = sub[sub["pair_med_vol"] >= med]; lo = sub[sub["pair_med_vol"] < med]
        for name, g in [("liquide Haelfte", hi), ("illiquide Haelfte", lo)]:
            e = edge(g)
            print(f"    {name:18s} n={e['n']:5d}  net-mean={e['mean']:+.4f}%  "
                  f"ohneT5={e['wo5']:+.4f}%  WR={e['wr']:.1f}%  Monate+={e['mpos']}/{e['mtot']}")
    else:
        print("\n[3] Liquiditaets-Split uebersprungen (keine Volumen-Spalte in den Features).")

    print("\n" + "=" * 88)
    print("LESART: Bleibt in [2] net-mean UND ohneT5 bis mind. 0.10% Penetration positiv")
    print("und ueber die Monate stabil, ist die Kante KEIN Wick-Artefakt -> Live-Bau lohnt.")
    print("Bricht sie schon bei 0.05% weg, lebte sie nur von 1-Tick-Dochten -> Finger weg.")


if __name__ == "__main__":
    main()
