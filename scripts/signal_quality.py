#!/usr/bin/env python3
"""
ENDVISION-1h — Signal-Qualitaets-Test (die entscheidende Diagnose).

Frage: Haben rsi_cross / pullback UEBERHAUPT eine Richtungs-Vorhersage, oder
ist der Einstieg Rauschen? Das trennt zwei Welten:
  - Rohkante vorhanden, aber Exit zerstoert sie  -> Exit neu bauen (Hoffnung).
  - Rohkante ~0 / negativ                         -> Signal wertlos, Strategie-
                                                     klasse wechseln (ehrlich).

Misst OHNE Exit und OHNE Kosten die reine Vorwaerts-Rendite nach jedem Signal:
  - close[i] -> close[i+k] fuer k in {1,3,6,12,24} Bars (Mittel, Median, Trefferquote)
  - t-Wert = Mittel/Std*sqrt(n): ist der Effekt ueberhaupt von 0 unterscheidbar?
  - MFE/MAE ueber 12 Bars: gibt es eine ASYMMETRISCHE Ausbeute, die ein guter
    Exit einfangen koennte? (MFE >> |MAE| = ja; symmetrisch = Rauschen.)
  - Vergleich gegen die UNBEDINGTE Drift (zufaellige Bars): schlaegt das Signal
    ueberhaupt "einfach im Markt sein"?

Aufruf im trading-system:
  .venv/bin/python3 /root/TradeBot/scripts/signal_quality.py data/candles
"""
from __future__ import annotations
import sys, glob, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

CANDLES = sys.argv[1] if len(sys.argv) > 1 else "data/candles"
STRATS = ["rsi_cross", "pullback"]
HORIZONS = [1, 3, 6, 12, 24]
MFE_WIN = 12

files = sorted(glob.glob(os.path.join(CANDLES, "*_1h_features.csv")))
# Sammelbehälter: pro Strategie Listen von Vorwaerts-Renditen je Horizont
fwd = {s: {k: [] for k in HORIZONS} for s in STRATS}
mfe = {s: [] for s in STRATS}
mae = {s: [] for s in STRATS}
month = {s: {k: [] for k in HORIZONS} for s in STRATS}   # (ym, ret) fuer k=6
base_fwd = {k: [] for k in HORIZONS}                      # unbedingte Drift

rng = np.random.default_rng(42)

for fp in files:
    try:
        d = pd.read_csv(fp)
    except Exception:
        continue
    if len(d) < 60 or "close" not in d.columns:
        continue
    for c in ["close", "high", "low"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values("timestamp").reset_index(drop=True)
    n = len(d)
    close = d["close"].values; high = d["high"].values; low = d["low"].values
    ym = pd.to_datetime(d["timestamp"].astype("int64")
                        * (1000 if d["timestamp"].median() < 1e12 else 1),
                        unit="ms", utc=True).dt.strftime("%Y-%m").values

    # unbedingte Drift: 200 zufaellige Bars je Pair
    for i in rng.choice(np.arange(0, n - max(HORIZONS) - 1), size=min(200, max(1, n - 30)), replace=False):
        for k in HORIZONS:
            if i + k < n and close[i] > 0:
                base_fwd[k].append((close[i + k] / close[i] - 1) * 100)

    for s in STRATS:
        col = f"signal_{s}"
        if col not in d.columns:
            continue
        sig = np.where(d[col].fillna(False).astype(bool).values)[0]
        for i in sig:
            if close[i] <= 0:
                continue
            for k in HORIZONS:
                if i + k < n:
                    r = (close[i + k] / close[i] - 1) * 100
                    fwd[s][k].append(r)
                    if k == 6:
                        month[s][k].append((ym[i], r))
            if i + MFE_WIN < n:
                seg_hi = high[i + 1:i + 1 + MFE_WIN]; seg_lo = low[i + 1:i + 1 + MFE_WIN]
                if len(seg_hi):
                    mfe[s].append((seg_hi.max() / close[i] - 1) * 100)
                    mae[s].append((seg_lo.min() / close[i] - 1) * 100)


def stats(arr):
    a = np.array(arr, float)
    a = a[np.isfinite(a)]
    if len(a) < 5:
        return None
    m = a.mean(); sd = a.std(ddof=1)
    t = m / sd * np.sqrt(len(a)) if sd > 0 else 0
    return len(a), m, np.median(a), (a > 0).mean() * 100, t


print("=" * 86)
print("SIGNAL-QUALITAETS-TEST  —  reine Vorwaerts-Rendite OHNE Exit/Kosten")
print(f"{len(files)} Dateien.  Frage: sagt das Signal die Richtung vorher?")
print("=" * 86)

print("\n[0] UNBEDINGTE Drift (zufaellige Bars, Referenz 'einfach im Markt sein'):")
for k in HORIZONS:
    st = stats(base_fwd[k])
    if st:
        print(f"    +{k:2d}h: Mittel={st[1]:+.4f}%  Median={st[2]:+.4f}%  WR={st[3]:.1f}%  (n={st[0]})")

for s in STRATS:
    print(f"\n[{s}] Vorwaerts-Rendite nach Signal:")
    print(f"    {'Horizont':>9} | {'n':>6} | {'Mittel':>9} | {'Median':>9} | {'WR':>6} | {'t-Wert':>7} | vs Drift")
    print("    " + "-" * 78)
    for k in HORIZONS:
        st = stats(fwd[s][k]); bd = stats(base_fwd[k])
        if not st:
            continue
        edge = st[1] - (bd[1] if bd else 0)
        sig = "signifikant" if abs(st[4]) >= 2.0 else "~Rauschen"
        print(f"    {('+'+str(k)+'h'):>9} | {st[0]:6d} | {st[1]:+8.4f}% | {st[2]:+8.4f}% | "
              f"{st[3]:5.1f}% | {st[4]:+6.2f} | {edge:+.4f}%  [{sig}]")
    # MFE/MAE
    sm = stats(mfe[s]); sa = stats(mae[s])
    if sm and sa:
        asym = sm[1] / abs(sa[1]) if sa[1] != 0 else 0
        print(f"    MFE(12h)={sm[1]:+.3f}%  MAE(12h)={sa[1]:+.3f}%  "
              f"Asymmetrie MFE/|MAE|={asym:.2f}  "
              f"({'einfangbar' if asym > 1.3 else 'symmetrisch=Rauschen'})")
    # Monatsstabilitaet bei +6h
    mm = {}
    for ymv, r in month[s][6]:
        mm.setdefault(ymv, []).append(r)
    cells = [f"{k}:{np.mean(v):+.3f}" for k, v in sorted(mm.items()) if len(v) >= 20]
    if cells:
        print("    +6h je Monat (n>=20): " + "  ".join(cells))

print("\n" + "=" * 86)
print("LESART:")
print(" - Ist 'Mittel' nahe der Drift und t-Wert < 2 -> das Signal sagt die Richtung")
print("   NICHT vorher. Kein Exit kann das retten. Diese Signalklasse ist tot.")
print(" - Ist 'Mittel' klar ueber der Drift mit t >= 2 UND MFE/|MAE| > 1.3 -> es gibt")
print("   eine einfangbare Rohkante; dann lohnt ein neuer Exit statt neuer Signale.")
