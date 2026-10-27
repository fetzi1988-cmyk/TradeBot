#!/usr/bin/env python3
"""
ENDVISION-1h — Integritaets-Check VOR dem Live-Gang.

Klaert die eine Frage, an der das ganze +266%-Replay haengt: stimmen die
Einheiten von atr_pct zwischen Signal-Bau (feature_engine behandelt atr_pct als
BRUCH, z.B. 0.0030 = 0.3%) und der Portfolio-Engine (rechnet -2.5*atr_pct gegen
Renditen in PROZENT)? Passt das nicht zusammen, sitzen die Stops ~100x falsch
und das Ergebnis waere ein Artefakt.

Prueft nuechtern an den ECHTEN Daten:
  [A] Verteilung von atr_pct (settlet Bruch vs. Prozent).
  [B] Stop-Distanz, die die Engine benutzt (2.5*atr_pct), vs. typische
      1-Bar-Bewegung. Ist die Stop-Distanz ~gleich gross oder groesser ->
      sinnvoll. Ist sie ~100x kleiner -> Stops sitzen im Rauschen -> BUG.
  [C] 5 echte rsi_cross-Trades Bar fuer Bar durchgespielt, damit man SIEHT,
      wie weit der Stop sitzt und wie lange gehalten wird.

Aufruf im trading-system:
  .venv/bin/python3 /root/TradeBot/scripts/integrity_check.py data/candles
"""
from __future__ import annotations
import sys, glob, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

CANDLES = sys.argv[1] if len(sys.argv) > 1 else "data/candles"
INIT_STOP_ATR = 2.5
STRATS = ["signal_rsi_cross", "signal_pullback"]

files = sorted(glob.glob(os.path.join(CANDLES, "*_1h_features.csv")))
print("=" * 80)
print(f"INTEGRITAETS-CHECK  —  {len(files)} Feature-Dateien in {CANDLES}")
print("=" * 80)

atr_all = []
atr_at_sig = {s: [] for s in STRATS}
bar_moves = []          # |close/prev_close - 1| * 100  (typische Bar-Bewegung in %)
examples = []           # (pair, idx, df) fuer rsi_cross-Beispiele

for fp in files:
    pair = os.path.basename(fp).replace("_1h_features.csv", "")
    try:
        d = pd.read_csv(fp)
    except Exception:
        continue
    if len(d) < 60 or "atr_pct" not in d.columns:
        continue
    for c in ["close", "high", "low", "atr_pct"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values("timestamp").reset_index(drop=True)
    atr = d["atr_pct"].dropna().values
    atr_all.append(atr)
    mv = (d["close"] / d["close"].shift(1) - 1.0).abs().dropna().values * 100.0
    bar_moves.append(mv)
    for s in STRATS:
        if s in d.columns:
            m = d[s].fillna(False).astype(bool).values
            atr_at_sig[s].append(d["atr_pct"].values[m])
            if s == "signal_rsi_cross" and m.any() and len(examples) < 5:
                idxs = np.where(m)[0]
                for ii in idxs:
                    if ii + 14 < len(d) and len(examples) < 5:
                        examples.append((pair, ii, d)); break

atr_all = np.concatenate(atr_all) if atr_all else np.array([])
bar_moves = np.concatenate(bar_moves) if bar_moves else np.array([])

# [A] Einheiten
print("\n[A] atr_pct-Verteilung ueber alle Bars:")
for q in [1, 10, 50, 90, 99]:
    print(f"    P{q:2d} = {np.percentile(atr_all, q):.5f}")
med = np.median(atr_all)
einheit = "BRUCH (z.B. 0.008 = 0.8%)" if med < 0.1 else "PROZENT (z.B. 0.8 = 0.8%)"
print(f"    Median = {med:.5f}  ->  Einheit ist {einheit}")

# [B] Stop-Distanz vs. typische Bewegung
print("\n[B] Stop-Distanz der Engine (2.5 * atr_pct) vs. typische 1-Bar-Bewegung:")
# Die Engine setzt stop_ret_pct = -2.5*atr_pct und vergleicht gegen lo_ret in %.
# D.h. sie INTERPRETIERT 2.5*atr_pct als Prozentwert.
stop_used_as_pct = INIT_STOP_ATR * med          # so wie die Engine es als % nimmt
typ_move_pct = np.median(bar_moves)
print(f"    Engine-Stop  = 2.5 * {med:.5f} = {stop_used_as_pct:.5f}  (als % interpretiert)")
print(f"    typ. Bar-Bewegung (Median |ret|) = {typ_move_pct:.4f} %")
ratio = stop_used_as_pct / typ_move_pct if typ_move_pct else 0
print(f"    Verhaeltnis Stop/Bar = {ratio:.2f}")
if ratio < 0.2:
    print("    -> WARNUNG: Stop sitzt tief IM Rauschen (viel kleiner als eine Kerze).")
    print("       Fast jeder Trade wird sofort ausgestoppt = atr_pct-Einheiten passen NICHT.")
    print("       Richtig waere stop_ret_pct = -2.5*atr_pct*100, wenn atr_pct ein Bruch ist.")
elif ratio > 5:
    print("    -> Stop sehr weit (viele Kerzen entfernt). Plausibel fuer 2.5-ATR-Initialstop.")
else:
    print("    -> Stop-Distanz liegt in sinnvoller Groessenordnung (einige Kerzen).")
print("    Merke: eine 2.5-ATR-Stopdistanz SOLLTE ~2-3x der typischen Bar-Bewegung sein.")

# atr_pct an Signalbars
print("\n    atr_pct an den Signal-Bars (Median):")
for s in STRATS:
    v = np.concatenate([a for a in atr_at_sig[s] if len(a)]) if any(len(a) for a in atr_at_sig[s]) else np.array([])
    if len(v):
        print(f"      {s:18s} n={len(v):5d}  Median atr_pct={np.median(v):.5f}  "
              f"-> Stop als % = {INIT_STOP_ATR*np.median(v):.4f}")

# [C] echte Trades Bar fuer Bar
print("\n[C] 5 echte rsi_cross-Trades Bar fuer Bar (so wie die Engine sie sieht):")
print("    Entry=Signal-Close. Renditen in % vom Entry. Stop(init) = -2.5*atr_pct (als %).")
for pair, ii, d in examples:
    entry = d["close"].values[ii]; atr = d["atr_pct"].values[ii]
    stop_pct = -INIT_STOP_ATR * atr
    print(f"\n    {pair}  Entry@{entry:.6g}  atr_pct={atr:.5f}  init-Stop={stop_pct:+.4f}%")
    line = []
    for j in range(1, 9):
        if ii + j >= len(d):
            break
        lo = (d["low"].values[ii + j] / entry - 1) * 100
        hi = (d["high"].values[ii + j] / entry - 1) * 100
        hit = "  <-STOP" if lo <= stop_pct else ""
        line.append(f"      Bar+{j}: low={lo:+.3f}% high={hi:+.3f}%{hit}")
        if lo <= stop_pct:
            print("\n".join(line)); break
    else:
        print("\n".join(line))

print("\n" + "=" * 80)
print("VERDIKT: Ist [A] ein BRUCH und [B]-Verhaeltnis < 0.2, dann rechnet die")
print("Portfolio-Engine mit 100x zu engen Stops -> das +266%-Replay ist ein")
print("Einheiten-Artefakt und MUSS korrigiert und neu gerechnet werden.")
print("Ist das Verhaeltnis in [B] bei ~2-3, passen die Einheiten und das Replay haelt.")
