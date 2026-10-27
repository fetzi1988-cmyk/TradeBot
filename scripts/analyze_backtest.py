#!/usr/bin/env python3
"""
Slippage-Stress-Test auf dem Backtest-Trade-Log.

Nimmt das von backtest_1h.py geschriebene Log (gross_pct je Trade) und rechnet
den Netto-Erwartungswert bei steigenden Kosten durch. Beantwortet: ueberlebt
eine Strategie-Kante auch, wenn die reale Ausfuehrung teurer ist als die
angenommenen 0.235%? Eine Kante, die bei 0.30% schon kippt, ist zu duenn
fuer den Live-Betrieb.

Aufruf:  python3 /root/TradeBot/scripts/analyze_backtest.py /root/trading-system/logs/trades_1h_backtest.csv
"""
import sys
import numpy as np
import pandas as pd

f = sys.argv[1] if len(sys.argv) > 1 else "logs/trades_1h_backtest.csv"
r = pd.read_csv(f)
COSTS = [0.235, 0.26, 0.28, 0.30, 0.35]

print(f"SLIPPAGE-STRESS-TEST  n={len(r)}  Datei: {f}")
print("Frage: haelt die Kante hoehere Kosten aus? (net-mean UND ohneT5 muessen positiv bleiben)")
print("=" * 86)

def line(net, label):
    n = len(net)
    rs = np.sort(net.values)[::-1]
    wo5 = rs[5:].mean() if n > 5 else net.mean()
    m = net.mean()
    mark = "POS" if m > 0 else "neg"
    t5 = "ohneT5+" if wo5 > 0 else "ohneT5-"
    return f"    {label}: net-mean={m:+.4f}% [{mark}]  ohneT5={wo5:+.4f}% [{t5}]  net-WR={(net>0).mean()*100:4.1f}%"

order = [s for s in r["strategy"].value_counts().index if (r["strategy"] == s).sum() >= 30]
for strat in order:
    g = r[r["strategy"] == strat]
    print(f"\n### {strat}  n={len(g)}")
    for c in COSTS:
        print(line(g["gross_pct"] - c, f"c={c:.3f}"))

print("\n" + "=" * 86)
print("GESAMT (alle Strategien):")
for c in COSTS:
    print(line(r["gross_pct"] - c, f"c={c:.3f}"))
print("\nLesart: Strategie ist live-tauglich-robust, wenn net-mean UND ohneT5 bis")
print("mindestens c=0.30 positiv bleiben. Kippt sie frueher -> Kante zu duenn.")
