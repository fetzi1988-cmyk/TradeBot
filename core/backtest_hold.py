"""
ENDVISION — Halten vs. getimter Bot: faengt simples Halten das "taegliche Gruen"?

Fetzis Beobachtung stimmt: jeden Tag sind viele der ~218 Coins gruen. Die Frage
ist nur, ob man das mit HALTEN einfaengt (statt mit getimtem Rein-Raus, das vom
Stop zerhackt wird). Dieser Backtest misst es an den echten Preisdaten:

  1) EQUAL-WEIGHT-INDEX  — "einfach alles gleichgewichtet halten". Benchmark fuer
     die reine Marktdrift (implizit jede Bar rebalanciert, keine Gebuehr -> das ist
     ungefaehr das, was Halten einbringt).
  2) MOMENTUM TOP-N      — jede Woche die staerksten N Coins (hoechste Trailing-
     Rendite) gleichgewichtet halten, woechentlich umschichten, MIT Gebuehren auf
     den Umschlag. Faengt "halte die Gewinner".
  3) Beide gegen den getimten Scalp-Bot (den kennen wir: ~0 nach Kosten).

Ausgabe je Strategie: Endstand aus 1000 USDC, Gesamt-/annualisierte Rendite,
Max-Drawdown, Monats-Split. EHRLICH: der Max-Drawdown zeigt das Minus, das beim
Halten im Abschwung zurueckkommt.

Aufruf auf dem Server:
  .venv/bin/python3 /root/TradeBot/core/backtest_hold.py data/candles 1h
"""
from __future__ import annotations
import sys, glob, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

CANDLES = sys.argv[1] if len(sys.argv) > 1 else "data/candles"
TF = sys.argv[2] if len(sys.argv) > 2 else "1h"

BARS_PER_DAY = {"1h": 24, "5m": 288, "15m": 96, "4h": 6, "1d": 1}.get(TF, 24)
REB_DAYS = 7                      # woechentlich umschichten
TOP_N = 20                        # Momentum: so viele Gewinner halten
MOM_LOOK_DAYS = 7                 # Trailing-Fenster fuer die Momentum-Rangliste
FEE = 0.001                       # 0.10% je Seite auf den umgeschlagenen Anteil (Maker)


def load_matrix():
    pat = os.path.join(CANDLES, f"*_{TF}_features.csv")
    files = sorted(glob.glob(pat))
    if not files:
        files = sorted(f for f in glob.glob(os.path.join(CANDLES, f"*_{TF}.csv"))
                       if not f.endswith(f"_{TF}_features.csv"))
    ser = {}
    for fp in files:
        pair = os.path.basename(fp).replace(f"_{TF}_features.csv", "").replace(f"_{TF}.csv", "")
        try:
            d = pd.read_csv(fp, usecols=lambda c: c in ("timestamp", "close"))
        except Exception:
            continue
        if "close" not in d.columns or len(d) < 60:
            continue
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        ts = d["timestamp"].astype("int64")
        if ts.median() < 1e12:
            ts = ts * 1000
        ser[pair] = pd.Series(d["close"].values, index=ts.values)
    if not ser:
        return None
    px = pd.DataFrame(ser).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    return px


def dd(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak) * 100.0)


def monthly(equity, idx):
    s = pd.Series(equity, index=pd.to_datetime(idx, unit="ms", utc=True))
    m = s.resample("MS").last() / s.resample("MS").first() - 1.0
    return {t.strftime("%Y-%m"): v * 100 for t, v in m.items() if np.isfinite(v)}


def report(name, equity, idx, cap0):
    end = equity[-1]; ret = (end / cap0 - 1) * 100
    days = (idx[-1] - idx[0]) / 86400000.0
    ann = ((end / cap0) ** (365.0 / days) - 1) * 100 if days > 0 else 0
    print(f"\n### {name}")
    print(f"  Start {cap0:.0f} -> Ende {end:.0f} USDC  ({ret:+.1f}%, annualisiert ~{ann:+.0f}%)  "
          f"ueber {days:.0f} Tage")
    print(f"  Max-Drawdown = {dd(equity):.1f}%   <-- das Minus, das Halten im Abschwung frisst")
    mon = monthly(equity, idx)
    cells = "  ".join(f"{k}:{v:+.0f}%" for k, v in list(mon.items()))
    print(f"  Monate: {cells}")


def main():
    px = load_matrix()
    if px is None:
        print(f"Keine {TF}-Daten in {CANDLES}."); return
    idx = px.index.values
    rets = px.pct_change().values            # Bar-Renditen je Coin (NaN wo nicht gelistet)
    n, m = rets.shape
    cap0 = 1000.0
    print("=" * 92)
    print(f"HALTEN vs. GETIMTER BOT   {m} Coins, TF={TF}, {n} Bars")
    print(f"Faengt simples Halten das 'taegliche Gruen'? (ehrlich inkl. Drawdown)")
    print("=" * 92)

    # (1) Equal-Weight-Index: Mittel aller verfuegbaren Coin-Renditen je Bar
    ew = np.nanmean(rets, axis=1)
    ew = np.nan_to_num(ew, nan=0.0)
    eq_ew = cap0 * np.cumprod(1.0 + ew)
    report("EQUAL-WEIGHT-INDEX  ('einfach alles halten')", eq_ew, idx, cap0)

    # (2) Momentum Top-N, woechentliche Umschichtung, mit Gebuehren
    reb = REB_DAYS * BARS_PER_DAY
    look = MOM_LOOK_DAYS * BARS_PER_DAY
    eq = cap0
    eq_curve = np.empty(n); eq_curve[0] = cap0
    weights = np.zeros(m)
    logclose = np.log(px.values)
    for i in range(1, n):
        # Tagesrendite des gehaltenen Korbs
        r = rets[i]
        port_r = np.nansum(weights * np.where(np.isfinite(r), r, 0.0))
        eq *= (1.0 + port_r)
        # Umschichten?
        if i >= look and i % reb == 0:
            trail = logclose[i] - logclose[i - look]       # Trailing-Log-Rendite
            valid = np.isfinite(trail)
            order = np.argsort(np.where(valid, trail, -np.inf))[::-1]
            pick = order[:TOP_N]
            new_w = np.zeros(m)
            # nur Coins mit positiver Trailing-Rendite halten, sonst Cash
            chosen = [p for p in pick if valid[p] and trail[p] > 0]
            if chosen:
                for p in chosen:
                    new_w[p] = 1.0 / len(chosen)
            turnover = np.nansum(np.abs(new_w - weights))
            eq *= (1.0 - FEE * turnover)                   # Gebuehr auf den Umschlag
            weights = new_w
        eq_curve[i] = eq
    report(f"MOMENTUM TOP-{TOP_N}  (woechentlich, nur positive Trailing-Rendite, mit Gebuehr)",
           eq_curve, idx, cap0)

    print("\n" + "=" * 92)
    print("LESART: Wenn der Index/Momentum klar im Plus endet, hattest du RECHT — das")
    print("'taegliche Gruen' ist real und per HALTEN greifbar; der getimte Bot (~0) hat")
    print("es nur zerhackt. ABER: der Max-Drawdown zeigt, was im Abschwung zurueckkommt.")
    print("Das ist Markt-Exposure (Beta), kein kostenloses Plus. Ehrlich, aber real.")


if __name__ == "__main__":
    main()
