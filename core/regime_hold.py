"""
REGIME-HALTEN — Beta kassieren, Crash aussitzen (der eine ehrliche Hebel).

Befund des Projekts: aktive Strategien verlieren, nur HALTEN macht Plus (Beta),
aber mit brutalem Drawdown. Ein Bot kann hier keine Rendite ERZEUGEN; er kann
nur das RISIKO steuern. Dieser Test prueft die einzige Idee, die uebrig bleibt:

  - Markt-Schalter per BTC: solange BTC ueber seinem langen Durchschnitt liegt
    (REGIME_MA Bars), ist der Markt "risk-on" -> Gleichgewicht-Korb halten.
  - Faellt BTC darunter -> "risk-off" -> komplett Cash, Crash aussitzen.
  - EIN grober Schalter fuer das GANZE Buch (kein Per-Coin-Timing -> kein Whipsaw).

EHRLICHKEIT:
  - Per-Bar-Coin-Renditen werden auf +-CLIP gekappt, um die Mikro-Coin-Pump-Fata-
    Morgana (ein Coin +5000% = Index-Artefakt) zu entschaerfen. Das macht die Zahl
    konservativer und handelbarer.
  - Regime-Entscheidung nutzt nur Vergangenheit (BTC-Schluss t-1 vs MA t-1).
  - Vergleich: IMMER-HALTEN vs REGIME-GESCHALTET. Frage: kappt der Schalter den
    Drawdown deutlich und behaelt genug vom Plus?

Aufruf:
  .venv/bin/python3 /root/TradeBot/core/regime_hold.py data/candles 1h
"""
from __future__ import annotations
import sys, glob, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

CANDLES = sys.argv[1] if len(sys.argv) > 1 else "data/candles"
TF = sys.argv[2] if len(sys.argv) > 2 else "1h"
BARS_PER_DAY = {"1h": 24, "5m": 288, "4h": 6, "1d": 1}.get(TF, 24)
REGIME_MA = 30 * BARS_PER_DAY     # ~30 Tage BTC-Durchschnitt als Markt-Regime
CLIP = 0.30                        # Per-Bar-Coin-Rendite auf +-30% kappen (Outlier-Schutz)
SWITCH_FEE = 0.001                 # 0.10% Gebuehr beim Ein-/Aussteigen des ganzen Korbs


def load_matrix():
    pat = os.path.join(CANDLES, f"*_{TF}_features.csv")
    files = sorted(glob.glob(pat)) or sorted(
        f for f in glob.glob(os.path.join(CANDLES, f"*_{TF}.csv"))
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
        return None, None
    px = pd.DataFrame(ser).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    btc_col = next((c for c in ("BTC_USDT", "BTC_USDC", "BTCUSDT", "BTCUSDC") if c in px.columns), None)
    if btc_col is None:  # flexibel: irgendeine Spalte, die mit BTC beginnt
        btc_col = next((c for c in px.columns if c.upper().startswith("BTC")), None)
    return px, btc_col


def top_by_volume(candles_dir, tf, n):
    """Top-n Keys nach Median-Quote-Volumen (close*volume) ueber die Historie.
    Naeherung fuer 'die groessten/liquidesten Coins'. _USDC bevorzugt (handelbar)."""
    rows = []
    for fp in glob.glob(os.path.join(candles_dir, f"*_{tf}*.csv")):
        key = os.path.basename(fp).replace(f"_{tf}_features.csv", "").replace(f"_{tf}.csv", "")
        if key.endswith("_USDT"):      # USDT-Taktgeber nicht in den Korb
            continue
        try:
            d = pd.read_csv(fp, usecols=lambda c: c in ("close", "volume"))
        except Exception:
            continue
        if "close" not in d or "volume" not in d or len(d) < 60:
            continue
        cc = pd.to_numeric(d["close"], errors="coerce").dropna()
        if len(cc) < 60 or cc.min() <= 0 or cc.max() / cc.min() < 1.5:
            continue    # Stablecoin/Peg (bewegt sich nie >50%) -> totes Gewicht, raus
        qv = (cc * pd.to_numeric(d["volume"], errors="coerce")).median()
        if np.isfinite(qv) and qv > 0:
            rows.append((key, qv))
    rows.sort(key=lambda x: -x[1])
    return [k for k, _ in rows[:n]]


def dd(eq):
    peak = np.maximum.accumulate(eq)
    return float(np.max((peak - eq) / peak) * 100)


def monthly(eq, idx):
    s = pd.Series(eq, index=pd.to_datetime(idx, unit="ms", utc=True))
    m = s.resample("MS").last() / s.resample("MS").first() - 1
    return "  ".join(f"{t.strftime('%Y-%m')}:{v*100:+.0f}%" for t, v in m.items() if np.isfinite(v))


def rep(name, eq, idx, extra=""):
    end = eq[-1]; ret = (end / 1000 - 1) * 100
    days = (idx[-1] - idx[0]) / 86400000.0
    ann = ((end / 1000) ** (365.0 / days) - 1) * 100 if days > 0 else 0
    print(f"\n### {name}")
    print(f"  Start 1000 -> Ende {end:.0f} USDC  ({ret:+.1f}%, ann ~{ann:+.0f}%)  ueber {days:.0f} Tage{extra}")
    print(f"  Max-Drawdown = {dd(eq):.1f}%")
    print(f"  Monate: {monthly(eq, idx)}")


def main():
    px, btc = load_matrix()
    if px is None:
        print(f"Keine {TF}-Daten in {CANDLES}."); return
    idx = px.index.values
    # optionaler Instrument-Filter: argv[4]
    #   "top40"  -> die 40 groessten Coins nach Volumen
    #   "BTC_,ETH_,..." -> Komma-Liste von Key-Fragmenten
    #   ohne -> alle Coins
    a4 = sys.argv[4].strip() if len(sys.argv) > 4 else None
    if a4 and a4.lower().startswith("top") and a4[3:].isdigit():
        topkeys = set(top_by_volume(CANDLES, TF, int(a4[3:])))
        bcols = [c for c in px.columns if c in topkeys]
    elif a4:
        wl = [w.strip() for w in a4.split(",")]
        bcols = [c for c in px.columns if any(w in c for w in wl)]
    else:
        bcols = list(px.columns)
    rets = np.clip(px[bcols].pct_change().values, -CLIP, CLIP)   # Outlier-gekappt
    basket = np.nan_to_num(np.nanmean(rets, axis=1), nan=0.0)

    print("=" * 92)
    print(f"REGIME-HALTEN   Korb={len(bcols)} Coins von {px.shape[1]}, TF={TF}, {len(idx)} Bars   "
          f"(Per-Bar auf +-{CLIP*100:.0f}% gekappt)")
    if a4:
        print(f"  Instrument-Filter: '{a4}'  ->  {len(bcols)} Coins: {', '.join(bcols[:12])}{' ...' if len(bcols) > 12 else ''}")
    print("=" * 92)

    eq_hold = 1000 * np.cumprod(1 + basket)
    rep("IMMER-HALTEN (gekappter Gleichgewicht-Korb)", eq_hold, idx)

    dt = pd.to_datetime(idx, unit="ms", utc=True)
    # Regime-Quelle waehlen: BTC, falls es genug Historie hat (>=40 Tage), sonst
    # der Korb-Index selbst (sein eigener Trend) -> robust gegen duennes BTC-Paar.
    use_btc = False
    if btc is not None:
        bspan = pd.Series(pd.to_numeric(px[btc], errors="coerce").values, index=dt).dropna()
        if len(bspan) and (bspan.index.max() - bspan.index.min()).days >= 40:
            use_btc = True
    if use_btc:
        market = pd.Series(pd.to_numeric(px[btc], errors="coerce").values, index=dt).ffill()
        src = f"BTC ({btc})"
    else:
        market = pd.Series(eq_hold, index=dt)
        src = "Korb-Index (eigener Trend)"

    # Regime auf TAGESBASIS: Markt-Serie auf Tage resampeln, MA, Entscheidung
    # aus dem VORTAG (kein Look-ahead). MA-Tage via argv[3] (Default 30; fuer
    # saubere Tagesdaten besser 100-200).
    REGIME_MA_DAYS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    daily = market.resample("1D").last().ffill()
    ma_daily = daily.rolling(REGIME_MA_DAYS, min_periods=10).mean()
    on = (daily > ma_daily).shift(1).fillna(False)   # Vortag -> kein Look-ahead
    on_by_date = {ts.strftime("%Y-%m-%d"): bool(v) for ts, v in on.items()}
    risk_on = np.array([on_by_date.get(k, False) for k in dt.strftime("%Y-%m-%d")])
    print(f"\n  [Diagnose] Regime-Quelle: {src}   Tage gesamt={len(daily)}")
    print(f"  [Diagnose] risk-on an {int(on.sum())}/{len(on)} Tagen  -> {int(risk_on.sum())}/{len(risk_on)} Bars im Markt")

    eq = 1000.0; curve = np.empty(len(idx)); curve[0] = 1000.0; invested = False
    days_in = 0
    for i in range(1, len(idx)):
        if risk_on[i]:
            eq *= (1 + basket[i])
            days_in += 1
            if not invested:
                eq *= (1 - SWITCH_FEE); invested = True   # einsteigen
        else:
            if invested:
                eq *= (1 - SWITCH_FEE); invested = False   # aussteigen in Cash
        curve[i] = eq
    pct_in = days_in / (len(idx) - 1) * 100
    rep(f"REGIME-GESCHALTET ({src} > {REGIME_MA_DAYS}-Tage-MA = halten, sonst Cash)",
        curve, idx, extra=f"   [{pct_in:.0f}% der Zeit im Markt]")

    print("\n" + "=" * 92)
    print("LESART: Drueckt REGIME-GESCHALTET den Max-Drawdown deutlich unter IMMER-HALTEN")
    print("und behaelt einen grossen Teil des Plus -> DAS ist deine Maschine: Prozente")
    print("mitnehmen, Crash aussitzen. Bringt der Schalter nichts -> das Ende der Fahnenstange.")


if __name__ == "__main__":
    main()
