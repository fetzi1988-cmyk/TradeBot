#!/usr/bin/env python3
"""
Opportunity Scanner — REINER LOGGER, tradet nichts, ändert nichts.
Scannt alle USDC-Pairs alle 30min auf Re-Akkumulations-Muster.
Loggt Funde mit Preis + Timestamp nach data/history/opportunities.csv
Validierung später: ist der Preis nach dem Fund gestiegen?
"""
import ccxt, csv, os, math
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

OUT = Path("/root/trading-system/data/history/opportunities.csv")
EXCLUDE = {"USD1/USDC","EUR/USDC","FDUSD/USDC","PAXG/USDC","USDT/USDC",
           "AUD/USDC","EURI/USDC","XAUT/USDC","USDE/USDC","BFUSD/USDC",
           "USDSOLD/USDC","BCHABC/USDC","BSV/USDC","BTT/USDC"}

def _log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}")

def compute_indicators(df):
    c = df['close'].astype(float); h = df['high'].astype(float)
    l = df['low'].astype(float); v = df['volume'].astype(float)
    o = df['open'].astype(float)
    out = {}
    # EMA200 / EMA50 / EMA20
    ema200 = c.ewm(span=200, adjust=False).mean()
    ema50  = c.ewm(span=50, adjust=False).mean()
    ema20  = c.ewm(span=20, adjust=False).mean()
    # RSI 14
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100/(1 + gain/loss.replace(0, np.nan))
    # VWAP (Session-Approx über gesamtes Fenster)
    tp = (h+l+c)/3
    vwap = (tp*v).cumsum() / v.cumsum().replace(0, np.nan)
    # OBV
    obv = (np.sign(c.diff().fillna(0)) * v).cumsum()
    # CVD Proxy aus OHLCV
    rng = (h-l).replace(0, np.nan)
    cvd_delta = ((c-o)/rng * v).fillna(0)
    cvd_5 = cvd_delta.rolling(5).sum()
    cvd_rising = cvd_5 > cvd_5.shift(2)
    # Higher Low (choch-Basis)
    higher_low = (l > l.shift(1)) & (l.shift(1) > l.shift(2))
    structure_break = c > h.rolling(5).max().shift(1)
    choch = higher_low & structure_break
    # Wyckoff Spring
    low20 = l.rolling(20).min().shift(1)
    spring = (l < low20*0.998) & (c > low20)
    # Volume Verlauf: Bounce-Vol > Rückgang-Vol
    vol_ratio = v / v.rolling(20).mean()
    # Tief + Recovery
    low_idx = l.iloc[-288:].idxmin() if len(l) >= 288 else l.idxmin()
    low_val = float(l.loc[low_idx])
    recovery_pct = (float(c.iloc[-1]) - low_val)/low_val*100 if low_val > 0 else 0

    i = -1
    out['close'] = float(c.iloc[i])
    out['over_ema200'] = bool(c.iloc[i] > ema200.iloc[i])
    out['over_ema50']  = bool(c.iloc[i] > ema50.iloc[i])
    out['below_vwap']  = bool(c.iloc[i] < vwap.iloc[i])
    out['rsi'] = float(rsi.iloc[i]) if not pd.isna(rsi.iloc[i]) else 50.0
    out['choch_recent']  = bool(choch.iloc[-4:].any())
    out['spring_recent'] = bool(spring.iloc[-4:].any())
    out['cvd_rising'] = bool(cvd_rising.iloc[i]) if not pd.isna(cvd_5.iloc[i]) else False
    out['cvd_5'] = float(cvd_5.iloc[i]) if not pd.isna(cvd_5.iloc[i]) else 0.0
    out['obv_rising'] = bool(obv.iloc[i] > obv.iloc[i-3])
    out['vol_ratio'] = float(vol_ratio.iloc[i]) if not pd.isna(vol_ratio.iloc[i]) else 0.0
    out['recovery_pct'] = recovery_pct
    out['low_time'] = str(df.loc[low_idx, 'dt'])[:16] if 'dt' in df.columns else ''
    return out

def main():
    exchange = ccxt.binance()
    exchange.load_markets()
    pairs = [s for s in exchange.markets if s.endswith("/USDC") and s not in EXCLUDE]
    _log(f"Scanne {len(pairs)} Pairs...")

    new_file = not OUT.exists()
    found = 0
    with open(OUT, 'a', newline='') as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(['ts_utc','pair','price','score','over_ema200','choch','spring',
                        'cvd_rising','cvd_5','obv_rising','rsi','below_vwap','vol_ratio',
                        'recovery_pct','low_time'])
        for p in pairs:
            try:
                rows = exchange.fetch_ohlcv(p, timeframe='5m', limit=300)
                if len(rows) < 100: continue
                df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
                df['dt'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                ind = compute_indicators(df)
                # Re-Akkumulations-Score (gleiche Logik wie Analyse vom 12.06)
                score = sum([
                    ind['over_ema200'],
                    ind['choch_recent'] or ind['spring_recent'],
                    ind['cvd_rising'] and ind['obv_rising'],
                    35 < ind['rsi'] < 62,
                    ind['below_vwap'],
                ])
                if score >= 4:
                    found += 1
                    w.writerow([datetime.now(timezone.utc).isoformat(), p,
                        f"{ind['close']:.8f}", score, ind['over_ema200'],
                        ind['choch_recent'], ind['spring_recent'], ind['cvd_rising'],
                        f"{ind['cvd_5']:.0f}", ind['obv_rising'], f"{ind['rsi']:.1f}",
                        ind['below_vwap'], f"{ind['vol_ratio']:.2f}",
                        f"{ind['recovery_pct']:.2f}", ind['low_time']])
                    _log(f"FUND {p} score={score} preis={ind['close']:.6f} rsi={ind['rsi']:.0f} recovery={ind['recovery_pct']:.1f}%")
            except Exception as e:
                pass
    _log(f"Fertig. {found} Funde geloggt nach {OUT}")

if __name__ == "__main__":
    main()
