"""
Erweitertes Indikator-Set fuer Klassifikations-Analyse.
EIGENSTAENDIGES Modul, NICHT Teil der Live-Pipeline (feature_engine.py).
Berechnet rein aus OHLCV - keine neuen Rohdaten noetig, volle Historie nutzbar.
"""
import pandas as pd
import numpy as np


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def sma(series, period):
    return series.rolling(period).mean()


def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean().replace(0, 1e-10)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(close, fast=12, slow=26, signal=9):
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    return dif, dea, dif - dea  # dif, dea, macd_hist


def stoch_rsi(close, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3):
    r = rsi(close, rsi_period)
    low_r = r.rolling(stoch_period).min()
    high_r = r.rolling(stoch_period).max()
    raw = (r - low_r) / (high_r - low_r).replace(0, 1e-10) * 100
    k = raw.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d


def williams_r(high, low, close, period=14):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return (hh - close) / (hh - ll).replace(0, 1e-10) * -100


def kdj(high, low, close, period=9, smooth_k=3, smooth_d=3):
    ll = low.rolling(period).min()
    hh = high.rolling(period).max()
    rsv = (close - ll) / (hh - ll).replace(0, 1e-10) * 100
    k = rsv.ewm(alpha=1/smooth_k, adjust=False).mean()
    d = k.ewm(alpha=1/smooth_d, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def bollinger(close, period=20, std_mult=2.0):
    mb = sma(close, period)
    std = close.rolling(period).std()
    up = mb + std_mult * std
    dn = mb - std_mult * std
    return up, mb, dn


def parabolic_sar(high, low, close, af_start=0.02, af_step=0.02, af_max=0.20):
    n = len(close)
    sar = np.zeros(n)
    trend_up = True
    af = af_start
    ep = low.iloc[0]
    sar[0] = high.iloc[0]
    for i in range(1, n):
        prev_sar = sar[i-1]
        if trend_up:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low.iloc[i-1], low.iloc[i-2] if i >= 2 else low.iloc[i-1])
            if low.iloc[i] < sar[i]:
                trend_up = False
                sar[i] = ep
                ep = low.iloc[i]
                af = af_start
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high.iloc[i-1], high.iloc[i-2] if i >= 2 else high.iloc[i-1])
            if high.iloc[i] > sar[i]:
                trend_up = True
                sar[i] = ep
                ep = high.iloc[i]
                af = af_start
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_step, af_max)
    return pd.Series(sar, index=close.index)


def avl(close, volume, period=20):
    """
    Average Value Line - Annaeherung als Volumen-gewichtetes gleitendes Mittel.
    ACHTUNG: Binances exakte interne Formel/Periode ist nicht oeffentlich
    dokumentiert - das ist eine plausible Annaeherung (VWMA), keine
    bestaetigte exakte Nachbildung. Periode 20 als Annahme, nicht verifiziert.
    """
    pv = close * volume
    return pv.rolling(period).sum() / volume.rolling(period).sum().replace(0, 1e-10)


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Nimmt OHLCV-DataFrame, gibt DataFrame mit allen neuen Spalten zurueck."""
    out = df.copy()
    out['rsi_6']  = rsi(out['close'], 6)
    out['rsi_14'] = rsi(out['close'], 14)
    out['rsi_24'] = rsi(out['close'], 24)

    out['dif'], out['dea'], out['macd_hist'] = macd(out['close'])

    out['ema_7'], out['ema_25'], out['ema_99'] = ema(out['close'],7), ema(out['close'],25), ema(out['close'],99)
    out['ma_5'], out['ma_7'], out['ma_10'], out['ma_25'], out['ma_99'] = (
        sma(out['close'],5), sma(out['close'],7), sma(out['close'],10),
        sma(out['close'],25), sma(out['close'],99))

    out['stochrsi_k'], out['stochrsi_d'] = stoch_rsi(out['close'])
    out['williams_r_14'] = williams_r(out['high'], out['low'], out['close'], 14)
    out['kdj_k'], out['kdj_d'], out['kdj_j'] = kdj(out['high'], out['low'], out['close'])
    out['obv'] = obv(out['close'], out['volume'])
    out['obv_ma_7']  = sma(out['obv'], 7)
    out['obv_ema_7'] = ema(out['obv'], 7)
    # Fix: OBV-Steigung relativ zur eigenen 20-Kerzen-Volatilitaet normieren,
    # NICHT durch Preis teilen - OBV und Preis sind unterschiedliche Einheiten.
    obv_std_20 = out['obv'].rolling(20).std().replace(0, 1e-10)
    out['obv_zscore_slope'] = (out['obv'] - out['obv'].shift(3)) / obv_std_20
    out['boll_up'], out['boll_mb'], out['boll_dn'] = bollinger(out['close'])
    out['sar'] = parabolic_sar(out['high'], out['low'], out['close'])
    out['avl'] = avl(out['close'], out['volume'])  # Annaeherung, siehe Docstring
    return out
