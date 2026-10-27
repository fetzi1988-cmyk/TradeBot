"""
HTF Feature Engine — Higher Timeframe Bias Berechnung
Berechnet 1h und 15m Kontext für jedes Pair.
Wird als Gate im Signal Router verwendet.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

def _log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}")

class HTFFeatureEngine:
    def __init__(self, candles_dir: str = "data/candles"):
        self.candles_dir = Path(candles_dir)

    def _load(self, pair: str, timeframe: str) -> pd.DataFrame | None:
        fname = pair.replace("/", "_") + f"_{timeframe}.csv"
        path = self.candles_dir / fname
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if len(df) < 50:
            return None
        return df

    def _adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low  = df["low"]
        close = df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        dm_plus  = (high - high.shift(1)).clip(lower=0)
        dm_minus = (low.shift(1) - low).clip(lower=0)
        dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
        dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
        atr  = tr.ewm(alpha=1/period, adjust=False).mean()
        di_plus  = 100 * dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
        di_minus = 100 * dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan)
        dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan))
        return dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)

    def compute(self, pair: str) -> dict:
        """
        Gibt HTF Bias für ein Pair zurück.
        Wird vom Signal Router als Gate verwendet.
        """
        result = {
            "htf_bullish":      True,   # Default: nicht blockieren
            "htf_bearish":      False,
            "htf_trending":     False,
            "htf_ranging":      True,
            "htf_1h_close":     0.0,
            "htf_1h_ema50":     0.0,
            "htf_1h_ema200":    0.0,
            "htf_1h_adx":       0.0,
            "htf_15m_bullish":  True,
            "htf_valid":        False,
        }

        # ── 1h Analyse ───────────────────────────────────────────
        df_1h = self._load(pair, "1h")
        if df_1h is not None and len(df_1h) >= 50:
            close = df_1h["close"]
            ema50  = close.ewm(span=50,  adjust=False).mean()
            ema200 = close.ewm(span=200, adjust=False).mean()
            adx    = self._adx(df_1h)

            last_close  = float(close.iloc[-1])
            last_ema50  = float(ema50.iloc[-1])
            last_ema200 = float(ema200.iloc[-1])
            last_adx    = float(adx.iloc[-1])

            # Bias bestimmen
            htf_bullish  = last_close > last_ema200
            htf_bearish  = last_close < last_ema200
            htf_trending = last_adx > 22
            htf_ranging  = last_adx < 18

            result.update({
                "htf_bullish":   htf_bullish,
                "htf_bearish":   htf_bearish,
                "htf_trending":  htf_trending,
                "htf_ranging":   htf_ranging,
                "htf_1h_close":  last_close,
                "htf_1h_ema50":  last_ema50,
                "htf_1h_ema200": last_ema200,
                "htf_1h_adx":    last_adx,
                "htf_valid":     True,
            })

        # ── 15m Analyse ──────────────────────────────────────────
        df_15m = self._load(pair, "15m")
        if df_15m is not None and len(df_15m) >= 20:
            close_15m = df_15m["close"]
            ema20_15m = close_15m.ewm(span=20, adjust=False).mean()
            last_15m  = float(close_15m.iloc[-1])
            last_ema20_15m = float(ema20_15m.iloc[-1])

            result["htf_15m_bullish"] = last_15m > last_ema20_15m

        return result

    def process_many(self, pairs: list[str]) -> dict[str, dict]:
        """Berechnet HTF Features für alle Pairs"""
        htf_map = {}
        valid = 0
        for pair in pairs:
            htf_map[pair] = self.compute(pair)
            if htf_map[pair]["htf_valid"]:
                valid += 1
        _log(f"htf_features pairs={len(pairs)} valid={valid}")
        return htf_map
