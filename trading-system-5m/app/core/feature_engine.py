from pathlib import Path
from datetime import datetime, timezone
# Housekeeping 01.07.2026: compute_for_pair (lookback=500, statisches Snapshot)
# wird NIRGENDWO in dieser Datei genutzt - toter Import, war Verwechslungsgefahr
# mit der tatsaechlich aktiven Methode. Live laeuft ausschliesslich
# add_vp_features (lookback=200, rollend pro Kerze, letzte VP_LIVE_TARGET_ROWS
# Zeilen echt berechnet, aeltere neutrale Defaults) - siehe process_pair().
from app.core.volume_profile import add_vp_features as _vp_add_features

import numpy as np
import pandas as pd
# Log-Schutz: Pandas Fragmentation-Warnings nicht ins Journal spammen.
# Das aendert keine Trading-Logik, verhindert aber massive Log-/CPU-Bremse.
import warnings as _warnings
try:
    from pandas.errors import PerformanceWarning as _PandasPerformanceWarning
    _warnings.filterwarnings("ignore", category=_PandasPerformanceWarning)
except Exception:
    pass



def _log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}")


class FeatureEngine:
    def __init__(self, candles_dir: str = "data/candles"):
        self.candles_dir = Path(candles_dir)

    def _pair_to_filename(self, pair: str) -> str:
        return pair.replace("/", "_") + "_5m.csv"

    def _pair_to_feature_filename(self, pair: str) -> str:
        return pair.replace("/", "_") + "_5m_features.csv"

    def _rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        return atr.fillna(0.0)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()
        out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()
        out["ema_200"] = out["close"].ewm(span=200, adjust=False).mean()
        out["ema_9"] = out["close"].ewm(span=9, adjust=False).mean()
        # VWAP session-based (Reset täglich 00:00 UTC)
        typical_price = (out["high"] + out["low"] + out["close"]) / 3
        out["_tp_vol"] = typical_price * out["volume"]
        if "datetime" in out.columns:
            out["_date"] = pd.to_datetime(out["datetime"], utc=True).dt.date
        else:
            out["_date"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True).dt.date
        out["_cum_tp_vol"] = out.groupby("_date")["_tp_vol"].cumsum()
        out["_cum_vol"] = out.groupby("_date")["volume"].cumsum()
        out["vwap"] = (out["_cum_tp_vol"] / out["_cum_vol"].replace(0, np.nan)).ffill()
        out.drop(columns=["_tp_vol", "_date", "_cum_tp_vol", "_cum_vol"], inplace=True)

        out["rsi_14"] = self._rsi(out["close"], 14)
        out["rsi_6"] = self._rsi(out["close"], 6)
        # StochRSI
        rsi14 = out["rsi_14"]
        rsi14_low = rsi14.rolling(14).min()
        rsi14_high = rsi14.rolling(14).max()
        stoch_rsi_raw = (rsi14 - rsi14_low) / (rsi14_high - rsi14_low).replace(0, np.nan)
        out["stoch_rsi_k"] = stoch_rsi_raw.rolling(3).mean().fillna(0.5)
        out["stoch_rsi_d"] = out["stoch_rsi_k"].rolling(3).mean().fillna(0.5)
        out["stoch_rsi_oversold"] = (out["stoch_rsi_k"] < 0.20).fillna(False)
        out["stoch_rsi_turning"] = (
            (out["stoch_rsi_k"] < 0.20) &
            (out["stoch_rsi_k"] > out["stoch_rsi_k"].shift(1))
        ).fillna(False)
        out["rsi_14"] = self._rsi(out["close"], 14)
        out["atr_14"] = self._atr(out, 14)
        # ADX — Trendstärke-Indikator
        def _calc_adx(df, period=14):
            high = df['high']
            low = df['low']
            close = df['close']
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            dm_plus = ((high - high.shift(1)).clip(lower=0)).where(
                (high - high.shift(1)) > (low.shift(1) - low), 0)
            dm_minus = ((low.shift(1) - low).clip(lower=0)).where(
                (low.shift(1) - low) > (high - high.shift(1)), 0)
            atr_s = tr.ewm(alpha=1/period, adjust=False).mean()
            di_plus = 100 * dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan)
            di_minus = 100 * dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan)
            dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
            return dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)
        out['adx'] = _calc_adx(out)
        out['adx_trend'] = out['adx'] > 25
        out['adx_range'] = out['adx'] < 20
        out['adx_rising'] = (out['adx'].diff() > 0).fillna(False)
        out['adx_squeeze_breakout'] = (out['adx_range'].shift(3).fillna(False) & out['adx_trend']).fillna(False)
        # Bollinger Bänder (20, 2)
        sma20 = out['close'].rolling(window=20).mean()
        std20 = out['close'].rolling(window=20).std()
        out['bb_upper'] = sma20 + (std20 * 2)
        out['bb_lower'] = sma20 - (std20 * 2)
        out['bb_mid'] = sma20
        out['bb_percent'] = (out['close'] - out['bb_lower']) / (out['bb_upper'] - out['bb_lower']).replace(0, np.nan)
        # BB Squeeze: Bandbreite sehr eng = Compression
        out['bb_width'] = (out['bb_upper'] - out['bb_lower']) / out['close'].replace(0, np.nan)
        out['bb_width_avg'] = out['bb_width'].rolling(20).mean()
        out['bb_squeeze'] = (out['bb_width'] < out['bb_width_avg'] * 0.75).fillna(False)
        # Setup Layer: War Compression in letzten 1-5 Kerzen aktiv?
        out['bb_squeeze_recent'] = (
            out['bb_squeeze'].shift(1).fillna(False) |
            out['bb_squeeze'].shift(2).fillna(False) |
            out['bb_squeeze'].shift(3).fillna(False) |
            out['bb_squeeze'].shift(4).fillna(False) |
            out['bb_squeeze'].shift(5).fillna(False)
        ).fillna(False)
        # ADX Range recent: Markt schlief in letzten 3 Kerzen
        out['adx_range_recent'] = (
            out['adx_range'].shift(1).fillna(False) |
            out['adx_range'].shift(2).fillna(False) |
            out['adx_range'].shift(3).fillna(False)
        ).fillna(False)
        # Vol-Spike und Rücksetzer — Entry nach Akkumulation nicht am Top
        # Vol-Spike und Rücksetzer — Entry nach Akkumulation nicht am Top

        out["volume_sma_20"] = out["volume"].rolling(20).mean()
        out["vol_ratio"] = (out["volume"] / out["volume_sma_20"].replace(0, np.nan)).fillna(0.0)

        # OFI — Order Flow Imbalance (bewiesen aus 291 Coin Analyse)
        # Bulk Volume Classification: (close-low)/(high-low) = Buy Ratio
        _range   = (out["high"] - out["low"]).replace(0, np.nan)
        _buy_r   = ((out["close"] - out["low"]) / _range).fillna(0.5)
        _buy_vol  = out["volume"] * _buy_r
        _sell_vol = out["volume"] * (1 - _buy_r)
        _ofi_n    = 10
        _ofi_sum  = (_buy_vol - _sell_vol).rolling(_ofi_n).sum()
        _vol_sum  = out["volume"].rolling(_ofi_n).sum().replace(0, np.nan)
        out["ofi"]         = (_ofi_sum / _vol_sum).fillna(0.0)
        out["ofi_bullish"] = out["ofi"] > 0.10   # Käufer dominieren
        out["ofi_bearish"] = out["ofi"] < -0.10  # Verkäufer dominieren
        # VPIN — Volume-Synchronized Probability of Informed Trading
        # Bulk Volume Classification aus OHLCV (Cornell Paper Easley/O'Hara)
        _range = (out['high'] - out['low']).replace(0, np.nan)
        _body = out['close'] - out['open']
        # Buy Volume: proportional zum bullischen Anteil der Kerze
        _buy_vol = out['volume'] * ((_body.clip(lower=0)) / _range).fillna(0.5)
        _sell_vol = out['volume'] - _buy_vol
        # VPIN = rolling |Buy - Sell| / Total über 30 Kerzen (2.5h)
        _n = 30
        _imbalance = (_buy_vol - _sell_vol).abs().rolling(_n).sum()
        _total = out['volume'].rolling(_n).sum().replace(0, np.nan)
        out['vpin'] = (_imbalance / _total).fillna(0.0)
        # VPIN Regime-Klassifikation
        out['vpin_balanced'] = out['vpin'] < 0.35          # Range/Reversal
        out['vpin_active']   = (out['vpin'] >= 0.35) & (out['vpin'] < 0.55)  # Normal
        out['vpin_toxic']    = out['vpin'] >= 0.55          # Trend/Breakout
        out['vpin_extreme']  = out['vpin'] >= 0.70          # Extreme Move loading
        # VPIN steigt = Move lädt sich auf
        out['vpin_rising'] = (out['vpin'] > out['vpin'].shift(3).fillna(0))

        # CVD — echte WebSocket-Daten wenn verfügbar, sonst OHLCV-Proxy
        import json as _json
        _pair_name = getattr(self, '_current_pair', '') or ''
        _cvd_path = Path('data/cvd') / (_pair_name.replace('/','_') + '_cvd.json')
        _cvd_loaded = False
        if _cvd_path.exists():
            try:
                _cvd_data = _json.loads(_cvd_path.read_text())
                _cvd5  = float(_cvd_data.get('cvd_5', 0) or 0)
                _cvd10 = float(_cvd_data.get('cvd_10', 0) or 0)
                _cvd_r = bool(_cvd_data.get('cvd_rising', False))
                _cvd_f = bool(_cvd_data.get('cvd_falling', False))
                # CVD Fix 18.06.2026: history-Array fuer echte Zeitreihe nutzen
                # ALT: Scalar-Broadcast — alle Zeilen bekamen denselben Wert
                # -> bimodale 0/1-Verteilung, historische Analyse bedeutungslos
                # Scalar-Felder unveraendert (aktuellster Snapshot reicht)
                out['cvd_5']    = _cvd5
                out['cvd_10']   = _cvd10
                out['cvd_rising']  = _cvd_r
                out['cvd_falling'] = _cvd_f
                # Scalar-Defaults fuer Fallback
                _cvd_buildup = float(_cvd_data.get('buildup_score', 0) or 0)
                _cvd_trend6  = float(_cvd_data.get('cvd_trend_6', 0) or 0)
                _cvd_bsr     = float(_cvd_data.get('buy_sell_ratio', 1) or 1)
                # History-basierte Zeitreihe (Producer schreibt letzte 10 Kerzen)
                _hist = _cvd_data.get('history', [])
                # Defaults - werden nur bei erfolgreichem Bucket-Mapping unten
                # ueberschrieben (Patch E). Bleiben sonst 0.0 (zu wenig History
                # oder Mapping schlaegt fehl - selbe Faelle wie cvd_buildup etc.).
                out['ofi_real'] = 0.0
                out['vpin_real'] = 0.0
                if len(_hist) >= 2 and 'datetime' in out.columns:
                    import pandas as _pd_cvd
                    _hdf = _pd_cvd.DataFrame(_hist).copy()
                    _hdf['ts'] = _pd_cvd.to_datetime(_hdf['ts'], unit='ms', utc=True)
                    _hdf = _hdf.sort_values('ts').drop_duplicates('ts').reset_index(drop=True)
                    # Rolling buildup_score pro History-Eintrag berechnen
                    _b_scores, _t6_vals, _bsr_vals = [], [], []
                    # ECHTER OFI/VPIN 16.07.2026: aus echtem buy_vol/sell_vol
                    # statt Kerzenform-Naeherung. Fenster = alles was das Live-
                    # JSON vorhaelt (max 10 Buckets=~50min) - bewusst NICHT 1:1
                    # Ersatz fuer bestehendes vpin (30 Kerzen=2.5h), eigene Felder
                    # neben den bestehenden fuer sauberen A/B-Vergleich.
                    _ofi_vals, _vpin_vals = [], []
                    for _i in range(len(_hdf)):
                        _w = _hdf.iloc[max(0, _i-5):_i+1]
                        _cv = _w['cvd'].tolist()
                        _bv = _w['buy_vol'].tolist()
                        _sv = _w['sell_vol'].tolist()
                        _t6v = _cv[-1] - _cv[0] if len(_cv) >= 2 else 0.0
                        _bv2 = sum(_bv); _sv2 = sum(_sv) + 0.001
                        _bsrv = _bv2 / _sv2
                        _posv = sum(1 for c in _cv if c > 0)
                        _sc = int(_t6v > 0) + int(_bsrv > 1.5) + int(_posv >= 4)
                        _b_scores.append(_sc)
                        _t6_vals.append(_t6v)
                        _bsr_vals.append(_bsrv)
                        _w10 = _hdf.iloc[max(0, _i-9):_i+1]
                        _bv3 = sum(_w10['buy_vol'].tolist())
                        _sv3 = sum(_w10['sell_vol'].tolist())
                        _tot3 = _bv3 + _sv3
                        _ofi_vals.append((_bv3 - _sv3) / _tot3 if _tot3 > 0 else 0.0)
                        _vpin_vals.append(abs(_bv3 - _sv3) / _tot3 if _tot3 > 0 else 0.0)
                    _hdf['_bs'] = _b_scores
                    _hdf['_t6'] = _t6_vals
                    _hdf['_bsr'] = _bsr_vals
                    _hdf['_ofi'] = _ofi_vals
                    _hdf['_vpin'] = _vpin_vals
                    # Out-Zeitstempel auf 5min-Bucket mappen
                    try:
                        _out_dt = _pd_cvd.to_datetime(out['datetime'], utc=True, errors='coerce')
                        # CVD-Bucket-Fix 30.06.2026: .astype('int64') auf tz-aware
                        # Pandas-Spalten liefert je nach Pandas-Version Mikro- statt
                        # Nanosekunden (datetime64[us,UTC] ist neuer Standard) -
                        # alte Formel quetschte dadurch ALLE Zeilen auf denselben
                        # Bucket, das gesamte history-Mapping fiel still auf den
                        # Scalar-Fallback zurueck (cvd_trend_6/buildup/bsr waren
                        # ueber Wochen komplett konstant, kein echtes Signal).
                        _ts_ms = _out_dt.dt.tz_localize(None).astype('datetime64[ns]').astype('int64') // 1_000_000
                        _out_bkt_ms = (_ts_ms // 300_000) * 300_000
                        # CVD-Bucket-Fix 30.06.2026 Teil 2: gleicher us/ns-Bug wie oben,
                        # hier in der history-Lookup-Tabelle selbst - ohne diesen
                        # zweiten Fix waeren ts_ms-Werte 1000x zu klein und nicht
                        # mehr eindeutig (mehrere history-Zeilen kollabieren auf
                        # denselben verkuerzten Schluessel), .map() schlaegt fehl
                        # und faellt im aeusseren try/except still auf den Scalar
                        # zurueck - identisches Symptom wie der erste Fund.
                        _hdf['ts_ms'] = _hdf['ts'].dt.tz_localize(None).astype('datetime64[ns]').astype('int64') // 1_000_000
                        _lkp_b = _hdf.set_index('ts_ms')['_bs']
                        _lkp_t = _hdf.set_index('ts_ms')['_t6']
                        _lkp_r = _hdf.set_index('ts_ms')['_bsr']
                        out['cvd_buildup']        = _out_bkt_ms.map(_lkp_b).ffill().fillna(_cvd_buildup)
                        out['cvd_trend_6']        = _out_bkt_ms.map(_lkp_t).ffill().fillna(_cvd_trend6)
                        out['cvd_buy_sell_ratio'] = _out_bkt_ms.map(_lkp_r).ffill().fillna(_cvd_bsr)
                        _lkp_ofi  = _hdf.set_index('ts_ms')['_ofi']
                        _lkp_vpin = _hdf.set_index('ts_ms')['_vpin']
                        out['ofi_real']  = _out_bkt_ms.map(_lkp_ofi).ffill().fillna(0.0)
                        out['vpin_real'] = _out_bkt_ms.map(_lkp_vpin).ffill().fillna(0.0)
                    except Exception:
                        out['cvd_buildup']        = _cvd_buildup
                        out['cvd_trend_6']        = _cvd_trend6
                        out['cvd_buy_sell_ratio'] = _cvd_bsr
                else:
                    # Zu wenig History oder kein datetime -> Scalar-Fallback
                    out['cvd_buildup']        = _cvd_buildup
                    out['cvd_trend_6']        = _cvd_trend6
                    out['cvd_buy_sell_ratio'] = _cvd_bsr
                out['cvd_accumulation'] = out['cvd_buildup'] >= 2
                out['ofi_real_bullish'] = out['ofi_real'] > 0.10
                out['ofi_real_bearish'] = out['ofi_real'] < -0.10
                out['cvd_real_available'] = True
                _cvd_loaded = True
            except: pass
        if not _cvd_loaded:
            # Fallback: OHLCV-Proxy
            _hl    = (out['high'] - out['low']).replace(0, float('nan'))
            _delta = ((out['close'] - out['open']) / _hl * out['volume']).fillna(0)
            out['cvd_5']       = _delta.rolling(5).sum()
            out['cvd_10']      = _delta.rolling(10).sum()
            out['cvd_rising']  = out['cvd_5'] > out['cvd_5'].shift(2).fillna(0)
            out['cvd_falling'] = out['cvd_5'] < out['cvd_5'].shift(2).fillna(0)
            out['ofi_real'] = np.nan
            out['vpin_real'] = np.nan
            out['ofi_real_bullish'] = False
            out['ofi_real_bearish'] = False
            out['cvd_real_available'] = False
        # Divergenz berechnen
        _price_up = out['close'] > out['close'].shift(3)
        _price_dn = out['close'] < out['close'].shift(3)
        out['cvd_bear_div']    = _price_up & out['cvd_falling']
        out['cvd_bull_div']    = _price_dn & out['cvd_rising']
        _vol_low = out['volume'] < out['volume'].rolling(10).mean() * 0.60
        out['shakeout_signal'] = _price_dn & _vol_low & out['cvd_bull_div']
        _vol_sma = out['volume'].rolling(6).mean().replace(0, np.nan)
        _vol_slope = out['volume'].rolling(6).apply(lambda x: np.polyfit(range(len(x)),x,1)[0] if len(x)==6 else 0, raw=True)
        out['vol_trend_pct'] = (_vol_slope / _vol_sma * 100).fillna(0)
        out['vol_accelerating'] = out['vol_trend_pct'] > 30

        out["ret_1"] = out["close"].pct_change().fillna(0.0)
        out["ret_3"] = out["close"].pct_change(3).fillna(0.0)
        out["ret_12"] = out["close"].pct_change(12).fillna(0.0)

        out["hh_20_prev"] = out["high"].rolling(20).max().shift(1)
        out["ll_20_prev"] = out["low"].rolling(20).min().shift(1)

        out["trend_up"] = (
            (out["ema_20"] > out["ema_50"]) &
            (out["ema_50"] > out["ema_200"])
        )

        out["trend_down"] = (
            (out["ema_20"] < out["ema_50"]) &
            (out["ema_50"] < out["ema_200"])
        )

        out["breakout_up"] = (
            (out["close"] > out["hh_20_prev"]) &
            (out["vol_ratio"] > 1.2)
        ).fillna(False)


        out["breakout_down"] = (
            (out["close"] < out["ll_20_prev"]) &
            (out["vol_ratio"] > 1.2)
        ).fillna(False)

        # NEW CORE: DIF / DEA / VWAP / OBV / ATR%
        ema_fast = out["close"].ewm(span=12, adjust=False).mean()
        ema_slow = out["close"].ewm(span=26, adjust=False).mean()
        out["dif"] = ema_fast - ema_slow
        out["dea"] = out["dif"].ewm(span=9, adjust=False).mean()
        out["macd_hist"] = out["dif"] - out["dea"]

        out["dif_above_dea"] = (out["dif"] > out["dea"]).fillna(False)
        out["dif_cross_up_dea"] = (
            (out["dif"].shift(1) <= out["dea"].shift(1)) &
            (out["dif"] > out["dea"])
        ).fillna(False)
        out["dif_cross_down_dea"] = (
            (out["dif"].shift(1) >= out["dea"].shift(1)) &
            (out["dif"] < out["dea"])
        ).fillna(False)
        out["dif_above_zero"] = (out["dif"] > 0).fillna(False)
        out["dif_below_zero"] = (out["dif"] < 0).fillna(False)
        out["hist_rising"] = (out["macd_hist"].diff() > 0).fillna(False)

        delta = out["close"].diff().fillna(0.0)
        signed_volume = np.where(delta > 0, out["volume"], np.where(delta < 0, -out["volume"], 0.0))
        out["obv"] = pd.Series(signed_volume, index=out.index).cumsum()
        out["obv_ema"] = out["obv"].ewm(span=21, adjust=False).mean()
        out["obv_up"] = (out["obv"] > out["obv_ema"]).fillna(False)
        out["obv_rising"] = (out["obv"].diff() > 0).fillna(False)

        dt = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
        session_key = dt.dt.strftime("%Y-%m-%d").fillna("unknown")
        typical = ((out["high"] + out["low"] + out["close"]) / 3.0).fillna(out["close"])
        tpv = typical * out["volume"].fillna(0.0)
        cum_vol = out["volume"].fillna(0.0).groupby(session_key).cumsum()
        out["vwap"] = (tpv.groupby(session_key).cumsum() / cum_vol.replace(0, np.nan)).ffill()
        out["above_vwap"] = (out["close"] > out["vwap"]).fillna(False)
        out["below_vwap"] = (out["close"] < out["vwap"]).fillna(False)
        # Docht-Logik
        total_range = (out['high'] - out['low']).replace(0, np.nan)
        upper_wick = out['high'] - out[['open', 'close']].max(axis=1)
        lower_wick = out[['open', 'close']].min(axis=1) - out['low']
        body = (out['close'] - out['open']).abs()
        # Bullischer Docht: langer unterer Docht = Käufer haben Boden gefunden
        out['bullish_wick'] = (
            (lower_wick > total_range * 0.60) &
            (body < total_range * 0.30) &
            (out['vol_ratio'] > out['vol_ratio'].rolling(20).mean())
        ).fillna(False)
        # Bärischer Docht: langer oberer Docht = Verkäufer haben Top gehalten
        out['bearish_wick'] = (
            (upper_wick > total_range * 0.60) &
            (body < total_range * 0.30) &
            (out['vol_ratio'] > out['vol_ratio'].rolling(20).mean())
        ).fillna(False)
        vol = out['vol_ratio']
        out['vol_spike_and_hold'] = (
            (vol.shift(2) > 1.50) &
            (vol.shift(1) > 1.50) &
            (vol > 1.10) &
            (vol < 1.30) &
            out['above_vwap']
        ).fillna(False)
        # Range-Bottom Detection: StochRSI + BB Lower
        rsi_s = out['rsi_14']
        _stoch_k = ((rsi_s - rsi_s.rolling(14).min()) / (rsi_s.rolling(14).max() - rsi_s.rolling(14).min()).replace(0, np.nan)).rolling(3).mean() * 100
        out['range_bottom_hit'] = ((_stoch_k < 20) & (out['close'] <= out['bb_lower'])).fillna(False)
        out['range_top_hit'] = ((_stoch_k > 80) & (out['close'] >= out['bb_upper'])).fillna(False)
        out["vwap_reclaim"] = (
            (out["close"] > out["vwap"]) &
            (out["close"].shift(1) <= out["vwap"].shift(1))
        ).fillna(False)
        out["vwap_reject"] = (
            (out["close"] < out["vwap"]) &
            (out["close"].shift(1) >= out["vwap"].shift(1))
        ).fillna(False)

        out["atr_pct"] = (out["atr_14"] / out["close"].replace(0, np.nan)).fillna(0.0)
        out["stronger_than_btc"] = True
        out["rel_strength_3"] = 0.0
        out["rel_strength_12"] = 0.0

        # Liquidity Sweep — Stops wurden geholt, dann Reclaim
        local_low_5 = out["low"].rolling(5).min()
        local_low_10 = out["low"].rolling(10).min()
        out["liquidity_sweep_low"] = (
            (out["low"].shift(1) < local_low_5.shift(2)) &
            (out["close"] > local_low_5.shift(1)) &
            (out["close"] > out["close"].shift(1))
        ).fillna(False)
        out["liquidity_sweep_low_10"] = (
            (out["low"].shift(1) < local_low_10.shift(2)) &
            (out["close"] > local_low_10.shift(1)) &
            (out["close"] > out["close"].shift(1))
        ).fillna(False)
        out["swept_and_reclaimed"] = (
            out["liquidity_sweep_low"] | out["liquidity_sweep_low_10"]
        ).fillna(False)

        # Orderblock: letzte rote Kerze vor impulsivem Anstieg
        # Bullischer OB = letzte bearishe Kerze vor starkem Aufwärtsimpuls
        bearish_candle = out["close"] < out["open"]
        bullish_impulse = (
            (out["close"].shift(-1) > out["open"].shift(-1)) &
            (out["close"].shift(-1) - out["open"].shift(-1) > out["atr_14"].shift(-1) * 0.8)
        )
        out["orderblock_bull"] = (bearish_candle & bullish_impulse).fillna(False)

        # Preis kehrt in OB-Zone zurück (Retest)
        ob_high = out["high"].where(out["orderblock_bull"]).ffill()
        ob_low  = out["low"].where(out["orderblock_bull"]).ffill()
        out["in_orderblock_zone"] = (
            (out["close"] >= ob_low) &
            (out["close"] <= ob_high * 1.005)
        ).fillna(False)

        # Fair Value Gap (FVG): Lücke zwischen Kerze i-2 high und Kerze i low
        out["fvg_bull"] = (
            out["low"] > out["high"].shift(2)
        ).fillna(False)
        out["fvg_bear"] = (
            out["high"] < out["low"].shift(2)
        ).fillna(False)

        # Change of Character (ChoCh): erstes höheres Tief + Bruch des letzten Hochs
        higher_low = (
            (out["low"] > out["low"].shift(1)) &
            (out["low"].shift(1) > out["low"].shift(2))
        ).fillna(False)
        structure_break = out["close"] > out["high"].rolling(5).max().shift(1)
        out["choch_bull"] = (higher_low & structure_break).fillna(False)

        # Kombinierter Level-3-Kontext
        out["l3_context"] = (
            out["swept_and_reclaimed"] |
            out["in_orderblock_zone"] |
            out["choch_bull"]
        ).fillna(False)


        out["pullback_long"] = (
            # Bias: über EMA200 — übergeordnet bullisch
            (out["close"] > out["ema_200"]) &
            # Rücksetzer: unter EMA20 ODER unter VWAP
            (
                (out["close"] <= out["ema_20"]) |
                (out["close"] <= out["vwap"])
            ) &
            # Hält über EMA50 — Trend noch intakt
            (out["close"] > out["ema_50"]) &
            # MACD noch bullisch
            out["dif_above_dea"] &
            # VPIN moderat — kein extremer toxischer Flow
            (out["vpin"] < 0.65) &
            # RSI — Rücksetzer aber nicht Crash
            (out["rsi_14"] > 35) &
            (out["rsi_14"] < 65) &
            # Vol moderat — echter Rücksetzer, kein Absturz
            (out["vol_ratio"] > 0.80) &
            (out["vol_ratio"] < 3.00) &
            # Mindest-Volatilität
            (out["atr_pct"] > 0.0012)
        ).fillna(False)

        # range_regime Fix 18.06.2026: dif/macd_hist auf close-Preis normiert
        # ALT: absolute Schwellen — fuer BTC (close~62k) strukturell nie erfuellt,
        #      fuer Cent-Coins trivial immer True. Auskommentiert:
        # out["range_regime"] = (
        #     (out["atr_14"] / out["close"].replace(0, np.nan) < 0.010) &
        #     (out["dif"].abs() < 0.5) &     # BUG: absolut, nicht relativ
        #     (out["macd_hist"].abs() < 0.3)  # BUG: absolut, nicht relativ
        # ).fillna(False)
        _close_safe = out["close"].replace(0, np.nan)
        out["range_regime"] = (
            (out["atr_14"] / _close_safe < 0.010) &
            ((out["dif"] / _close_safe).abs() < 0.005) &
            ((out["macd_hist"] / _close_safe).abs() < 0.003)
        ).fillna(False)





        _trend_raw = (
            # Übergeordnet bullisch — über EMA200
            (out["close"] > out["ema_200"]) &
            # TREND-BEGINN: noch unter oder nahe VWAP — kauft Discount, nicht Top
            # Daten: Wins above_vwap=0%, HS above_vwap=100%
            ~out["above_vwap"] &
            # ADX lädt sich auf — noch nicht überhitzt
            # Daten: Wins ADX=12.9, HS ADX=40.9
            (out["adx"] > 12) & (out["adx"] < 35) &
            # MACD Momentum aufbauend
            out["dif_above_dea"] &
            # OBV bestätigt Kaufdruck
            out["obv_rising"] &
            # RSI neutral-moderat — nicht überkauft
            # Daten: Wins RSI=52.7, HS RSI=73.2
            (out["rsi_14"] > 38) & (out["rsi_14"] < 62) &
            # Volumen zieht an
            (out["vol_ratio"] > 1.0) &
            (out["vol_ratio"] < 4.00) &
            # Mindest-Volatilität
            (out["atr_pct"] > 0.0015) &
            # BB nicht überstreckt
            (out["bb_percent"] < 0.85) &
            # Kein bearischer CVD
            ~out.get("cvd_bear_div", __import__("pandas").Series(False, index=out.index))
        ).fillna(False)
        # Trend Setup Layer:
        # Trend soll fruehen Trendaufbau / geordnete Trendfortsetzung handeln.
        # Der alte 2-Kerzen-Confirm hat signal_trend komplett totgefiltert.
        _trend_near_vwap = (
            ((out["close"] / out["vwap"].replace(0, np.nan) - 1.0).abs() <= 0.006)
        ).replace([np.inf, -np.inf], np.nan).fillna(False)

        out["trend_vwap_near"] = (
            (out["close"] > out["ema_200"]) &
            (((out["close"] <= out["vwap"]) | _trend_near_vwap)) &
            (out["adx"] > 12) & (out["adx"] < 35) &
            out["dif_above_dea"] &
            out["obv_rising"] &
            (out["rsi_14"] > 38) & (out["rsi_14"] < 62) &
            (out["vol_ratio"] > 0.80) &
            (out["vol_ratio"] < 4.00) &
            (out["atr_pct"] > 0.0015) &
            (out["bb_percent"] < 0.85) &
            ~out.get("cvd_bear_div", __import__("pandas").Series(False, index=out.index))
        ).fillna(False)

        out["trend_vol_080"] = (
            (out["close"] > out["ema_200"]) &
            (out["close"] <= out["vwap"]) &
            (out["adx"] > 12) & (out["adx"] < 35) &
            out["dif_above_dea"] &
            out["obv_rising"] &
            (out["rsi_14"] > 38) & (out["rsi_14"] < 62) &
            (out["vol_ratio"] > 0.80) &
            (out["vol_ratio"] < 4.00) &
            (out["atr_pct"] > 0.0015) &
            (out["bb_percent"] < 0.85) &
            ~out.get("cvd_bear_div", __import__("pandas").Series(False, index=out.index))
        ).fillna(False)

        out["trend_continuation"] = (
            (out["close"] > out["ema_200"]) &
            (out["close"] > out["ema_20"] * 0.995) &
            (out["ema_20"] > out["ema_50"]) &
            (out["adx"] > 12) & (out["adx"] < 35) &
            out["dif_above_dea"] &
            out["obv_rising"] &
            (out["rsi_14"] > 38) & (out["rsi_14"] < 62) &
            (out["vol_ratio"] > 0.75) &
            (out["vol_ratio"] < 3.00) &
            (out["atr_pct"] > 0.0015) &
            (out["bb_percent"] < 0.78) &
            ~out.get("cvd_bear_div", __import__("pandas").Series(False, index=out.index))
        ).fillna(False)

        _trend_score_checks = [
            (out["close"] > out["ema_200"]),
            ((out["close"] <= out["vwap"]) | _trend_near_vwap),
            ((out["adx"] > 12) & (out["adx"] < 35)),
            out["dif_above_dea"],
            out["obv_rising"],
            ((out["rsi_14"] > 38) & (out["rsi_14"] < 62)),
            ((out["vol_ratio"] > 0.80) & (out["vol_ratio"] < 4.00)),
            (out["atr_pct"] > 0.0015),
            (out["bb_percent"] < 0.85),
            ~out.get("cvd_bear_div", __import__("pandas").Series(False, index=out.index)),
        ]
        out["trend_setup_score"] = sum(x.astype(int) for x in _trend_score_checks)

        # Aktives Trend-Signal:
        # frueher Trendaufbau nahe VWAP, nicht mehr 2-Kerzen-Confirm.
        out["signal_trend"] = out["trend_vwap_near"].fillna(False)
        # Pfad A: Initial-Break (bisherige Logik, unveraendert)
        out["breakout_initial"] = (
            out["bb_squeeze_recent"] &
            (out["close"] > out["hh_20_prev"]) &
            (out["vol_ratio"] > 1.40) &
            (out["bb_percent"] > 0.50) &
            (out["bb_percent"] < 0.95) &
            (out["close"] > out["ema_200"]) &
            (out["close"] > out["vwap"]) &
            ~out["vpin_rising"] &
            ~out["ofi_bullish"] &
            ~out.get("signal_bull_flag", pd.Series(False, index=out.index)) &
            ~out.get("vp_above_vah", pd.Series(False, index=out.index)) &
            ~out.get("vol_accelerating", pd.Series(False, index=out.index)) &
            out["obv_up"] &
            (out["atr_pct"] > 0.0015)
        ).fillna(False)
        # Pfad B: Retest-Entry (Profi-Standard: Break -> leiser Pullback
        # zur Zone -> Zone haelt -> Bestaetigungskerze)
        _brk_now   = (out["close"] > out["hh_20_prev"]).fillna(False)
        _brk_event = (_brk_now & ~_brk_now.shift(1).fillna(False)).fillna(False)
        _brk_level = out["hh_20_prev"].where(_brk_event).ffill()
        _idx       = pd.Series(range(len(out)), index=out.index, dtype=float)
        _brk_idx   = _idx.where(_brk_event).ffill()
        _since_brk = _idx - _brk_idx
        out["breakout_retest"] = (
            # Break war vor 3-12 Kerzen (15-60min)
            (_since_brk >= 3) & (_since_brk <= 12) &
            # Preis zurueck an der Zone
            (out["close"] <= _brk_level * 1.005) &
            # Zone haelt — Low bricht nicht durch
            (out["low"] > _brk_level * 0.997) &
            # Retest leise — Profi-Regel: Pullback auf niedrigem Volumen
            (out["vol_ratio"] < 1.00) &
            # Bestaetigungskerze: bullisch, schliesst ueber dem Level
            (out["close"] > out["open"]) &
            (out["close"] > _brk_level) &
            # Guards
            (out["close"] > out["ema_200"]) &
            ~out["vpin_rising"] &
            ~out["ofi_bullish"] &
            (out["atr_pct"] > 0.0015)
        ).fillna(False)
        # Breakout Diagnose-/Quality-Layer
        # Wichtig:
        # Diese Spalten aendern signal_breakout NICHT.
        # Sie machen nur sichtbar, ob der Breakout-Retest wirklich sauber ist.
        _brk_level_safe = _brk_level.replace(0, np.nan)
        _brk_rng = (out["high"] - out["low"]).replace(0, np.nan)
        _brk_body_top = out[["open", "close"]].max(axis=1)

        out["breakout_level"] = _brk_level.fillna(0.0)
        out["breakout_age_5m"] = _since_brk.fillna(-1.0)

        out["breakout_dist_to_level_pct"] = (
            ((out["close"] / _brk_level_safe) - 1.0) * 100.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        out["breakout_low_vs_level_pct"] = (
            ((out["low"] / _brk_level_safe) - 1.0) * 100.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        out["breakout_close_pos"] = (
            (out["close"] - out["low"]) / _brk_rng
        ).replace([np.inf, -np.inf], np.nan).fillna(0.5)

        out["breakout_upper_wick"] = (
            (out["high"] - _brk_body_top) / _brk_rng
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        _bo_age_ok = ((_since_brk >= 3) & (_since_brk <= 12)).fillna(False)
        _bo_dist_ok = (
            (out["breakout_dist_to_level_pct"] >= -0.05) &
            (out["breakout_dist_to_level_pct"] <= 0.30)
        ).fillna(False)
        _bo_low_hold_ok = (out["breakout_low_vs_level_pct"] > -0.30).fillna(False)
        _bo_close_pos_ok = (out["breakout_close_pos"] >= 0.65).fillna(False)
        _bo_wick_ok = (out["breakout_upper_wick"] <= 0.35).fillna(False)
        _bo_vol_ok = ((out["vol_ratio"] >= 0.20) & (out["vol_ratio"] <= 1.20)).fillna(False)
        _bo_bb_ok = ((out["bb_percent"] >= 0.50) & (out["bb_percent"] <= 0.92)).fillna(False)
        _bo_above_level = (out["close"] > _brk_level_safe).fillna(False)

        out["breakout_retest_score"] = (
            _bo_age_ok.astype(int) +
            _bo_dist_ok.astype(int) +
            _bo_low_hold_ok.astype(int) +
            _bo_close_pos_ok.astype(int) +
            _bo_wick_ok.astype(int) +
            _bo_vol_ok.astype(int) +
            _bo_bb_ok.astype(int) +
            _bo_above_level.astype(int)
        ).fillna(0).astype(int)

        # Parallel-Signal zum Beobachten.
        # Wird aktuell NICHT im Router benutzt.
        out["signal_breakout_quality"] = (
            out["breakout_retest"] &
            (out["breakout_retest_score"] >= 6) &
            _bo_dist_ok &
            _bo_low_hold_ok &
            _bo_above_level
        ).fillna(False)

        out["signal_breakout"] = (
            out["breakout_initial"] | out["breakout_retest"]
        ).fillna(False)

        out["hidden_div_bull"] = (
            (out["low"] > out["low"].rolling(10).min().shift(1)) &   # Preis höheres Tief
            (out["rsi_14"] > out["rsi_14"].rolling(10).min().shift(1)) &  # RSI höheres Tief
            (out["close"] > out["ema_200"]) &                # Aufwärtstrend
            out["dif_above_dea"]                             # Momentum intakt
        ).fillna(False)

        out["wyckoff_spring"] = (
            (out["low"] < out["low"].rolling(20).min().shift(1) * 0.998) &
            (out["close"] > out["low"].rolling(20).min().shift(1)) &
            (out["bb_percent"] < 0.25) &
            out["below_vwap"].shift(1).fillna(False) &
            (out["vol_ratio"] > 1.20)
        ).fillna(False)
        out["signal_bull_flag"] = (
            # dif_above_dea 99% der Bull Flags
            out["dif_above_dea"] &
            # above_vwap 86%
            out["above_vwap"] &
            # obv_rising 94%
            out["obv_rising"] &
            # ADX Trend 84%
            out["adx_trend"] &
            # bb_percent_hi 83% — Preis am oberen Band
            (out["bb_percent"] > 0.70) &
            # RSI nicht extrem überkauft
            (out["rsi_14"] < 80)
        ).fillna(False)

        out["signal_pullback"] = out["pullback_long"].fillna(False)
        out["signal_range"] = (
            out["range_regime"] &
            out["adx_range"] &
            # VPIN: kein extremer Flow — echter Range Markt
            (out["vpin"] < 0.50) &
            # RSI in Mitte — 82% der echten Ranges
            (out["rsi_14"] > 35) & (out["rsi_14"] < 60) &
            out["below_vwap"] &
            out["dif_cross_up_dea"] &
            out["obv_rising"] &
            (out["vol_ratio"] > 0.60) &
            (out["vol_ratio"] < 1.50) &
            (out["atr_pct"] > 0.0008)
        ).fillna(False)
        # Momentum — repariert 25.06.2026:
        # Ziel: frischen MA-Reclaim / Start-Impuls kaufen,
        # nicht fertige Pump-Kerzen am oberen Bollinger-Band.
        out["momentum_ma7"] = out["close"].rolling(7).mean()
        out["momentum_ma25"] = out["close"].rolling(25).mean()
        out["momentum_ma99"] = out["close"].rolling(99).mean()

        _mom_range = (out["high"] - out["low"]).replace(0, np.nan)
        out["momentum_close_pos"] = ((out["close"] - out["low"]) / _mom_range).fillna(0.5)
        out["momentum_upper_wick_pct"] = ((out["high"] - out["close"]) / _mom_range).fillna(0.0)

        # MA-Struktur wie im Chart:
        # MA7 dreht hoch, Preis reclaimt MA7/MA25-Bereich, Kerze wird gekauft.
        out["momentum_ma7_turn_up"] = (
            (out["momentum_ma7"] > out["momentum_ma7"].shift(1)) &
            (out["momentum_ma7"].shift(1) <= out["momentum_ma7"].shift(3))
        ).fillna(False)

        out["momentum_ma_reclaim"] = (
            (out["close"] > out["open"]) &
            (out["close"] > out["close"].shift(1)) &
            (out["close"] > out["momentum_ma7"]) &
            (out["close"] > out["momentum_ma25"] * 0.995) &
            (out["momentum_close_pos"] > 0.60)
        ).fillna(False)

        # Kompression / enger MA-Bereich vor dem Impuls.
        out["momentum_ma_compression"] = (
            ((out["momentum_ma7"] / out["momentum_ma25"].replace(0, np.nan) - 1.0).abs() < 0.025) |
            ((out["momentum_ma25"] / out["momentum_ma99"].replace(0, np.nan) - 1.0).abs() < 0.045)
        ).fillna(False)

        # Flow muss reinkommen, aber kein komplett überdrehter Blowoff.
        out["momentum_flow_ok"] = (
            (out["vol_ratio"] > 1.15) &
            (out["vol_ratio"] < 4.20) &
            (out["obv_rising"] | out["obv_up"]) &
            ~out["vpin_rising"]
        ).fillna(False)

        # Momentum technisch intakt.
        out["momentum_direction_ok"] = (
            ((out["dif"] > out["dea"]) | out["dif_cross_up_dea"]) &
            (out["adx"] > out["adx"].shift(2)) &
            (out["atr_pct"] > 0.0015)
        ).fillna(False)

        # Anti-Chase: blockiert fertige Pump-Käufe.
        out["momentum_dist_ma25"] = (
            out["close"] / out["momentum_ma25"].replace(0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        out["momentum_runup_6"] = (
            out["close"] / out["close"].shift(6).replace(0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # vp_poc_dist wird erst später in process_pair() ergänzt.
        # Hier deshalb sicherer Fallback; das echte VP-Gate bleibt im SignalRouter aktiv.
        _mom_vp_poc_dist = out.get("vp_poc_dist", pd.Series(0.0, index=out.index))

        out["momentum_not_chasing"] = (
            (out["bb_percent"] < 1.05) &
            ~((out["rsi_14"] > 68.0) & (out["bb_percent"] > 0.95)) &
            (out["rsi_14"] < 72.0) &
            (_mom_vp_poc_dist < 3.0) &
            (out["momentum_dist_ma25"] < 0.040) &
            (out["momentum_runup_6"] < 0.060) &
            (out["momentum_upper_wick_pct"] < 0.38)
        ).fillna(False)

        # Start-Impuls: entweder Squeeze/ADX-Range war da ODER MAs waren eng.
        out["momentum_start_context"] = (
            out["bb_squeeze_recent"] |
            out["adx_range_recent"] |
            out["momentum_ma_compression"]
        ).fillna(False)

        # Fresh-Window:
        # Der Start passiert oft ueber 2-4 Kerzen, nicht in einer perfekten Einzelkerze.
        out["momentum_ma7_turn_recent"] = (
            out["momentum_ma7_turn_up"].rolling(3).max().fillna(0).astype(bool)
        )

        out["momentum_ma_reclaim_recent"] = (
            out["momentum_ma_reclaim"].rolling(3).max().fillna(0).astype(bool)
        )

        out["momentum_flow_recent"] = (
            out["momentum_flow_ok"].rolling(3).max().fillna(0).astype(bool)
        )

        out["momentum_direction_recent"] = (
            out["momentum_direction_ok"].rolling(2).max().fillna(0).astype(bool)
        )

        # Momentum Score Entry:
        # Nicht mehr jede Einzelbedingung hart erzwingen.
        # Beispiel SYN: 8/9 Setup darf feuern, wenn nur die spaete DIF/ADX-Bestaetigung fehlt.
        out["momentum_ma7_turn_recent"] = (
            out["momentum_ma7_turn_up"].rolling(4).max().fillna(0).astype(bool)
        )

        out["momentum_ma_reclaim_recent"] = (
            out["momentum_ma_reclaim"].rolling(4).max().fillna(0).astype(bool)
        )

        out["momentum_flow_recent"] = (
            out["momentum_flow_ok"].rolling(4).max().fillna(0).astype(bool)
        )

        out["momentum_direction_recent"] = (
            out["momentum_direction_ok"].rolling(3).max().fillna(0).astype(bool)
        )

        _mom_score_checks = [
            out["momentum_start_context"].astype(bool),
            out["momentum_ma7_turn_recent"].astype(bool),
            out["momentum_ma_reclaim_recent"].astype(bool),
            out["momentum_flow_recent"].astype(bool),
            out["momentum_direction_recent"].astype(bool),   # Bonus, nicht Pflicht
            out["momentum_not_chasing"].astype(bool),
            (out["close"] > out["momentum_ma7"]),
            (out["close"] > out["momentum_ma25"] * 0.995),
            (out["bb_percent"] < 1.05),
            (out["rsi_14"] < 72.0),
            (out["momentum_close_pos"] > 0.60),
            (out["vol_ratio"] > 1.05),
        ]

        out["momentum_entry_score"] = sum(s.astype(int) for s in _mom_score_checks)

        # Harte Pflichtbedingungen:
        # Ohne frischen Reclaim/Flow oder bei Chase kein Momentum-Entry.
        out["momentum_mandatory_ok"] = (
            out["momentum_start_context"] &
            out["momentum_ma_reclaim_recent"] &
            out["momentum_flow_recent"] &
            out["momentum_not_chasing"] &
            (out["close"] > out["momentum_ma7"]) &
            (out["close"] > out["momentum_ma25"] * 0.995)
        ).fillna(False)

        # Momentum 5m Setup-Score:
        # 5m findet den Move, aber der Entry wird spaeter im ExecutionManager
        # per 1m-Reclaim bestaetigt. Deshalb ist DIF/ADX nur Bonus, kein harter Blocker.
        out["momentum_ma7_turn_recent"] = (
            out["momentum_ma7_turn_up"].rolling(4).max().fillna(0).astype(bool)
        )

        out["momentum_ma_reclaim_recent"] = (
            out["momentum_ma_reclaim"].rolling(4).max().fillna(0).astype(bool)
        )

        out["momentum_flow_recent"] = (
            out["momentum_flow_ok"].rolling(4).max().fillna(0).astype(bool)
        )

        out["momentum_direction_recent"] = (
            out["momentum_direction_ok"].rolling(3).max().fillna(0).astype(bool)
        )

        _mom_score_checks = [
            out["momentum_start_context"].astype(bool),
            out["momentum_ma7_turn_recent"].astype(bool),
            out["momentum_ma_reclaim_recent"].astype(bool),
            out["momentum_flow_recent"].astype(bool),
            out["momentum_direction_recent"].astype(bool),
            out["momentum_not_chasing"].astype(bool),
            (out["close"] > out["momentum_ma7"]),
            (out["close"] > out["momentum_ma25"] * 0.995),
            (out["bb_percent"] < 1.05),
            (out["rsi_14"] < 72.0),
            (out["momentum_close_pos"] > 0.55),
            (out["vol_ratio"] > 1.05),
        ]

        out["momentum_entry_score"] = sum(s.astype(int) for s in _mom_score_checks)

        # Harte Pflicht auf 5m:
        # Setup muss frisch sein, Flow muss da sein, nicht chasen.
        # Der perfekte Entry kommt NICHT hier, sondern auf 1m.
        out["momentum_mandatory_ok"] = (
            out["momentum_start_context"] &
            out["momentum_ma_reclaim_recent"] &
            out["momentum_flow_recent"] &
            out["momentum_not_chasing"] &
            (out["close"] > out["momentum_ma7"]) &
            (out["close"] > out["momentum_ma25"] * 0.995)
        ).fillna(False)

        out["momentum_core_ok"] = (
            out["momentum_mandatory_ok"] &
            (out["momentum_entry_score"] >= 8)
        ).fillna(False)

        # Monitor-Score — Diagnose
        out["momentum_monitor_score"] = out["momentum_entry_score"].fillna(0).astype(int)

        # signal_momentum bedeutet: 5m-Setup ist aktiv.
        # Der echte Entry wird im ExecutionManager per 1m bestaetigt.
        #
        # Datencheck 01.07.2026 (VALIDIERUNG WAR ROHER FWD30, KEINE ECHTE
        # STOP/TRAIL-SIMULATION - erklaert vermutlich die Luecke unten):
        # original signal_momentum: n=1933, avg30=-0.0633%, med30=-0.0992%, up30=41.08%
        # original + ma7_turn_recent + direction_recent:
        # n=228, avg30=+0.0865%, med30=+0.0720%, up30=52.19%
        #
        # ECHTE LIVE-PERFORMANCE seit diesem Patch (302 Trades, echter Exit-Stack):
        # Netto=-0.2636%, Median=-0.2990%, WR=42.7% - deutlich schlechter als oben.
        #
        # NACHGEBESSERT 12.07.2026, gepaarter Backtest-Vergleich (echte Stop/Trail/
        # Promotion-Simulation, n_pairs~34, Fenster~1.45 Tage): direction_recent
        # isoliert getestet -> verschlechtert (core_ok allein WR=36.0%/Median-0.048%
        # vs. core_ok+direction_recent WR=26.4%/-0.210%). ma7_turn_recent isoliert
        # getestet -> verbessert (core_ok allein vs. +ma7_turn_recent: WR 36.0%->
        # 37.8%, Median -0.048%->-0.026%). RSI-Deckel obendrauf: monoton verbessernd
        # ueber die ganze Sweep-Reihe (WR 38.2%->54.4%, Netto -0.13%->-0.004% bei
        # RSI<=52, n=103). Naehert sich Breakeven, war aber NICHT klar positiv bei
        # dieser Stichprobengroesse - Zwei-Spur-Modell gilt, 30-50 Live-Trades noetig.
        # ADX-UNTERGRENZE ERGAENZT 07.08.2026.
        # BEFUND (n=1201 Trades mit ADX und ATR, disjunkte 3x4-Kreuztabelle):
        # ADX und ATR sind praktisch unkorreliert (+0.0526) und trennen
        # UNABHAENGIG voneinander. In JEDER ATR-Zeile steigt der WR mit ADX,
        # in JEDER ADX-Spalte steigt er mit ATR. Die Effekte addieren sich:
        #            ADX<20        ADX 20-25     ADX>=25
        #   V1_ruhig WR 31.9%      WR 40.7%      WR 41.0%
        #   V2       WR 39.8%      WR 39.6%      WR 46.4%
        #   V3       WR 41.3%      WR 41.0%      WR 52.4%
        #   V4_wild  WR 40.6%      WR 48.7%      WR 58.6%  (Netto nur -0.065%)
        # Alle 12 Zellen n>=64, keine Ausreisser-Struktur. Die 294 Trades bei
        # ADX<20 (24.5%) sind die schlechteste Teilmenge, alle Netto klar negativ.
        # DECKT SICH MIT DER LITERATUR: unter ADX 20 herrschen Range-Bedingungen,
        # in denen Momentum-Strategien zerhackt werden (Recherche 07.08.2026).
        # SCHWELLE 20, NICHT 25: 20 ist die konventionelle Range/Trend-Grenze und
        # entspricht der Empfehlung fuer Daytrading-Zeitrahmen. 25 waere der auf
        # UNSEREN Daten optimierte Wert - genau diese Sorte Optimierung hat schon
        # zweimal in die Irre gefuehrt (Liquiditaets-Cut 22.07., RSI-Deckel 13.07.).
        # UEBERSCHREIBT BEWUSST die adx_rising-Ausnahme im signal_router.py
        # (Zeile ~418/428, Juni eingebaut, damit momentum in Range-Phasen nicht
        # komplett blockiert). Genau ueber die kamen die ADX<20-Trades herein.
        # Der RSI<=52-Deckel BLEIBT: gegen die Lehrbuch-Definition geprueft
        # (Momentum kauft Staerke, nicht Schwaeche), aber die Daten stuetzen ihn -
        # RSI 45-52 ist in allen vier Kennzahlen die beste Gruppe (n=119,
        # WR 52.1%, Netto -0.088%), jede Gruppe ueber 52 schneidet schlechter ab.
        # ERWARTUNG: ~24% weniger Trades, WR von 42.9% auf ~45%.
        out["signal_momentum"] = (
            out["momentum_core_ok"] &
            out["momentum_ma7_turn_recent"] &
            (out["rsi_14"] <= 52) &
            (out["adx"] >= 20)
        ).fillna(False)

        out["signal_reversal"] = (
            out["dif_cross_up_dea"] &
            # VPIN: Flow war toxic, jetzt dreht er (Erschöpfung)
            (out["vpin"].shift(3).fillna(0) > 0.50) &
            # Panik-Volumen Pflicht — 96% der echten Reversals
            (out["vol_ratio"] > 2.0) &
            # Preis am unteren BB-Band — 90%
            (out["bb_percent"] < 0.20) &
            # Echter Oversold
            (out["rsi_14"] < 38) &
            (out["rsi_6"] > out["rsi_14"]) &
            # Sweep, Wick, Wyckoff Spring oder VWAP Reclaim
            (
                out["swept_and_reclaimed"] |
                out["swept_and_reclaimed"].shift(1).fillna(False) |
                out["swept_and_reclaimed"].shift(2).fillna(False) |
                out["bullish_wick"] |
                out["vwap_reclaim"] |
                out["wyckoff_spring"] |
                out["wyckoff_spring"].shift(1).fillna(False)
            ) &
            out["below_vwap"].shift(1).fillna(False) &
            (out["obv_rising"] | out["obv_up"]) &
            (out["atr_pct"] > 0.0012)
        ).fillna(False)
        # Squeeze Sniper: ADX-Squeeze + echter Volumen-Spike + DIF/DEA Cross
        out['signal_squeeze_sniper'] = (
            # vpin_toxic + ~ofi_bullish = Institutionen KAUFEN still
            out['vpin_toxic'] &
            ~out['ofi_bullish'] &
            # CVD Akkumulation bestätigt — stille Käufer seit 30min aktiv
            out.get('cvd_accumulation', __import__("pandas").Series(False, index=out.index)) &
            # MACD Cross — Momentum dreht gerade
            out['dif_cross_up_dea'] &
            # OBV bestätigt Kaufdruck
            out['obv_up'] &
            # Volumen-Spike
            (out['vol_ratio'] > 1.80) &
            # Preis über EMA200
            (out['close'] > out['ema_200']) &
            # Nicht überstreckt
            (out['bb_percent'] < 0.95) &
            (out['atr_pct'] > 0.0015)
        ).fillna(False)
        # Trend Rider: Pump erkannt, aber Entry erst beim bestätigten Reclaim nach Pullback
        out['high_3_prev'] = out['high'].rolling(3).max().shift(1)
        out['vol_spike_prev'] = out['vol_ratio'].shift(1)
        out['vol_spike_prev2'] = out['vol_ratio'].shift(2)

        out['had_spike'] = (
            (out['vol_spike_prev'] > 2.0) |
            (out['vol_spike_prev2'] > 2.0)
        ).fillna(False)

        # Reclaim-Filter:
        # Nicht in rote Pullback-Kerzen kaufen.
        # Erst kaufen, wenn der Rücksetzer sichtbar gekauft wird.
        _tr_range = (out['high'] - out['low']).replace(0, np.nan)
        out['trend_rider_close_pos'] = ((out['close'] - out['low']) / _tr_range).fillna(0.5)

        out['trend_rider_reclaim'] = (
            (out['close'] > out['open']) &
            (out['close'] > out['close'].shift(1)) &
            (out['trend_rider_close_pos'] > 0.55)
        ).fillna(False)

        # Anti-Chase:
        # Kein Entry, wenn der Move schon stark gelaufen ist.
        out['trend_rider_runup_12'] = (
            out['close'] / out['close'].shift(12) - 1.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        out['trend_rider_dist_ema20'] = (
            out['close'] / out['ema_20'].replace(0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        out['trend_rider_not_chasing'] = (
            (out['trend_rider_runup_12'] < 0.065) &
            (out['trend_rider_dist_ema20'] < 0.035) &
            (out['bb_percent'] < 0.95) &
            (out['rsi_14'] < 70.0)
        ).fillna(False)

        # TrendRider Teilbedingungen — repariert 25.06.2026:
        # Nicht mehr ALLES hart gleichzeitig verlangen.
        # CVD und Volumen-Pullback sind jetzt alternative Flow-Bestaetigungen.
        out['trend_rider_vol_pullback'] = (
            (out['vol_ratio'] < out['vol_spike_prev'] * 0.70) &
            (out['vol_ratio'] > 0.50)
        ).fillna(False)

        out['trend_rider_price_pullback'] = (
            (out['close'] < out['high_3_prev'] * 0.995) &
            (out['close'] > out['high_3_prev'] * 0.970)
        ).fillna(False)

        _cvd_acc = out.get('cvd_accumulation', pd.Series(False, index=out.index)).fillna(False).astype(bool)

        out['trend_rider_flow_confirm'] = (
            _cvd_acc |
            out['trend_rider_vol_pullback']
        ).fillna(False)

        out['trend_rider_core_ok'] = (
            # Spike war da — ohne Spike kein TrendRider
            out['had_spike'] &

            # VPIN erhöht, aber kein bullischer Blowoff/Überhitzungs-OFI
            out['vpin_toxic'] &
            ~out['ofi_bullish'] &

            # Rücksetzer muss wieder gekauft werden
            out['trend_rider_reclaim'] &

            # Nicht hinterherlaufen
            out['trend_rider_not_chasing'] &

            # Preis muss wirklich vom kleinen Hoch zurückgesetzt haben
            out['trend_rider_price_pullback'] &

            # Momentum intakt
            (out['dif'] > out['dea']) &

            # Trendrichtung
            (out['close'] > out['vwap']) &
            (out['close'] > out['ema_200']) &
            (out['atr_pct'] > 0.0015)
        ).fillna(False)

        out['signal_trend_rider'] = (
            out['trend_rider_core_ok'] &
            out['trend_rider_flow_confirm']
        ).fillna(False)

        # TrendRider Monitor Score — nur Diagnose, kein Entry-Einfluss
        _tr_checks = [
            out['had_spike'].astype(bool),
            out['vpin_toxic'].astype(bool),
            (~out['ofi_bullish']).astype(bool),
            out['trend_rider_reclaim'].astype(bool),
            out['trend_rider_not_chasing'].astype(bool),
            out['trend_rider_price_pullback'].astype(bool),
            (out['dif'] > out['dea']),
            (out['close'] > out['vwap']),
            (out['close'] > out['ema_200']),
            (out['atr_pct'] > 0.0015),
            out['trend_rider_flow_confirm'].astype(bool),
        ]

        out['trend_rider_monitor_score'] = sum(s.astype(int) for s in _tr_checks)

        # Hidden Divergence — Trend-Fortsetzungs-Signal
        # Preis höhere Tiefs + RSI höhere Tiefs = Trend intakt

        # Bull Flag — Promotion Layer Signal

        return out

    def _load_btc_returns(self) -> dict:
        btc_path = self.candles_dir / "BTC_USDC_5m.csv"
        if not btc_path.exists():
            return {}
        try:
            df = pd.read_csv(btc_path)
            if len(df) < 13:
                return {}
            close = df["close"]
            r1  = close.pct_change().iloc[-1]
            r3  = close.pct_change(3).iloc[-1]
            r12 = close.pct_change(12).iloc[-1]
            return {
                "btc_ret_1":  float(0.0 if r1  != r1  else r1),
                "btc_ret_3":  float(0.0 if r3  != r3  else r3),
                "btc_ret_12": float(0.0 if r12 != r12 else r12),
            }
        except Exception:
            return {}

    def process_pair(self, pair: str, btc_returns: dict = None) -> Path:
        if btc_returns is None:
            btc_returns = {}
        src = self.candles_dir / self._pair_to_filename(pair)
        dst = self.candles_dir / self._pair_to_feature_filename(pair)
        if not src.exists():
            _log(f"candle_missing pair={pair} — übersprungen")
            return None
        df = pd.read_csv(src)

        # Live-Speed-Fenster:
        # Nicht jedes Loop 1000 Kerzen komplett neu berechnen.
        # 420 x 5m = ca. 35h, genug fuer EMA200/VP200/Trend/Momentum-Kontext.
        try:
            import os as _os
            _live_rows = int(_os.environ.get("FEATURE_LIVE_ROWS", "420"))
        except Exception:
            _live_rows = 420

        if len(df) > _live_rows:
            df = df.tail(_live_rows).copy().reset_index(drop=True)

        self._current_pair = pair
        features = self.build_features(df)

        # Volume Profile — rolling pro Kerze, nicht ein aktueller Wert auf alle historischen Kerzen
        try:
            features = _vp_add_features(features, lookback=200)
        except Exception as _e:
            _log(f"vp_rolling_failed pair={pair} reason={_e}")
            features["vp_vpoc"] = features["vp_vah"] = features["vp_val"] = 0.0
            features["vp_in_va"] = True
            features["vp_above_vah"] = features["vp_below_val"] = False
            features["vp_poc_dist"] = 0.0

        if btc_returns and pair != "BTC/USDC":
            btc_r3  = btc_returns.get("btc_ret_3",  0.0)
            btc_r12 = btc_returns.get("btc_ret_12", 0.0)
            features["rel_strength_3"]  = features["ret_3"]  - btc_r3
            features["rel_strength_12"] = features["ret_12"] - btc_r12
            features["stronger_than_btc"] = (
                (features["rel_strength_3"]  > 0.002) |
                (features["rel_strength_12"] > 0.005)
            )
        else:
            features["rel_strength_3"]  = 0.0
            features["rel_strength_12"] = 0.0
            features["stronger_than_btc"] = True
        # --- signal_rsi_cross: RSI6/14-Kreuzung, 3-Kerzen-Sustain, rsi6<=50-Deckel ---
        # Backtest 02.07.2026 (n=132, bereinigt, 38 Pairs): Netto +0.3247%, WR 52.3%, Median +0.11%
        _rsi6_cross_up = (features["rsi_6"] > features["rsi_14"]) & (features["rsi_6"].shift(1) <= features["rsi_14"].shift(1))
        _rsi6_above = features["rsi_6"] > features["rsi_14"]
        _rsi6_sustain3 = _rsi6_cross_up.shift(2).fillna(False) & _rsi6_above.shift(1).fillna(False) & _rsi6_above
        # ATR-DECKEL ERGAENZT 27.07.2026: rsi_cross nutzt einen FIXEN -0.70%-Stop
        # (07./08.07. gegen ATR-Varianten getestet und gewonnen - Stop selbst bleibt
        # unangetastet). In volatilen Pairs liegt dieser fixe Stop aber INNERHALB der
        # normalen Kerzenschwankung und wird von Rauschen getroffen statt von einer
        # echten Gegenbewegung. BEWEIS ueber DISJUNKTE Buckets (n=102 echte Trades
        # seit Reset 12.07., Stop-Distanz in ATR-Einheiten gerechnet):
        #   <1.0 ATR: VerlustStop 62.5%, WR 25.0%, Netto -0.3329%  (n=8)
        #   1.0-2.0 : VerlustStop 31.2%, WR 31.2%, Netto -0.1739%  (n=16)
        #   2.0-4.0 : VerlustStop 23.1%, WR 38.5%, Netto -0.1286%  (n=52)
        #   >4.0 ATR: VerlustStop  7.7%, WR 46.2%, Netto -0.0362%  (n=26)
        # Alle vier Kennzahlen monoton ueber sich AUSSCHLIESSENDE Gruppen - anders als
        # der Liquiditaets-Sweep vom 22.07., dessen "Treppe" aus verschachtelten Mengen
        # bestand und live scheiterte (siehe update_watchlist.py).
        # SCHWELLE 0.0030 aus der Mechanik abgeleitet, NICHT aus dem Sweep-Bestwert:
        # 0.70% / 0.30% = 2.33 ATR Mindestabstand, sitzt knapp in der besseren Zone.
        # Bewusst nicht 0.0025 (im verschachtelten Sweep zwar Peak mit +0.014% Netto,
        # aber beide Nachbarwerte schlechter = Ausschlag, kein Plateau).
        # Schneidet weg: ATR>0.0030, n=40, WR 27.5%, Netto -0.2631%, Dollar -0.782.
        features["signal_rsi_cross"] = (
            _rsi6_sustain3
            & (features["rsi_6"] <= 50)
            & (features["atr_pct"] <= 0.0030)
        )

        features.to_csv(dst, index=False)
        last = features.iloc[-2]  # letzte abgeschlossene Kerze
        rs = float(last.get("rel_strength_3", 0.0) or 0.0)
        stronger = bool(last.get("stronger_than_btc", True))
        _log(
            f"features pair={pair} rows={len(features)} "
            f"trend={bool(last.get('signal_trend', False))} "
            f"breakout={bool(last.get('signal_breakout', False))} "
            f"pullback={bool(last.get('signal_pullback', False))} "
            f"range={bool(last.get('signal_range', False))} "
            f"momentum={bool(last.get('signal_momentum', False))} "
            f"reversal={bool(last.get('signal_reversal', False))} "
            f"rs={rs:.4f} stronger={stronger}"
        )
        return dst

    def process_many(self, pairs: list[str]) -> list[Path]:
        out = []

        # BTC zuerst berechnen, damit rel_strength fuer alle anderen Paare frische BTC-Daten nutzt.
        # Danach BTC im normalen Pair-Loop ueberspringen, sonst wird BTC pro Runde doppelt berechnet.
        if "BTC/USDC" in pairs:
            try:
                result = self.process_pair("BTC/USDC")
                if result is not None:
                    out.append(result)
            except Exception as _e:
                _log(f"process_pair_error pair=BTC/USDC reason={_e}")

        btc_returns = self._load_btc_returns()

        for pair in pairs:
            if pair == "BTC/USDC":
                continue

            try:
                result = self.process_pair(pair, btc_returns=btc_returns)
                if result is not None:
                    out.append(result)
            except Exception as _e:
                _log(f"process_pair_error pair={pair} reason={_e}")
        return out
