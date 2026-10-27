from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone

import ccxt
import pandas as pd
from dotenv import load_dotenv


def _log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}")


class MarketCollector:
    def __init__(self, config: dict):
        settings = config["settings"]

        self.exchange_name = settings.get("exchange", "binance")
        self.timeframe = settings.get("timeframe", "5m")
        self.candles_dir = Path("/root/trading-system/data/candles")
        self.candles_dir.mkdir(parents=True, exist_ok=True)

        load_dotenv("/root/trading-system/.env")

        api_key = os.getenv("BINANCE_API_KEY", "").strip()
        api_secret = os.getenv("BINANCE_API_SECRET", "").strip()

        self.auth_enabled = bool(api_key and api_secret)

        exchange_kwargs = {
            "enableRateLimit": True,
        }

        if self.auth_enabled and self.exchange_name.lower() == "binance":
            exchange_kwargs["apiKey"] = api_key
            exchange_kwargs["secret"] = api_secret

        exchange_class = getattr(ccxt, self.exchange_name)
        self.exchange = exchange_class(exchange_kwargs)

        mode = "keys_loaded" if self.auth_enabled else "public_only"
        _log(f"collector_ready exchange={self.exchange_name} timeframe={self.timeframe} mode={mode}")

    def _pair_to_filename(self, pair: str, timeframe: str = None) -> str:
        tf = timeframe or self.timeframe
        return f"{pair.replace('/', '_')}_{tf}.csv"

    def fetch_ohlcv_df(self, pair: str, limit: int = 1500, timeframe: str = None) -> pd.DataFrame:
        tf = timeframe or self.timeframe
        for attempt in range(3):
            try:
                rows = self.exchange.fetch_ohlcv(pair, timeframe=tf, limit=limit)
                df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                return df
            except Exception as e:
                if attempt < 2:
                    import time; time.sleep(2)
                else:
                    _log(f"fetch_ohlcv_failed pair={pair} tf={tf} reason={e}")
                    raise

    def save_pair(self, pair: str, limit: int = 1500, timeframe: str = None) -> dict:
        tf = timeframe or self.timeframe
        df = self.fetch_ohlcv_df(pair, limit=limit, timeframe=tf)
        dst = self.candles_dir / self._pair_to_filename(pair, tf)
        df.to_csv(dst, index=False)

        last = df.iloc[-1]
        out = {
            "pair": pair,
            "timeframe": tf,
            "rows": len(df),
            "close": float(last["close"]),
            "volume": float(last["volume"]),
            "datetime": str(last["datetime"]),
            "path": str(dst),
        }

        _log(
            f"collector pair={pair} tf={tf} rows={out['rows']} "
            f"close={out['close']} ts={out['datetime']}"
        )
        return out

    def save_htf(self, pair: str) -> None:
        """Lädt 15m und 1h Candles für HTF Analyse"""
        try:
            self.save_pair(pair, limit=1500, timeframe="15m")
        except Exception as e:
            _log(f"htf_15m_error pair={pair} reason={e}")
        try:
            self.save_pair(pair, limit=300, timeframe="1h")
        except Exception as e:
            _log(f"htf_1h_error pair={pair} reason={e}")

    def collect_htf_many(self, pairs: list[str]) -> None:
        """HTF Daten für alle Pairs laden"""
        for pair in pairs:
            self.save_htf(pair)
        _log(f"htf_collect_done pairs={len(pairs)}")

    def collect_many(self, pairs: list[str], limit: int = 1500) -> list[dict]:
        out = []
        for pair in pairs:
            out.append(self.save_pair(pair, limit=limit))
        return out

    def collect_htf_many(self, pairs: list[str]) -> None:
        """Lädt HTF Daten für alle Pairs — läuft seltener als 5m"""
        for pair in pairs:
            self.save_htf(pair)
        _log(f"htf_collect_done pairs={len(pairs)}")
