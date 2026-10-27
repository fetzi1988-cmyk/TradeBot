from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


def _log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}")


class RegimeManager:
    def __init__(self, candles_dir: str = "data/candles", state_dir: str = "data/state"):
        self.candles_dir = Path(candles_dir)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _pair_to_feature_filename(self, pair: str) -> str:
        return pair.replace("/", "_") + "_5m_features.csv"

    def load_features(self, pair: str) -> pd.DataFrame:
        path = self.candles_dir / self._pair_to_feature_filename(pair)
        if not path.exists():
            raise FileNotFoundError(f"Feature-Datei fehlt: {path}")
        return pd.read_csv(path)

    def classify_pair(self, pair: str) -> dict:
        df = self.load_features(pair)
        last = df.iloc[-1]

        trend = bool(last.get("signal_trend", False))
        breakout = bool(last.get("signal_breakout", False))
        pullback = bool(last.get("signal_pullback", False))
        range_sig = bool(last.get("signal_range", False))
        momentum = bool(last.get("signal_momentum", False))
        reversal = bool(last.get("signal_reversal", False))

        if breakout:
            regime = "breakout"
        elif momentum:
            regime = "momentum"
        elif trend:
            regime = "trend"
        elif pullback:
            regime = "pullback"
        elif reversal:
            regime = "reversal"
        elif range_sig:
            regime = "range"
        else:
            regime = "neutral"

        result = {
            "pair": pair,
            "regime": regime,
            "close": float(last.get("close", 0.0) or 0.0),
            "rsi_14": float(last.get("rsi_14", 0.0) or 0.0),
            "vol_ratio": float(last.get("vol_ratio", 0.0) or 0.0),
            "trend_up": bool(last.get("trend_up", False)),
            "range_regime": bool(last.get("range_regime", False)),
            "signal_trend": trend,
            "signal_breakout": breakout,
            "signal_pullback": pullback,
            "signal_range": range_sig,
            "signal_momentum": momentum,
            "signal_reversal": reversal,
            "stronger_than_btc": bool(last.get("stronger_than_btc", False)),
            "swept_and_reclaimed": bool(last.get("swept_and_reclaimed", False)),
            "dif": float(last.get("dif", 0.0) or 0.0),
            "rel_strength_3": float(last.get("rel_strength_3", 0.0) or 0.0),
        }

        _log(
            f"regime pair={pair} regime={regime} "
            f"rsi={result['rsi_14']:.2f} vol={result['vol_ratio']:.2f} "
            f"trend_up={result['trend_up']} range={result['range_regime']}"
        )
        return result

    def summarize_market(self, pairs: list[str]) -> dict:
        items = [self.classify_pair(pair) for pair in pairs]

        counts = {
            "trend": 0,
            "breakout": 0,
            "pullback": 0,
            "range": 0,
            "momentum": 0,
            "reversal": 0,
            "neutral": 0,
        }

        for item in items:
            counts[item["regime"]] += 1

        dominant_regime = max(counts, key=counts.get)
        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dominant_regime": dominant_regime,
            "counts": counts,
            "pairs": items,
        }

        out = self.state_dir / "market_regime.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        _log(
            f"market_regime dominant={dominant_regime} "
            f"trend={counts['trend']} breakout={counts['breakout']} "
            f"pullback={counts['pullback']} range={counts['range']} "
            f"momentum={counts['momentum']} reversal={counts['reversal']} "
            f"neutral={counts['neutral']}"
        )
        return summary
