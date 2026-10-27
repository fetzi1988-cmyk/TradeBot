from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.core.collector import MarketCollector
from app.core.feature_engine import FeatureEngine
from app.core.regime_manager import RegimeManager
from app.core.signal_router import SignalRouter
from app.core.risk_manager import RiskManager
from app.core.execution_manager import ExecutionManager
from app.core.trade_monitor import TradeMonitor
from app.core.exit_manager import ExitManager
from app.core.telegram_notifier import TelegramNotifier
from app.core.htf_feature_engine import HTFFeatureEngine
from app.core.hmm_regime import train as hmm_train, predict as hmm_predict
from app.core import cvd_collector as _cvd_collector


def _log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}")


class TradingSystemRunner:
    def __init__(self):
        self.base_dir = Path("/root/trading-system")
        self.config_dir = self.base_dir / "config"

        self.settings = self._load_yaml(self.config_dir / "settings.yaml")
        self.pairs = self._load_pairs(self.config_dir / "pairs.yaml")

        self.mode = self.settings.get("mode", "dry_run")
        self.exchange = self.settings.get("exchange", "binance")
        self.timeframe = self.settings.get("timeframe", "5m")
        self.quote_currency = self.settings.get("quote_currency", "USDC")
        self.max_open_trades = int(self.settings.get("max_open_trades", 6))
        self.loop_interval_seconds = int(self.settings.get("loop_interval_seconds", 15))

        self.enabled_strategies = list(
            self.settings.get(
                "enabled_strategies",
                ["trend", "breakout", "pullback", "range", "momentum", "reversal"],
            )
        )

        self.collector = MarketCollector({"settings": self.settings})
        self.feature_engine = FeatureEngine()
        self.regime_manager = RegimeManager()
        _cvd_collector.start()  # WebSocket CVD
        self.signal_router = SignalRouter()
        self.risk_manager = RiskManager()
        self.execution_manager = ExecutionManager()
        self.trade_monitor = TradeMonitor()
        self.exit_manager = ExitManager()
        self.htf_engine = HTFFeatureEngine()
        self.telegram = TelegramNotifier()

    def _load_yaml(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Config fehlt: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Ungültige YAML-Struktur in {path}")
        return data

    def _load_pairs(self, path: Path) -> list[str]:
        if not path.exists():
            raise FileNotFoundError(f"Pairs-Datei fehlt: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []

        disabled_pairs = {"FTM/USDC"}

        if isinstance(data, list):
            pairs = [str(x) for x in data]
            return [x for x in pairs if x not in disabled_pairs]

        if isinstance(data, dict):
            if "pairs" in data and isinstance(data["pairs"], list):
                pairs = [str(x) for x in data["pairs"]]
                return [x for x in pairs if x not in disabled_pairs]

        raise ValueError(f"Ungültige pairs.yaml Struktur in {path}")

    def _pin_open_position_pairs(self, pairs: list[str]) -> list[str]:
        """Fix 24.06.2026: Pairs mit offener Position bleiben im Collector,
        unabhaengig von Volumen-Rotation - sonst friert die Position ein
        (WLD/USDC-Fund: no_feature_data, Exit-Check komplett uebersprungen)."""
        try:
            import json as _json
            path = self.base_dir / "data" / "state" / "open_trades.json"
            if not path.exists():
                return pairs
            state = _json.loads(path.read_text(encoding="utf-8")) or {}
            open_pairs = {t.get("pair") for t in state.get("open_trades", [])
                          if t.get("status") == "open" and t.get("pair")}
            added = open_pairs - set(pairs)
            if added:
                _log(f"pairs_pinned_for_open_positions added={sorted(added)}")
            return sorted(set(pairs) | open_pairs)
        except Exception as _e:
            _log(f"pairs_pin_failed reason={_e}")
            return pairs

    def _data_health_check(self) -> dict:
        """Health-Check 25.06.2026: pro Pair pruefen ob Feature-Daten plausibel
        und aktuell sind, bevor geroutet/gehandelt wird. VORERST NUR Sichtbarkeit
        (Logging), kein Blocken - Beobachtungsphase wie bei vp_gate/atr_stake_debug."""
        import pandas as _pd
        ok_count = 0
        blocked = []
        for pair in self.pairs:
            reasons = []
            fpath = self.base_dir / "data" / "candles" / (pair.replace("/", "_") + "_5m_features.csv")
            if not fpath.exists():
                blocked.append((pair, "datei_fehlt")); continue
            try:
                df = _pd.read_csv(fpath)
            except Exception:
                blocked.append((pair, "lesefehler")); continue
            if len(df) == 0:
                blocked.append((pair, "leer")); continue
            if len(df) < 200:
                reasons.append("zu_wenig_kerzen")
            last = df.iloc[-1]
            try:
                ts_col = df.columns[0]
                raw_ts = last[ts_col]
                last_ts = _pd.to_datetime(raw_ts, utc=True, unit="ms", errors="coerce")
                if last_ts is not None and not _pd.isna(last_ts):
                    age_min = (_pd.Timestamp.now(tz="UTC") - last_ts).total_seconds() / 60
                    if age_min > 20:
                        reasons.append(f"veraltet_{age_min:.0f}min")
            except Exception:
                pass
            close = float(last.get("close", 0) or 0)
            if close <= 0:
                reasons.append("close_ungueltig")
            atr_pct = float(last.get("atr_pct", 0) or 0)
            if atr_pct <= 0:
                reasons.append("atr_pct_ungueltig")
            for col in ("dea", "dif", "bb_percent"):
                if col in df.columns and _pd.isna(last.get(col)):
                    reasons.append(f"{col}_nan")
            rsi = last.get("rsi_14", None)
            if rsi is not None and not _pd.isna(rsi) and not (0 <= float(rsi) <= 100):
                reasons.append("rsi_14_unplausibel")
            vol_ratio = last.get("vol_ratio", None)
            if vol_ratio is not None and not _pd.isna(vol_ratio) and float(vol_ratio) < 0:
                reasons.append("vol_ratio_negativ")
            if reasons:
                blocked.append((pair, ",".join(reasons)))
            else:
                ok_count += 1
        _log(f"data_health pairs={len(self.pairs)} ok={ok_count} blocked={len(blocked)}")
        for pair, reason in blocked:
            _log(f"data_health_issue pair={pair} reasons={reason}")
        return {"ok": ok_count, "blocked": blocked}

    def log_boot(self) -> None:
        _log("BOOT trading-system")
        self.telegram.send("🟢 SYSTEM BOOT\nMode: " + self.mode + "\nPairs: " + str(len(self.pairs)) + "\nStrategien: " + ", ".join(self.enabled_strategies))
        _log(f"mode={self.mode} | exchange={self.exchange} | timeframe={self.timeframe}")
        _log(f"max_open_trades={self.max_open_trades} | quote={self.quote_currency}")
        _log(f"pairs={len(self.pairs)} | enabled_strategies={self.enabled_strategies}")

    def run_once(self) -> None:
        _log("heartbeat")
        _log(f"watchlist={', '.join(self.pairs)}")
        _log(f"active_strategies={', '.join(self.enabled_strategies)}")

        # 5m Collector Gate:
        # Jedes Loop nur BTC + offene Positionen aktualisieren.
        # Die komplette Watchlist wird nur dann gesammelt, wenn eine neue 5m-Kerze da ist.
        _full_collect_due = True

        try:
            import json as _json
            import pandas as _pd

            if not hasattr(self, "_last_collect_5m_ts"):
                self._last_collect_5m_ts = ""

            _open_pairs = []
            _state_path = self.base_dir / "data" / "state" / "open_trades.json"
            if _state_path.exists():
                try:
                    _state = _json.loads(_state_path.read_text(encoding="utf-8")) or {}
                    _open_pairs = [
                        t.get("pair") for t in _state.get("open_trades", [])
                        if t.get("pair") and t.get("status", "open") == "open"
                    ]
                except Exception:
                    _open_pairs = []

            _fast_pairs = []
            for _p in ["BTC/USDC"] + _open_pairs:
                if _p and _p not in _fast_pairs:
                    _fast_pairs.append(_p)

            # Nach dem ersten Voll-Collect reicht BTC als Taktgeber.
            if self._last_collect_5m_ts:
                self.collector.collect_many(_fast_pairs, limit=500)

                _btc_path = self.base_dir / "data" / "candles" / "BTC_USDC_5m.csv"
                _btc_df = _pd.read_csv(_btc_path)
                _latest_ts = str(_btc_df.iloc[-1].get("timestamp", "")) if len(_btc_df) else ""

                if _latest_ts == self._last_collect_5m_ts:
                    _full_collect_due = False
                else:
                    self._last_collect_5m_ts = _latest_ts

            if _full_collect_due:
                # Beim ersten Lauf oder bei neuer BTC-5m-Kerze: alle Paare aktualisieren.
                # Falls BTC/open_pairs direkt davor schon aktualisiert wurden, diese hier nicht doppelt sammeln.
                if self._last_collect_5m_ts:
                    _remaining_pairs = [x for x in self.pairs if x not in _fast_pairs]
                    if _remaining_pairs:
                        self.collector.collect_many(_remaining_pairs, limit=1000)
                else:
                    self.collector.collect_many(self.pairs, limit=1000)

                try:
                    _btc_path = self.base_dir / "data" / "candles" / "BTC_USDC_5m.csv"
                    _btc_df = _pd.read_csv(_btc_path)
                    self._last_collect_5m_ts = str(_btc_df.iloc[-1].get("timestamp", "")) if len(_btc_df) else self._last_collect_5m_ts
                except Exception:
                    pass

                _log(f"collector_full reason=new_or_boot pairs={len(self.pairs)} ts={self._last_collect_5m_ts}")
            else:
                _log(f"collector_skipped reason=same_5m_candle fast_pairs={len(_fast_pairs)} ts={self._last_collect_5m_ts}")

        except Exception as _e:
            _log(f"collector_gate_failed reason={_e}")
            self.collector.collect_many(self.pairs, limit=1000)

        # HTF-Daten 15m/1h: nur beim Boot und danach gedrosselt.
        # Vorher lief collect_htf_many + htf_engine.process_many jedes Loop und process_many sogar doppelt.
        # Das machte den 15s Loop extrem langsam.
        if not hasattr(self, "_htf_counter"):
            self._htf_counter = 999999  # erzwingt HTF Refresh direkt beim ersten Loop
        if not hasattr(self, "htf_map"):
            self.htf_map = {}

        self._htf_counter += 1
        _htf_refresh_every = max(1, int((15 * 60) / max(1, self.loop_interval_seconds)))  # ca. 15 Minuten

        if self._htf_counter >= _htf_refresh_every:
            # Pairs dynamisch neu laden — kein Neustart nötig
            try:
                new_pairs = self._pin_open_position_pairs(self._load_pairs(self.config_dir / "pairs.yaml"))
                if new_pairs != self.pairs:
                    _log(f"pairs_reloaded old={len(self.pairs)} new={len(new_pairs)}")
                    self.pairs = new_pairs
            except Exception as _e:
                _log(f"pairs_reload_failed reason={_e}")

            try:
                self.collector.collect_htf_many(self.pairs)
                self.htf_map = self.htf_engine.process_many(self.pairs)
                self._htf_counter = 0
                _log(f"htf_data_refreshed every_loops={_htf_refresh_every} pairs={len(self.pairs)}")
            except Exception as _e:
                _log(f"htf_refresh_failed reason={_e}")

            # HMM: Neu trainieren + Regime Update, maximal sinnvoll bei HTF Refresh.
            try:
                import json as _json
                from pathlib import Path as _Path
                from app.core.hmm_regime import train as hmm_train, predict as hmm_predict
                import time as _time
                _model = _Path("data/state/hmm_model.pkl")
                _age_h = (_time.time() - _model.stat().st_mtime) / 3600 if _model.exists() else 999
                if _age_h > 24:
                    hmm_train()  # Max einmal täglich neu trainieren
                _hmm = hmm_predict()
                if _hmm["valid"]:
                    _log(f"hmm_regime={_hmm['regime']} confidence={_hmm['confidence']:.2%}")
                    _Path("data/state/hmm_regime.json").write_text(_json.dumps(_hmm))
            except Exception as _e:
                _log(f"hmm_update_failed reason={_e}")

        # Heavy-Analyse nur dann neu berechnen, wenn sich die letzte 5m-Kerze wirklich geändert hat.
        # Trade-Monitor, Exit und Execution laufen trotzdem jedes Loop weiter.
        _analysis_due = True
        try:
            import pandas as _pd
            _btc_path = self.base_dir / "data" / "candles" / "BTC_USDC_5m.csv"
            _btc_df = _pd.read_csv(_btc_path)
            _last_5m_ts = str(_btc_df.iloc[-1].get("timestamp", "")) if len(_btc_df) else ""

            if not hasattr(self, "_last_analysis_5m_ts"):
                self._last_analysis_5m_ts = ""

            if self._last_analysis_5m_ts == _last_5m_ts:
                _analysis_due = False
            else:
                self._last_analysis_5m_ts = _last_5m_ts

        except Exception as _e:
            _log(f"analysis_gate_failed reason={_e}")
            _analysis_due = True

        if _analysis_due:
            _log("analysis_due reason=new_5m_candle")
            self.feature_engine.process_many(self.pairs)
            _health = self._data_health_check()
            _blocked_pairs = {p for p, _ in _health.get("blocked", [])}
            if _blocked_pairs:
                _log(f"routing_blocked_pairs count={len(_blocked_pairs)} pairs={sorted(_blocked_pairs)}")
            self.regime_manager.summarize_market(self.pairs)
            self.signal_router.build_routes(htf_map=self.htf_map, blocked_pairs=_blocked_pairs)
            self.risk_manager.evaluate()
        else:
            _log("analysis_skipped reason=same_5m_candle")

        self.trade_monitor.run()
        self.exit_manager.run()

        self.execution_manager.run()

    def run_forever(self) -> None:
        self.log_boot()
        try:
            while True:
                self.run_once()
                time.sleep(self.loop_interval_seconds)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            self.telegram.send(
                '🔴 SYSTEM STOP\n'
                'Der Trading-Bot wurde gestoppt oder ist abgestürzt.\n'
                f'Mode: {self.mode}'
            )
