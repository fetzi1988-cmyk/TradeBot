from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone

import yaml
import ccxt
from dotenv import load_dotenv

from app.core.broker_guard import BrokerGuard
from app.core.telegram_notifier import TelegramNotifier
from app.core.control_state import ControlState
from app.core.trade_logger import log_open
from app.core.macro_filter import should_trade, load_macro_state


def _log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}")


class ExecutionManager:
    def __init__(
        self,
        risk_path: str = "data/state/risk_approved.json",
        routing_path: str = "data/state/routing.json",
        open_trades_path: str = "data/state/open_trades.json",
        risk_cfg_path: str = "config/risk.yaml",
        settings_cfg_path: str = "config/settings.yaml",
        env_path: str = ".env",
    ):
        self.risk_path = Path(risk_path)
        self.routing_path = Path(routing_path)
        self.open_trades_path = Path(open_trades_path)
        self.risk_cfg_path = Path(risk_cfg_path)
        self.settings_cfg_path = Path(settings_cfg_path)
        self.env_path = Path(env_path)

        self.open_trades_path.parent.mkdir(parents=True, exist_ok=True)

        self.risk_cfg = self._load_yaml(self.risk_cfg_path)
        self.cfg = self._load_yaml(self.settings_cfg_path)

        load_dotenv(self.env_path)
        self.live_trading_enabled = os.getenv("LIVE_TRADING_ENABLED", "false").strip().lower() == "true"

        self.api_key = os.getenv("BINANCE_API_KEY", "").strip()
        self.api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
        self.private_api_ready = bool(self.api_key and self.api_secret)
        self.exchange = None

        if self.private_api_ready:
            self.exchange = ccxt.binance({
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
            })

        self.broker_guard = BrokerGuard()
        self.telegram = TelegramNotifier()
        self.control_state = ControlState()

    def _load_json(self, path: Path, default):
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_json(self, path: Path, payload: dict) -> None:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_yaml(self, path: Path) -> dict:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # Kollisions-Fix 01.07.2026: In-Memory-Counter damit zwei gleichzeitige
    # Trade-Eroeffnungen im selben Loop-Durchlauf nie dieselbe ID bekommen.
    # Problem: _next_trade_id() liest max(CSV+State)+1 - wenn zwei Trades
    # geoeffnet werden bevor einer davon zurueckgeschrieben wird, sehen beide
    # Aufrufe denselben Max-Wert und bekommen identische IDs. Das fuehrt dazu
    # dass log_close() den FALSCHEN CSV-Row trifft und Preis/PnL/ExitReason
    # eines anderen Pairs ueberschreibt (4 bestaetigte Faelle April 2026).
    _last_issued_id: int = 0

    def _next_trade_id(self, open_trades: list, closed_trades: list) -> int:
        ids = []
        for t in open_trades:
            ids.append(int(t.get("trade_id", 0) or 0))
        for t in closed_trades:
            ids.append(int(t.get("trade_id", 0) or 0))
        # CSV als persistente Quelle — überlebt Restarts
        try:
            from pathlib import Path
            import csv
            csv_path = Path("data/history/trades.csv")
            if csv_path.exists():
                with open(csv_path, "r") as f:
                    for row in csv.DictReader(f):
                        try:
                            ids.append(int(row.get("trade_id", 0) or 0))
                        except (ValueError, TypeError):
                            pass
        except Exception:
            pass
        candidate = (max(ids) + 1) if ids else 1
        # Sicherstellen dass wir nie dieselbe ID zweimal ausgeben,
        # auch wenn der State noch nicht zurueckgeschrieben wurde.
        next_id = max(candidate, ExecutionManager._last_issued_id + 1)
        ExecutionManager._last_issued_id = next_id
        return next_id

    def _get_free_usdc(self) -> float:
        capital_cap = float(self.risk_cfg.get("capital_cap_usdc", 50.0) or 50.0)
        mode = str(self.cfg.get("mode", "dry_run")).lower()
        if mode != "live":
            return capital_cap
        if not self.private_api_ready or self.exchange is None:
            return capital_cap
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get("USDC", {}).get("free", 0.0) or 0.0)
        except Exception as e:
            _log(f"execution balance_fetch_failed using_capital_cap error={e}")
            return capital_cap

    def _compute_final_stake(self, proposed_stake: float, free_usdc: float) -> tuple[float, str | None]:
        capital_cap = float(self.risk_cfg.get("capital_cap_usdc", 50.0) or 50.0)
        capital_reserve = float(self.risk_cfg.get("capital_reserve_usdc", 10.0) or 10.0)
        min_order = float(self.risk_cfg.get("min_order_usdc", 6.0) or 6.0)
        max_stake_pct = float(self.risk_cfg.get("max_stake_pct_of_free_usdc", 0.40) or 0.40)

        capped_free = min(free_usdc, capital_cap)
        usable_after_reserve = max(0.0, capped_free - capital_reserve)
        stake_cap_by_free = free_usdc * max_stake_pct
        final_stake = min(proposed_stake, usable_after_reserve, stake_cap_by_free)

        final_stake = round(final_stake, 2)

        if usable_after_reserve < min_order:
            return 0.0, "reserve_blocks_trade"

        if final_stake < min_order:
            return 0.0, "stake_below_min_order"

        return final_stake, None

    def _confirm_momentum_1m_entry(self, pair: str, setup: dict | None = None) -> tuple[bool, str]:
        """1m Entry-Timing fuer Momentum.
        5m liefert nur das Setup. Entry nur, wenn 1m Reclaim sauber ist.
        """
        try:
            import ccxt
            import pandas as pd
            import numpy as np

            ex = ccxt.binance({"enableRateLimit": True})
            rows = ex.fetch_ohlcv(pair, timeframe="1m", limit=80)

            if not rows or len(rows) < 30:
                return False, "insufficient_1m_rows"

            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df["ma7"] = df["close"].rolling(7).mean()
            df["ma25"] = df["close"].rolling(25).mean()
            df["vol_ma20"] = df["volume"].rolling(20).mean()

            # letzte abgeschlossene 1m Kerze, nicht die laufende
            last = df.iloc[-2]
            prev = df.iloc[-3]

            rng = float(last["high"] - last["low"])
            if rng <= 0:
                return False, "bad_1m_range"

            close_pos = float((last["close"] - last["low"]) / rng)
            upper_wick = float((last["high"] - last["close"]) / rng)

            runup_5 = float(last["close"] / df["close"].iloc[-7] - 1.0) if float(df["close"].iloc[-7]) > 0 else 0.0

            setup = setup or {}
            setup_close = float(setup.get("close", 0.0) or 0.0)
            max_failed_impulse_drop = 0.0030  # 0.30% unter 5m Signal-Close blocken
            setup_drop = float(last["close"] / setup_close - 1.0) if setup_close > 0 else 0.0

            checks = {
                "green": bool(last["close"] > last["open"]),
                "not_failed_5m_impulse": bool(setup_close <= 0 or last["close"] >= setup_close * (1.0 - max_failed_impulse_drop)),
                "close_gt_prev": bool(last["close"] > prev["close"]),
                "above_ma7": bool(last["close"] > last["ma7"]),
                "ma7_rising": bool(last["ma7"] > prev["ma7"]),
                "above_ma25_zone": bool(last["close"] > last["ma25"] * 0.997),
                "close_pos_ok": bool(close_pos > 0.58),
                "wick_ok": bool(upper_wick < 0.40),
                "vol_ok": bool(last["volume"] > last["vol_ma20"] * 0.65),
                "not_1m_chase": bool(runup_5 < 0.035),
            }

            score = sum(1 for v in checks.values() if v)

            mandatory = (
                checks["green"] and
                checks["not_failed_5m_impulse"] and
                checks["above_ma7"] and
                checks["ma7_rising"] and
                checks["close_pos_ok"] and
                checks["wick_ok"] and
                checks["not_1m_chase"]
            )

            ok = bool(mandatory and score >= 7)

            _log(
                f"momentum_1m_confirm pair={pair} ok={ok} score={score}/{len(checks)} "
                f"close={float(last['close']):.8f} setup_close={setup_close:.8f} "
                f"setup_drop={setup_drop*100:.2f}% ma7={float(last['ma7']):.8f} "
                f"ma25={float(last['ma25']):.8f} close_pos={close_pos:.2f} "
                f"upper_wick={upper_wick:.2f} runup5={runup_5*100:.2f}% "
                f"checks={checks}"
            )

            if ok:
                return True, "confirmed"

            bad = [k for k, v in checks.items() if not v]
            return False, "1m_" + "_".join(bad[:3])

        except Exception as e:
            _log(f"momentum_1m_confirm_error pair={pair} reason={e}")
            return False, "1m_error"


    def _confirm_pullback_1m_entry(self, pair: str) -> tuple[bool, str]:
        """1m Entry-Timing fuer Pullback.
        5m liefert nur die Pullback-Zone. Entry erst, wenn 1m zeigt:
        Ruecksetzer wird wieder gekauft.
        """
        try:
            import ccxt
            import pandas as pd

            ex = ccxt.binance({"enableRateLimit": True})
            rows = ex.fetch_ohlcv(pair, timeframe="1m", limit=80)

            if not rows or len(rows) < 30:
                return False, "insufficient_1m_rows"

            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df["ma7"] = df["close"].rolling(7).mean()
            df["ma25"] = df["close"].rolling(25).mean()
            df["vol_ma20"] = df["volume"].rolling(20).mean()

            # letzte abgeschlossene 1m Kerze, nicht die laufende
            last = df.iloc[-2]
            prev = df.iloc[-3]

            rng = float(last["high"] - last["low"])
            if rng <= 0:
                return False, "bad_1m_range"

            close_pos = float((last["close"] - last["low"]) / rng)
            upper_wick = float((last["high"] - last["close"]) / rng)

            base_close = float(df["close"].iloc[-7])
            runup_5 = float(last["close"] / base_close - 1.0) if base_close > 0 else 0.0

            # Pullback ist kein Momentum-Chase:
            # Wir wollen den ersten sauberen Reclaim nach Ruecksetzer.
            checks = {
                "green": bool(last["close"] > last["open"]),
                "close_gt_prev": bool(last["close"] > prev["close"]),
                "above_ma7": bool(last["close"] > last["ma7"]),
                "ma7_not_falling": bool(last["ma7"] >= prev["ma7"] * 0.999),
                "above_ma25_zone": bool(last["close"] > last["ma25"] * 0.995),
                "close_pos_ok": bool(close_pos > 0.55),
                "wick_ok": bool(upper_wick < 0.45),
                "vol_ok": bool(last["volume"] > last["vol_ma20"] * 0.55),
                "not_1m_chase": bool(runup_5 < 0.025),
            }

            score = sum(1 for v in checks.values() if v)

            mandatory = (
                checks["green"] and
                checks["close_gt_prev"] and
                checks["above_ma7"] and
                checks["close_pos_ok"] and
                checks["wick_ok"] and
                checks["not_1m_chase"]
            )

            ok = bool(mandatory and score >= 7)

            _log(
                f"pullback_1m_confirm pair={pair} ok={ok} score={score}/9 "
                f"close={float(last['close']):.8f} ma7={float(last['ma7']):.8f} "
                f"ma25={float(last['ma25']):.8f} close_pos={close_pos:.2f} "
                f"upper_wick={upper_wick:.2f} runup5={runup_5*100:.2f}% "
                f"checks={checks}"
            )

            if ok:
                return True, "confirmed"

            bad = [k for k, v in checks.items() if not v]
            return False, "1m_" + "_".join(bad[:3])

        except Exception as e:
            _log(f"pullback_1m_confirm_error pair={pair} reason={e}")
            return False, "1m_error"

    def _confirm_rsi_cross_1m_entry(self, pair: str, setup: dict | None = None) -> tuple[bool, str]:
        """1m Entry-Timing fuer rsi_cross.
        5m liefert die RSI6/14-Kreuzung + Sustain3. Die Kreuzung allein sagt
        nichts darueber, ob die Bewegung JETZT noch traegt oder schon wieder
        dreht - genau das Muster hinter den vielen hard_stop-Treffern vom
        16.07. (6 von 12 Trades liefen direkt in -0.70%, PENDLE/ZEC/TREE/
        ZRO/XPL/UNI). Entry erst, wenn 1m zeigt: kein Fehlausbruch der schon
        am Kippen ist. Schwellen naeher an pullback (frueher Turn/Reclaim)
        als an momentum (Fortsetzung eines laufenden Moves), weil rsi_cross
        bewusst frueh einsteigen soll (rsi6<=50-Gate) - aber mit momentums
        not_failed_5m_impulse-Check uebernommen, da der exakt das beobachtete
        Versagensmuster trifft.
        """
        try:
            import ccxt
            import pandas as pd

            ex = ccxt.binance({"enableRateLimit": True})
            rows = ex.fetch_ohlcv(pair, timeframe="1m", limit=80)

            if not rows or len(rows) < 30:
                return False, "insufficient_1m_rows"

            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df["ma7"] = df["close"].rolling(7).mean()
            df["ma25"] = df["close"].rolling(25).mean()
            df["vol_ma20"] = df["volume"].rolling(20).mean()

            # letzte abgeschlossene 1m Kerze, nicht die laufende
            last = df.iloc[-2]
            prev = df.iloc[-3]

            rng = float(last["high"] - last["low"])
            if rng <= 0:
                return False, "bad_1m_range"

            close_pos = float((last["close"] - last["low"]) / rng)
            upper_wick = float((last["high"] - last["close"]) / rng)

            base_close = float(df["close"].iloc[-7])
            runup_5 = float(last["close"] / base_close - 1.0) if base_close > 0 else 0.0

            setup = setup or {}
            setup_close = float(setup.get("close", 0.0) or 0.0)
            max_failed_impulse_drop = 0.0030  # 0.30% unter 5m Signal-Close blocken
            setup_drop = float(last["close"] / setup_close - 1.0) if setup_close > 0 else 0.0

            checks = {
                "green": bool(last["close"] > last["open"]),
                "not_failed_5m_impulse": bool(setup_close <= 0 or last["close"] >= setup_close * (1.0 - max_failed_impulse_drop)),
                "close_gt_prev": bool(last["close"] > prev["close"]),
                "above_ma7": bool(last["close"] > last["ma7"]),
                "above_ma25_zone": bool(last["close"] > last["ma25"] * 0.995),
                "close_pos_ok": bool(close_pos > 0.55),
                "wick_ok": bool(upper_wick < 0.45),
                "vol_ok": bool(last["volume"] > last["vol_ma20"] * 0.55),
                "not_1m_chase": bool(runup_5 < 0.020),
            }

            score = sum(1 for v in checks.values() if v)

            mandatory = (
                checks["green"] and
                checks["not_failed_5m_impulse"] and
                checks["close_gt_prev"] and
                checks["above_ma7"] and
                checks["close_pos_ok"] and
                checks["wick_ok"] and
                checks["not_1m_chase"]
            )

            ok = bool(mandatory and score >= 7)

            _log(
                f"rsi_cross_1m_confirm pair={pair} ok={ok} score={score}/{len(checks)} "
                f"close={float(last['close']):.8f} setup_close={setup_close:.8f} "
                f"setup_drop={setup_drop*100:.2f}% ma7={float(last['ma7']):.8f} "
                f"ma25={float(last['ma25']):.8f} close_pos={close_pos:.2f} "
                f"upper_wick={upper_wick:.2f} runup5={runup_5*100:.2f}% "
                f"checks={checks}"
            )

            if ok:
                return True, "confirmed"

            bad = [k for k, v in checks.items() if not v]
            return False, "1m_" + "_".join(bad[:3])

        except Exception as e:
            _log(f"rsi_cross_1m_confirm_error pair={pair} reason={e}")
            return False, "1m_error"



    def run(self) -> dict:
        risk = self._load_json(
            self.risk_path,
            {
                "approved": [],
                "rejected": [],
                "dominant_regime": "neutral",
            },
        )

        routing = self._load_json(
            self.routing_path,
            {
                "routes": [],
                "dominant_regime": "neutral",
            },
        )

        state = self._load_json(
            self.open_trades_path,
            {
                "open_trades": [],
                "closed_trades": [],
                "closed_count_total": 0,
            },
        )

        max_open_positions = int(self.risk_cfg.get("max_open_positions", 2) or 2)
        max_positions_per_strategy = int(self.risk_cfg.get("max_positions_per_strategy", 1) or 1)

        open_trades = [t for t in state.get("open_trades", []) if t.get("status") == "open"]
        closed_trades = list(state.get("closed_trades", []))
        closed_count_total = int(state.get("closed_count_total", len(closed_trades)) or 0)

        paused = self.control_state.is_paused()

        if paused:
            out = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "live_trading_enabled": self.live_trading_enabled,
                "private_api_ready": self.private_api_ready,
                "execution_mode": "live" if self.live_trading_enabled else "paper",
                "trading_paused": True,
                "free_usdc_snapshot": round(self._get_free_usdc(), 2),
                "open_trades": open_trades,
                "created": [],
                "skipped": [],
                "open_count": len([t for t in open_trades if t.get("status") == "open"]),
                "closed_trades": closed_trades,
                "closed_now": list(state.get("closed_now", [])),
                "closed_count_total": closed_count_total,
            }
            self._save_json(self.open_trades_path, out)
            _log(
                f"execution_summary mode={out['execution_mode']} private_api_ready={out['private_api_ready']} "
                f"paused=yes free_usdc={out['free_usdc_snapshot']:.2f} created=0 skipped=0 open_count={out['open_count']}"
            )
            return out

        route_map = {r["pair"]: r for r in routing.get("routes", [])}

        created = []
        skipped = []

        next_id = self._next_trade_id(open_trades, closed_trades)

        already_open_pairs = {
            t["pair"]
            for t in open_trades
            if t.get("status") == "open"
        }

        strategy_open_counts = {}
        for t in open_trades:
            if t.get("status") != "open":
                continue
            strat = t.get("strategy", "unknown")
            strategy_open_counts[strat] = strategy_open_counts.get(strat, 0) + 1

        total_open_now = len([t for t in open_trades if t.get("status") == "open"])
        free_usdc = self._get_free_usdc()

        for item in risk.get("approved", []):
            pair = item["pair"]
            strategy = item["strategy"]

            if pair in already_open_pairs:
                skipped.append({
                    "pair": pair,
                    "strategy": strategy,
                    "reason": "already_open",
                })
                continue
            # Gestaffelter Fächer: Zweite Tranche wenn VWAP-Retest bestätigt
            _open_trade = next((t for t in open_trades if t['pair'] == pair and t.get('tranche_2_ready') and not t.get('tranche_2_filled')), None)
            if _open_trade:
                _t2_stake = round(float(_open_trade.get('stake_usdc', 10.5)) * 0.67, 2)
                _t2_final, _t2_block = self._compute_final_stake(_t2_stake, free_usdc)
                if not _t2_block:
                    _open_trade['tranche_2_filled'] = True
                    _open_trade['stake_usdc'] = round(_open_trade['stake_usdc'] + _t2_final, 2)
                    _log(f"tranche_2_executed pair={pair} stake={_t2_final:.2f}")
                continue
            # Cooldown nach hard_stop
            _cooldown_path = self.open_trades_path.parent / 'cooldown.json'
            _cooldown = {}
            if _cooldown_path.exists():
                try:
                    import json as _json
                    from datetime import datetime as _dt, timezone as _tz
                    _cooldown = _json.loads(_cooldown_path.read_text())
                    # Alle abgelaufenen Einträge bereinigen
                    _now = _dt.now(_tz.utc).isoformat()
                    _cooldown = {k: v for k, v in _cooldown.items() if v > _now}
                    _cooldown_path.write_text(_json.dumps(_cooldown))
                except Exception:
                    _cooldown = {}
            if pair in _cooldown:
                # Datencheck 01.07.2026:
                # Globaler Pair-Cooldown nach hard_stop blockt auch gute Reentries
                # anderer Strategien. Deshalb nicht mehr pauschal skippen.
                _log(f"cooldown_ignored pair={pair} strategy={strategy} reason=global_hard_stop_pair_lock_disabled")

            # Makro-Filter aktiv
            _macro = load_macro_state()
            _macro_ok, _macro_reason = should_trade(pair, strategy, _macro)
            if not _macro_ok:
                skipped.append({"pair": pair, "strategy": strategy, "reason": _macro_reason})
                continue

            # Momentum: 5m ist nur Setup. Entry erst nach 1m-Bestaetigung.
            if strategy == "momentum":
                _m1_ok, _m1_reason = self._confirm_momentum_1m_entry(pair, route_map.get(pair, {}))
                if not _m1_ok:
                    skipped.append({
                        "pair": pair,
                        "strategy": strategy,
                        "reason": "momentum_1m_not_confirmed_" + str(_m1_reason),
                    })
                    continue

            # Pullback: 5m ist nur Pullback-Zone.
            # Entry erst, wenn 1m zeigt: der Ruecksetzer wird wieder gekauft.
            if strategy == "pullback":
                _pb1_ok, _pb1_reason = self._confirm_pullback_1m_entry(pair)
                if not _pb1_ok:
                    skipped.append({
                        "pair": pair,
                        "strategy": strategy,
                        "reason": "pullback_1m_not_confirmed_" + str(_pb1_reason),
                    })
                    continue

            # rsi_cross 1m-BESTAETIGUNG DEAKTIVIERT 30.07.2026.
            # Die Methode _confirm_rsi_cross_1m_entry() bleibt im Code erhalten,
            # wird aber nicht mehr aufgerufen - Reaktivierung ist ein Einzeiler.
            # GRUND 1 (Frequenz): rsi_cross lief vom 12.-27.07. mit ~6.8 Trades/Tag
            # (n=102). Nach ATR-Deckel + Vol-Untergrenze zusaetzlich zur 1m-Pruefung
            # nur noch ~0.5/Tag - ein Rueckgang um 93%. Vier einzeln begruendete
            # Filter multiplizieren sich zu einem Sieb; keine der vier Aenderungen
            # ist dadurch noch bewertbar (30 Trades wuerden ~2 Monate brauchen).
            # GRUND 2 (Trennschaerfe): die Aufschluesselung der 270 Blockierungen
            # zeigte, dass die fuenf haeufigsten Gruende (above_ma7 84.8%,
            # close_gt_prev 78.9%, green 74.1%, close_pos_ok 70.4%, wick_ok 70.4%)
            # im Kern DIESELBE Frage messen: "ist die 1m-Kerze gerade stark gruen".
            # Der eigentlich intendierte Schutz gegen Fehlausbrueche
            # (not_failed_5m_impulse) war mit 30.7% der SELTENSTE Grund. Der Filter
            # blockierte also ueberwiegend aus einem anderen Grund als dem, fuer den
            # er gebaut wurde. Gegenprobe: blockierte Kandidaten hatten WR 27.5%
            # (n=284 simuliert) gegen ~35% der durchgelassenen - schwache Trennung.
            # Von den vier rsi_cross-Filtern ist das der am schlechtesten belegte;
            # der ATR-Deckel (disjunkte Buckets, vier monotone Kennzahlen) bleibt.
            # ERWARTUNG: Frequenz zurueck Richtung 3-5 Trades/Tag. Steigt sie nicht,
            # liegt der Engpass bei ATR-Deckel oder Vol-Untergrenze.

            if total_open_now >= max_open_positions:
                skipped.append({
                    "pair": pair,
                    "strategy": strategy,
                    "reason": "max_open_positions_reached",
                })
                continue

            current_strategy_count = strategy_open_counts.get(strategy, 0)
            if current_strategy_count >= max_positions_per_strategy:
                skipped.append({
                    "pair": pair,
                    "strategy": strategy,
                    "reason": "max_positions_per_strategy_reached",
                })
                continue

            proposed_stake = float(item.get("stake_usdc", 0.0) or 0.0)
            # Volatilitaets-adjustierte Positionsgroesse (19.06.2026): Stake invers zur
            # ATR skalieren - SYN/RE-Befund: gleicher prozentualer Stop, aber riesiger
            # Dollar-Verlust bei 50 USDC Kapital, weil Stake unabhaengig von Coin-
            # Volatilitaet immer gleich war. reference_atr=BTC/ETH-Baseline-Niveau,
            # Faktor auf 0.50-1.0 begrenzt (ruhige Coins unveraendert, max. Halbierung
            # bei extremer Volatilitaet). Binance min_notional=5.0 USDC verifiziert,
            # min_order_floor mit Sicherheitsabstand auf 6.0 gesetzt (siehe risk.yaml).
            _route_info_stake = route_map.get(pair, {})
            # Fix 20.06.2026: 'or 0.003' faengt auch ein ECHTES 0.0 ab (falsy-zero-
            # Falle), nicht nur fehlende Keys. Explizite None-Pruefung stattdessen.
            _atr_pct_raw_stake = _route_info_stake.get("atr_pct")
            _atr_pct_stake = float(_atr_pct_raw_stake) if _atr_pct_raw_stake is not None else 0.003
            _ref_atr = float(self.risk_cfg.get("atr_stake_reference", 0.0030) or 0.0030)
            _atr_factor_min = float(self.risk_cfg.get("atr_stake_factor_min", 0.50) or 0.50)
            _atr_stake_factor = max(_atr_factor_min, min(1.0, _ref_atr / _atr_pct_stake)) if _atr_pct_stake > 0 else 1.0
            _log(f"atr_stake_debug pair={pair} in_route_map={pair in route_map} atr_pct_used={_atr_pct_stake:.5f} factor={_atr_stake_factor:.3f} stake_before={proposed_stake:.2f}")
            proposed_stake = round(proposed_stake * _atr_stake_factor, 2)
            # Gestaffelter Entry: Momentum nur 60% der Stake beim ersten Entry
            if strategy == "momentum":
                proposed_stake = round(proposed_stake * 0.60, 2)
            elif strategy == "squeeze_sniper":
                proposed_stake = round(proposed_stake * 0.40, 2)
            elif strategy == "trend_rider":
                proposed_stake = round(proposed_stake * 0.70, 2)  # Fix: war 0.70x0.40=28%
            # 17.06.2026: risk_manager garantiert stake_usdc>=min_order, die Tranchen-
            # Kuerzung oben kann das wieder unterlaufen. Statt zu skippen, auf min_order
            # aufrunden (min_order<=stake_usdc gilt immer, also nie ueber volle Stake hinaus).
            _min_order_floor = float(self.risk_cfg.get("min_order_usdc", 6.0) or 6.0)
            if strategy in ("momentum", "squeeze_sniper", "trend_rider") and 0 < proposed_stake < _min_order_floor:
                proposed_stake = _min_order_floor
            final_stake, block_reason = self._compute_final_stake(proposed_stake, free_usdc)

            if block_reason:
                skipped.append({
                    "pair": pair,
                    "strategy": strategy,
                    "reason": block_reason,
                    "free_usdc": round(free_usdc, 2),
                    "proposed_stake": round(proposed_stake, 2),
                })
                continue

            route_info = route_map.get(pair, {})
            # Aktuellen Marktpreis holen statt altem Kerzen-Close
            try:
                import ccxt as _ccxt
                _exchange = _ccxt.binance({"enableRateLimit": True})
                _ticker = _exchange.fetch_ticker(pair.replace("/", "/"))
                entry_price = float(_ticker.get("last") or _ticker.get("close") or route_info.get("close", 0.0))
            except Exception as _e:
                entry_price = float(route_info.get("close", 0.0) or 0.0)
                _log(f"price_fetch_fallback pair={pair} reason={_e}")

            # Momentum Failed-Impulse Guard:
            # 5m Signal darf nicht deutlich zusammenfallen, bevor der echte Entry ausgefuehrt wird.
            # SYN/USDC 26.06.2026: Signal-Close 0.34868, Entry 0.34646 = -0.64%, danach Hard-Stop.
            if strategy == "momentum":
                _setup_close = float(route_info.get("close", 0.0) or 0.0)
                _max_failed_impulse_drop = 0.0030  # 0.30%
                if _setup_close > 0:
                    _entry_drop = float(entry_price / _setup_close - 1.0)
                    if _entry_drop < -_max_failed_impulse_drop:
                        _log(
                            f"momentum_failed_impulse_guard pair={pair} "
                            f"setup_close={_setup_close:.8f} entry={entry_price:.8f} "
                            f"drop={_entry_drop*100:.2f}%"
                        )
                        skipped.append({
                            "pair": pair,
                            "strategy": strategy,
                            "reason": f"momentum_failed_impulse_entry_drop_{_entry_drop*100:.2f}pct",
                            "setup_close": round(_setup_close, 8),
                            "entry_price": round(entry_price, 8),
                        })
                        continue

            preview = self.broker_guard.preview_order(pair, final_stake, entry_price)
            if preview.get("status") != "ok":
                skipped.append({
                    "pair": pair,
                    "strategy": strategy,
                    "reason": f"broker_guard_{preview.get('reason')}",
                    "proposed_stake": round(proposed_stake, 2),
                    "final_stake": round(final_stake, 2),
                })
                continue

            # Entry-Snapshot: letzte abgeschlossene Kerze aus Feature-CSV
            _ec = {}
            try:
                import pandas as _pd, json as _json
                from pathlib import Path as _P
                _fp = _P("data/candles") / (pair.replace("/","_") + "_5m_features.csv")
                if _fp.exists():
                    _fdf = _pd.read_csv(_fp)
                    if len(_fdf) >= 2:
                        _row = _fdf.iloc[-2]
                        _ec = {k: (v.item() if hasattr(v,"item") else v) for k,v in _row.items()}
                        _snap_dir = _P("data/history/entry_snapshots")
                        _snap_dir.mkdir(parents=True, exist_ok=True)
                        _sid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + pair.replace("/","_")
                        (_snap_dir / (_sid + ".json")).write_text(_json.dumps({k: (None if _pd.isna(v) else (bool(v) if isinstance(v,(bool,)) else v)) for k,v in _ec.items()}, default=str))
            except Exception as _e:
                _log(f"entry_snapshot_failed pair={pair} reason={_e}")
            trade = {
                "trade_id": next_id,
                "pair": pair,
                "strategy": strategy,
                "dominant_regime": item.get("dominant_regime", risk.get("dominant_regime", "neutral")),
                "stake_usdc": final_stake,
                "proposed_stake_usdc": round(proposed_stake, 2),
                "free_usdc_at_entry": round(free_usdc, 2),
                "entry_price": entry_price,
                "rounded_entry_price": preview.get("rounded_price"),
                "rounded_amount": preview.get("rounded_amount"),
                "final_notional": preview.get("final_notional"),
                "last_price": entry_price,
                "highest_price": entry_price,
                "lowest_price": entry_price,
                "entry_rsi_14": _ec.get("rsi_14", item.get("rsi_14")),
                "entry_vol_ratio": _ec.get("vol_ratio", item.get("vol_ratio")),
                "entry_vpin": _ec.get("vpin", route_info.get("vpin")),
                "entry_dif":         _ec.get("dif", 0.0),
                "entry_dea":         _ec.get("dea", 0.0),
                "entry_adx":         _ec.get("adx", 0.0),
                "entry_bb_percent":  _ec.get("bb_percent", 0.0),
                "entry_above_vwap":  _ec.get("above_vwap", False),
                "entry_bullish_wick": _ec.get("bullish_wick", False),
                "entry_ofi":         _ec.get("ofi", 0.0),
                "entry_atr_pct":     _ec.get("atr_pct", 0.0),
                "entry_adx_trend":   _ec.get("adx_trend", False),
                "entry_adx_range":   _ec.get("adx_range", False),
                "entry_l3_context":  _ec.get("l3_context", False),
                "entry_btc_corr":    route_info.get("btc_corr", 0.0),
                "entry_htf_bullish":     route_info.get("htf_bullish", True),
                "entry_htf_15m_bullish": route_info.get("htf_15m_bullish", True),
                "entry_is_decorr":       route_info.get("is_decorr", False),
                "entry_is_counter":      route_info.get("is_counter", False),
                "entry_hmm_regime":  route_info.get("hmm_regime", ""),
                "entry_hmm_conf":    route_info.get("hmm_conf", 0.0),
                "entry_time_utc": datetime.now(timezone.utc).isoformat(),
                "status": "open",
                "side": "long",
                "execution_mode": "live" if self.live_trading_enabled else "paper",
            }

            open_trades.append(trade)
            log_open(trade)
            created.append(trade)
            already_open_pairs.add(pair)

            strategy_open_counts[strategy] = strategy_open_counts.get(strategy, 0) + 1
            total_open_now += 1
            free_usdc = max(0.0, free_usdc - final_stake)

            _log(
                f"execution mode={'live' if self.live_trading_enabled else 'paper'} "
                f"pair={pair} strategy={strategy} entry={entry_price:.6f} "
                f"stake={final_stake:.2f} proposed={proposed_stake:.2f} free_usdc={trade['free_usdc_at_entry']:.2f} status=open"
            )

            if self.telegram.enabled:
                try:
                    self.telegram.send(
                        f"🟢 OPEN {pair}\n"
                        f"Strategy: {strategy}\n"
                        f"Mode: {'LIVE' if self.live_trading_enabled else 'PAPER'}\n"
                        f"Stake: {final_stake:.2f} USDC\n"
                        f"Price: {entry_price:.6f}\n"
                        f"Amount: {preview.get('rounded_amount')}\n"
                        f"Notional: {float(preview.get('final_notional', 0.0) or 0.0):.2f}"
                    )
                except Exception as e:
                    _log(f"telegram_open_alert_failed pair={pair} error={e}")

            next_id += 1

        out = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "live_trading_enabled": self.live_trading_enabled,
            "private_api_ready": self.private_api_ready,
            "execution_mode": "live" if self.live_trading_enabled else "paper",
            "trading_paused": False,
            "free_usdc_snapshot": round(self._get_free_usdc(), 2),
            "open_trades": open_trades,
            "created": created,
            "skipped": skipped,
            "open_count": len([t for t in open_trades if t.get("status") == "open"]),
            "closed_trades": closed_trades,
            "closed_now": [],
            "closed_count_total": closed_count_total,
        }

        self._save_json(self.open_trades_path, out)

        _log(
            f"execution_summary mode={out['execution_mode']} private_api_ready={out['private_api_ready']} "
            f"free_usdc={out['free_usdc_snapshot']:.2f} created={len(created)} skipped={len(skipped)} open_count={out['open_count']}"
        )

        return out
