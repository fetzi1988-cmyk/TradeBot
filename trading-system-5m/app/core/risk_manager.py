from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import yaml


def _log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}")


class RiskManager:
    def __init__(
        self,
        routing_path: str = "data/state/routing.json",
        risk_config_path: str = "config/risk.yaml",
        output_path: str = "data/state/risk_approved.json",
    ):
        self.routing_path = Path(routing_path)
        self.risk_config_path = Path(risk_config_path)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.cfg = self._load_yaml(self.risk_config_path)

    def _load_yaml(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Risk-Konfig fehlt: {path}")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_routing(self) -> dict:
        if not self.routing_path.exists():
            raise FileNotFoundError(f"Routing-Datei fehlt: {self.routing_path}")
        return json.loads(self.routing_path.read_text(encoding="utf-8"))

    def _vol_ok(self, strategy: str, vol_ratio: float) -> bool:
        min_vol = float(self.cfg["min_vol_ratio"].get(strategy, 0.0))
        max_vol = float(self.cfg.get("max_vol_ratio", {}).get(strategy, 99.0))
        if vol_ratio < min_vol:
            return False
        if vol_ratio > max_vol:
            return False
        # Breakout: Fakeout-Zone 1.0-1.5 verboten
        if strategy == "breakout" and 1.0 <= vol_ratio <= 1.5:
            return False
        return True

    def _as_bool(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "y", "ja"}

    def _latest_feature_row(self, pair: str) -> dict:
        """
        Holt die letzte abgeschlossene Feature-Zeile fuer Zusatzpruefungen.
        Risk bekommt nicht immer alle Detailfelder aus dem Router.
        """
        try:
            import pandas as _pd
            fpath = Path("data/candles") / (pair.replace("/", "_") + "_5m_features.csv")
            if not fpath.exists():
                return {}
            df = _pd.read_csv(fpath)
            if len(df) >= 2:
                return df.iloc[-2].to_dict()
            if len(df) == 1:
                return df.iloc[-1].to_dict()
        except Exception as _e:
            _log(f"risk_feature_read_failed pair={pair} reason={_e}")
        return {}

    def _get_num(self, route: dict, feat: dict, key: str, default: float = 0.0) -> float:
        try:
            value = route.get(key, None)
            if value is None or value == "":
                value = feat.get(key, default)
            return float(value or default)
        except Exception:
            return float(default)

    def _get_bool(self, route: dict, feat: dict, key: str) -> bool:
        value = route.get(key, None)
        if value is None or value == "":
            value = feat.get(key, False)
        return self._as_bool(value)

    def _momentum_low_vol_override(self, pair: str, route: dict, vol_ratio: float, rsi: float) -> bool:
        """
        Momentum-Spezialfall:
        Ein starkes 5m-Setup mit niedrigem Volumen darf bis zur 1m-Entry-Bestaetigung weiter.
        Das ist KEIN globales Lockern des Risk-Filters.
        """
        feat = self._latest_feature_row(pair)

        score = int(self._get_num(route, feat, "momentum_entry_score", 0))
        mandatory_ok = self._get_bool(route, feat, "momentum_mandatory_ok")
        not_chasing = self._get_bool(route, feat, "momentum_not_chasing")
        reclaim_recent = self._get_bool(route, feat, "momentum_ma_reclaim_recent")
        flow_recent = self._get_bool(route, feat, "momentum_flow_recent")

        ofi = self._get_num(route, feat, "ofi_score", 0.0)
        adx = self._get_num(route, feat, "adx", 0.0)
        vp_dist = self._get_num(route, feat, "vp_poc_dist", 0.0)
        bb = self._get_num(route, feat, "bb_width", 0.0)

        flow_support = flow_recent or ofi >= 0.02

        ok = (
            vol_ratio >= 0.25 and
            score >= 9 and
            mandatory_ok and
            not_chasing and
            reclaim_recent and
            flow_support and
            adx >= 20.0 and
            38.0 <= rsi <= 68.0 and
            abs(vp_dist) <= 3.0 and
            bb <= 0.08
        )

        if ok:
            _log(
                f"risk_vol_override pair={pair} strategy=momentum "
                f"reason=strong_setup_to_1m_confirm score={score} vol={vol_ratio:.2f} "
                f"rsi={rsi:.2f} adx={adx:.1f} vp={vp_dist:+.2f}% "
                f"ofi={ofi:+.3f} flow_recent={flow_recent}"
            )

        return ok

    def _momentum_reentry_near_prev_high_block(self, pair: str, route: dict) -> tuple[bool, str]:
        """
        Momentum-Re-Entry-Schutz:
        Datencheck 29.06.2026:
        Momentum->Momentum nach hard_stop_price war insgesamt positiv,
        aber Sofort-Re-Entry knapp unter/am vorherigen Highest war negativ.

        Block nur wenn:
        - gleicher Pair
        - vorheriger Trade momentum + hard_stop_price
        - letzter Exit <= 1 Minute her
        - aktueller Route-Close liegt zwischen -0.10% und 0.00% zum vorherigen Highest

        Kein normaler Cooldown. Neues Hoch bleibt erlaubt.
        """
        try:
            import pandas as _pd

            path = Path("data/history/trades.csv")
            if not path.exists():
                return False, "-"

            close_now = float(route.get("close", 0.0) or 0.0)
            if close_now <= 0:
                return False, "-"

            t = _pd.read_csv(path)

            needed = {"pair", "strategy", "exit_reason", "close_time_utc", "highest_price_at_exit"}
            if not needed.issubset(set(t.columns)):
                return False, "-"

            t["close_time_utc"] = _pd.to_datetime(t["close_time_utc"], utc=True, errors="coerce")
            t["highest_price_at_exit"] = _pd.to_numeric(t["highest_price_at_exit"], errors="coerce")

            hits = t[
                (t["pair"] == pair) &
                (t["strategy"] == "momentum") &
                (t["exit_reason"] == "hard_stop_price")
            ].dropna(subset=["close_time_utc", "highest_price_at_exit"]).copy()

            if hits.empty:
                return False, "-"

            last = hits.sort_values("close_time_utc").iloc[-1]

            gap_min = (_pd.Timestamp.now(tz="UTC") - last["close_time_utc"]).total_seconds() / 60.0
            prev_high = float(last["highest_price_at_exit"] or 0.0)

            if prev_high <= 0:
                return False, "-"

            dist_pct = (close_now / prev_high - 1.0) * 100.0

            blocked = (
                gap_min <= 1.0 and
                dist_pct >= -0.10 and
                dist_pct <= 0.0
            )

            if blocked:
                _log(
                    f"risk_momentum_reentry_block pair={pair} "
                    f"gap_min={gap_min:.3f} close={close_now:.8f} "
                    f"prev_high={prev_high:.8f} dist_pct={dist_pct:+.4f}% "
                    f"reason=momentum_reentry_near_prev_high_blocked"
                )
                return True, "momentum_reentry_near_prev_high_blocked"

            return False, "-"

        except Exception as _e:
            _log(f"risk_momentum_reentry_check_failed pair={pair} reason={_e}")
            return False, "-"


    def _trend_rider_quality_gate(self, pair: str, route: dict, rsi: float, vol_ratio: float) -> tuple[bool, str]:
        """
        Trend_Rider braucht Spike -> Pullback -> Reclaim -> Flow -> Anti-Chase.
        Details kommen aus der letzten Feature-Zeile.
        """
        feat = self._latest_feature_row(pair)

        score = int(self._get_num(route, feat, "trend_rider_monitor_score", 0))
        adx = self._get_num(route, feat, "adx", 0.0)
        vp_dist = self._get_num(route, feat, "vp_poc_dist", 999.0)
        atr_pct = self._get_num(route, feat, "atr_pct", 0.0)

        reclaim = self._get_bool(route, feat, "trend_rider_reclaim")
        not_chasing = self._get_bool(route, feat, "trend_rider_not_chasing")
        price_pullback = self._get_bool(route, feat, "trend_rider_price_pullback")
        vol_pullback = self._get_bool(route, feat, "trend_rider_vol_pullback")
        flow_confirm = self._get_bool(route, feat, "trend_rider_flow_confirm")
        core_ok = self._get_bool(route, feat, "trend_rider_core_ok")
        above_vwap = self._get_bool(route, feat, "above_vwap")
        cvd_acc = self._get_bool(route, feat, "cvd_accumulation")

        vol_support = bool(vol_ratio >= 0.20 or vol_pullback or cvd_acc)

        checks = {
            "score": score >= 10,
            "core": core_ok,
            "flow": flow_confirm,
            "reclaim": reclaim,
            "not_chasing": not_chasing,
            "pullback": price_pullback,
            "above_vwap": above_vwap,
            "rsi": 40.0 <= rsi <= 68.0,
            "adx": adx >= 18.0,
            "vp": abs(vp_dist) <= 3.0,
            "atr": atr_pct >= 0.0015,
            "vol_support": vol_support,
        }

        failed = [k for k, v in checks.items() if not v]
        if failed:
            return False, "trend_rider_quality_" + "+".join(failed[:4])

        _log(
            f"risk_trend_rider_quality pair={pair} status=ok "
            f"score={score} rsi={rsi:.2f} vol={vol_ratio:.2f} "
            f"adx={adx:.1f} vp={vp_dist:+.2f}% atr={atr_pct:.4f} "
            f"flow={flow_confirm} cvd_acc={cvd_acc}"
        )
        return True, "-"

    def _grade_signal(
        self,
        route: dict,
        strategy: str,
        dominant_regime: str,
        rsi: float,
        vol_ratio: float,
    ) -> tuple[int, str]:
        score = 0

        pair_regime = str(route.get("pair_regime", "neutral") or "neutral")
        candidates = route.get("candidates", []) or []

        min_vol = float(self.cfg["min_vol_ratio"].get(strategy, 0.0))
        lo, hi = self.cfg["rsi_bounds"].get(strategy, [0, 100])
        lo = float(lo)
        hi = float(hi)

        if pair_regime == strategy:
            score += 1

        if dominant_regime == strategy:
            score += 1

        if strategy in candidates and len(candidates) >= 2:
            score += 1

        if min_vol > 0:
            if vol_ratio >= (min_vol * 1.60):
                score += 2
            elif vol_ratio >= (min_vol * 1.20):
                score += 1

        if score >= 5:
            grade = "A"
        elif score >= 3:
            grade = "B"
        else:
            grade = "C"

        # Qualitäts-Bonus: eigene Stärke vs BTC
        stronger = bool(route.get('stronger_than_btc', False))
        if stronger:
            score += 1
        # Qualitäts-Bonus: Liquidity Sweep davor
        swept = bool(route.get('swept_and_reclaimed', False))
        if swept:
            score += 1
        # Qualitäts-Bonus: DIF nahe Nulllinie
        dif = float(route.get('dif', 999.0) or 999.0)
        if abs(dif) < 0.5:
            score += 1
        return score, grade

    def _calc_stake(self, strategy: str, dominant_regime: str, vol_ratio: float, grade: str) -> float:
        base = float(self.cfg.get("base_position_size_usdc", 11.0))
        min_order = float(self.cfg.get("min_order_usdc", 10.5))
        strat_mult = float(self.cfg["strategy_multiplier"].get(strategy, 1.0))
        regime_mult = float(self.cfg["dominant_regime_multiplier"].get(dominant_regime, 1.0))

        context_mult = max(0.75, (strat_mult * regime_mult) ** 0.5)
        vol_mult = min(max(vol_ratio, 0.80), 1.30)

        # GRADE-BONUS ENTFERNT 30.07.2026 (war A:+4.0 / B:+2.0 / C:+0.0).
        # BEFUND: der Grade hat keine Vorhersagekraft. Korrelation
        # conviction_score vs PnL = +0.027 (n=117). Grade C vs B: WR 36.9% vs
        # 40.6% - Unterschied komplett im Rauschen (Standardfehler ~8.7pp bei
        # n=32). Grade A existierte NIE (0 von 231 Freigaben), da score>=5
        # verlangt wird, das Maximum vor den Qualitaets-Boni aber genau 5 ist -
        # der A-Zweig war toter Code.
        # SCHADEN: der Bonus wurde NACH dem min_order-Floor addiert. Bei einem
        # Trade am Minimum von 6.00 bedeutete Grade B +33% Positionsgroesse,
        # allein aufgrund eines Signals ohne Informationsgehalt. Das erklaert die
        # nicht-monotone Sizing-Performance (klein 53.2% / mittel 34.4% / gross
        # 56.2% WR) - die Groesse korrelierte nicht mit Trade-Qualitaet.
        # KEIN ERSATZ NOETIG: die Normalisierung des Dollar-Risikos leistet
        # bereits die ATR-Skalierung in execution_manager.py (atr_stake_reference
        # / atr_stake_factor_min). Messung bestaetigt das: ueber die ATR-Quartile
        # steigt |PnL%| von 0.193% auf 0.432% (Faktor 2.2), die Positionsgroesse
        # sinkt gegenlaeufig von 8.02 auf 6.00 (Korrelation ATR/stake = -0.38),
        # und das |Dollar|-Risiko bleibt annaehernd flach (0.018-0.028). Der
        # Grade-Bonus arbeitete gegen diese Normalisierung.
        # Code-Inkonsistenz: der Grade wird in _grade_signal() VOR den drei
        # Qualitaets-Boni festgelegt, zurueckgegeben wird der Score MIT -
        # ein Trade konnte score=6 und Grade C haben.
        # NICHT angetastet: _grade_rank() in der Sortierung und die
        # trend_grade_c_blocked-Sperre. Beides eigene Entscheidungen.
        raw = base * context_mult * vol_mult
        stake = max(min_order, raw)

        return round(stake, 2)

    def _grade_rank(self, grade: str) -> int:
        return {"A": 3, "B": 2, "C": 1}.get(grade, 0)

    def evaluate(self) -> dict:
        routing = self._load_routing()
        dominant_regime = routing["dominant_regime"]
        routes = routing["active_routes"]

        approved = []
        rejected = []
        candidates = []

        max_open = int(self.cfg.get("max_open_positions", 2))
        max_per_strategy = int(self.cfg.get("max_positions_per_strategy", 1))

        # Phase 1: harte Filter
        for route in routes:
            pair = route["pair"]
            strategy = route["selected_strategy"]
            rsi = float(route.get("rsi_14", 0.0) or 0.0)
            vol_ratio = float(route.get("vol_ratio", 0.0) or 0.0)

            reason = None

            candle_rows = int(route.get("candle_rows", 999) or 999)
            if strategy is None:
                reason = "no_strategy_selected"
            elif candle_rows < 50:
                reason = f"too_few_candles_{candle_rows}"
            elif not self._vol_ok(strategy, vol_ratio):
                min_vol = float(self.cfg["min_vol_ratio"].get(strategy, 0.0))
                if (
                    strategy == "momentum"
                    and vol_ratio < min_vol
                    and self._momentum_low_vol_override(pair, route, vol_ratio, rsi)
                ):
                    reason = None
                else:
                    reason = "vol_too_low"

            if reason is None and strategy == "trend_rider":
                _tr_ok, _tr_reason = self._trend_rider_quality_gate(pair, route, rsi, vol_ratio)
                if not _tr_ok:
                    reason = _tr_reason

            if reason is None and strategy == "momentum":
                _mr_block, _mr_reason = self._momentum_reentry_near_prev_high_block(pair, route)
                if _mr_block:
                    reason = _mr_reason

            if reason is not None:
                item = {
                    "pair": pair,
                    "strategy": strategy,
                    "dominant_regime": dominant_regime,
                    "pair_regime": route.get("pair_regime", "neutral"),
                    "rsi_14": rsi,
                    "vol_ratio": vol_ratio,
                    "status": "rejected",
                    "reason": reason,
                }
                rejected.append(item)
                _log(
                    f"risk pair={pair} strategy={strategy} status=rejected "
                    f"reason={reason} rsi={rsi:.2f} vol={vol_ratio:.2f}"
                )
                continue

            score, grade = self._grade_signal(
                route=route,
                strategy=strategy,
                dominant_regime=dominant_regime,
                rsi=rsi,
                vol_ratio=vol_ratio,
            )
            stake = self._calc_stake(strategy, dominant_regime, vol_ratio, grade)

            candidates.append({
                "pair": pair,
                "strategy": strategy,
                "dominant_regime": dominant_regime,
                "pair_regime": route.get("pair_regime", "neutral"),
                "rsi_14": rsi,
                "vol_ratio": vol_ratio,
                "conviction_score": score,
                "grade": grade,
                "stake_usdc": stake,
                "status": "candidate",
            })

        # Phase 2: beste Kandidaten zuerst
        candidates.sort(
            key=lambda x: (
                -x["conviction_score"],
                -self._grade_rank(x["grade"]),
                -x["vol_ratio"],
                x["pair"],
            )
        )

        per_strategy_count = {}

        for item in candidates:
            pair = item["pair"]
            strategy = item["strategy"]
            rsi = float(item.get("rsi_14", 0.0) or 0.0)
            vol_ratio = float(item.get("vol_ratio", 0.0) or 0.0)
            score = int(item.get("conviction_score", 0) or 0)
            grade = str(item.get("grade", "C") or "C")
            stake = float(item.get("stake_usdc", 0.0) or 0.0)

            reason = None

            if per_strategy_count.get(strategy, 0) >= max_per_strategy:
                reason = "outscored_by_better_same_strategy"
            elif len(approved) >= max_open:
                reason = "global_limit_reached"
            elif strategy == "trend" and grade == "C":
                reason = "trend_grade_c_blocked"

            if reason is None:
                approved_item = {
                    "pair": pair,
                    "strategy": strategy,
                    "dominant_regime": dominant_regime,
                    "pair_regime": item.get("pair_regime", "neutral"),
                    "rsi_14": rsi,
                    "vol_ratio": vol_ratio,
                    "conviction_score": score,
                    "grade": grade,
                    "stake_usdc": stake,
                    "status": "approved",
                }
                approved.append(approved_item)
                per_strategy_count[strategy] = per_strategy_count.get(strategy, 0) + 1

                _log(
                    f"risk pair={pair} strategy={strategy} status=approved "
                    f"grade={grade} score={score} stake={stake} rsi={rsi:.2f} vol={vol_ratio:.2f}"
                )
            else:
                rejected_item = {
                    "pair": pair,
                    "strategy": strategy,
                    "dominant_regime": dominant_regime,
                    "pair_regime": item.get("pair_regime", "neutral"),
                    "rsi_14": rsi,
                    "vol_ratio": vol_ratio,
                    "conviction_score": score,
                    "grade": grade,
                    "stake_usdc": stake,
                    "status": "rejected",
                    "reason": reason,
                }
                rejected.append(rejected_item)

                _log(
                    f"risk pair={pair} strategy={strategy} status=rejected "
                    f"reason={reason} grade={grade} score={score} rsi={rsi:.2f} vol={vol_ratio:.2f}"
                )

        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dominant_regime": dominant_regime,
            "approved": approved,
            "rejected": rejected,
            "approved_count": len(approved),
            "rejected_count": len(rejected),
        }

        self.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        _log(
            f"risk_summary dominant={dominant_regime} "
            f"approved={len(approved)} rejected={len(rejected)}"
        )
        return summary
