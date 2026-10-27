from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from app.core.telegram_notifier import TelegramNotifier
from app.core.trade_logger import log_close


def _log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}")


class ExitManager:
    def __init__(
        self,
        open_trades_path: str = "data/state/open_trades.json",
        candles_dir: str = "data/candles",
        timeframe: str = "5m",
    ):
        self.open_trades_path = Path(open_trades_path)
        self.candles_dir = Path(candles_dir)
        self.timeframe = timeframe

        self.open_trades_path.parent.mkdir(parents=True, exist_ok=True)
        self.candles_dir.mkdir(parents=True, exist_ok=True)

        self.telegram = TelegramNotifier()

        # globale Sicherheitsregeln
        self.hard_stop_pct = -0.80

    def _load_json(self, path: Path, default):
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_json(self, path: Path, payload: dict) -> None:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _pair_to_feature_filename(self, pair: str) -> str:
        return f"{pair.replace('/', '_')}_{self.timeframe}_features.csv"

    def _load_last_feature_row(self, pair: str) -> dict | None:
        f = self.candles_dir / self._pair_to_feature_filename(pair)
        if not f.exists():
            return None

        df = pd.read_csv(f)
        if df.empty:
            return None

        last = df.iloc[-1].to_dict()
        return last

    def _pair_to_raw_filename(self, pair: str) -> str:
        return f"{pair.replace('/', '_')}_{self.timeframe}.csv"

    def _load_last_raw_candle_row(self, pair: str) -> dict | None:
        f = self.candles_dir / self._pair_to_raw_filename(pair)
        if not f.exists():
            return None
        df = pd.read_csv(f)
        if df.empty:
            return None
        return df.iloc[-1].to_dict()

    def _calc_hold_minutes(self, entry_time_utc: str) -> float:
        try:
            dt = datetime.fromisoformat(entry_time_utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return round((now - dt).total_seconds() / 60.0, 1)
        except Exception:
            return 0.0

    def _calc_pnl_pct(self, entry_price: float, last_price: float) -> float:
        if entry_price <= 0:
            return 0.0
        return round(((last_price / entry_price) - 1.0) * 100.0, 3)

    def _decide_exit(self, trade: dict, last: dict) -> str | None:
        strategy = str(trade.get("strategy", "") or "")
        pnl = float(trade.get("pnl_pct", 0.0) or 0.0)
        hold = float(trade.get("hold_minutes", 0.0) or 0.0)

        rsi = float(last.get("rsi_14", 0.0) or 0.0)
        trend_up = bool(last.get("trend_up", False))
        range_regime = bool(last.get("range_regime", False))
        reversal_long = bool(last.get("reversal_long", False))

        signal_trend = bool(last.get("signal_trend", False))
        signal_breakout = bool(last.get("signal_breakout", False))
        signal_pullback = bool(last.get("signal_pullback", False))
        signal_range = bool(last.get("signal_range", False))
        signal_momentum = bool(last.get("signal_momentum", False))

        # harte Sicherheitsleine
        # ATR-basierter dynamischer Stop — per-Coin automatisch über ATR
        atr_pct = float(last.get('atr_pct', 0.008) or 0.008)

        # ── ALT: Globale feste Grenzen (auskommentiert 18.06.2026) ──────────
        # Problem: feste s_max ignoriert Coin-Eigenvolatilität
        # SYN (ATR 2.1%) bekam denselben -1.00% Deckel wie BTC (ATR 0.2%)
        # stop_config = {
        #     "range":         (-0.40, -0.80),   # sehr eng
        #     "reversal":      (-0.45, -0.90),   # eng
        #     "pullback":      (-0.50, -1.00),   # mittel
        #     "momentum":      (-0.50, -1.00),   # mittel
        #     "breakout":      (-0.55, -1.20),   # etwas weiter
        #     "trend":         (-0.80, -2.00),   # braucht mehr Luft bei volatilen Alts
        #     "squeeze_sniper":(-0.40, -0.80),   # sehr eng
        #     "trend_rider":   (-0.50, -1.20),   # mittel
        # }
        # s_min, s_max = stop_config.get(strategy, (-0.50, -1.50))
        # dynamic_stop = max(s_max, min(s_min, -(atr_pct * 150)))
        # ────────────────────────────────────────────────────────────────────

        # ── NEU: Per-Coin ATR-Stop (18.06.2026) ─────────────────────────────
        # Jeder Coin berechnet Stop aus eigener Volatilität — automatisch
        # für alle Coins, auch neu aufgenommene in der dynamischen Watchlist.
        # TRX (ATR 0.06%) → -0.09% → Boden greift → -0.30%
        # BTC (ATR 0.22%) → -0.33%
        # SYN (ATR 2.13%) → -3.19%
        # RE  (ATR 4.32%) → -6.48%
        atr_mult = {
            "range":          120,
            "reversal":       130,
            "pullback":       150,
            "momentum":       150,
            "breakout":       160,
            "trend":          200,
            "squeeze_sniper": 120,
            "trend_rider":    150,
        }.get(strategy, 150)
        dynamic_stop = -(atr_pct * atr_mult)
        dynamic_stop = max(-8.0, min(-0.30, dynamic_stop))

        # Momentum ist schnell: Verlust deckeln, keine -1% bis -2% Stops mehr
        if strategy == "momentum":
            dynamic_stop = max(-0.70, dynamic_stop)

        # Pullback ist profitabel, aber historische hard_stop-Verlierer fressen Gewinn.
        # Datencheck 29.06.2026:
        # Pullback hard_stop: 36 Trades, -43.831%, Ø -1.218%.
        # Grobe Cap-Simulation -0.80%: +17.237 Prozentpunkte Verbesserung.
        # Pfad-Sample: -0.80% verbessert ebenfalls, nur ein Mini-Gewinner (+0.05%) waere betroffen.
        if strategy == "pullback":
            dynamic_stop = max(-0.80, dynamic_stop)

        # rsi_cross: fixer Stop validiert schlaegt jede ATR-Variante (Backtest
        # 07.-08.07.2026, 1.0x-3.0x getestet). Echter Override, kein Deckel wie bei
        # momentum/pullback - der ATR-Wert soll hier keine Rolle spielen.
        if strategy == "rsi_cross":
            dynamic_stop = -0.70
        # ────────────────────────────────────────────────────────────────────

        # State-basierter Hard-Stop / Gewinnschutz
        # Wichtig: break_even_set/hard_stop_price wurden bisher gesetzt, aber hier nicht global geprüft.
        _state_hard_stop = float(trade.get("hard_stop_price", 0.0) or 0.0)
        _last_close_for_stop = float(
            last.get("close", trade.get("last_price", 0.0)) or trade.get("last_price", 0.0) or 0.0
        )
        _last_low_for_stop = float(last.get("low", _last_close_for_stop) or _last_close_for_stop or 0.0)

        # Paper/Live-Logik Long-Stop:
        # Stop ist beruehrt, sobald Candle-Low <= hard_stop_price ist.
        # Der spaetere Candle-Close darf nicht als Fill-Preis genutzt werden.
        if _state_hard_stop > 0 and _last_low_for_stop > 0 and _last_low_for_stop <= _state_hard_stop:
            trade["hard_stop_trigger_low"] = _last_low_for_stop
            trade["hard_stop_trigger_close"] = _last_close_for_stop
            return "hard_stop_price"

        if pnl <= dynamic_stop:
            trade["dynamic_stop_pct"] = dynamic_stop
            trade["dynamic_stop_trigger_pnl"] = pnl
            return "hard_stop"

        # ── Promotion Layer ───────────────────────────────────────────
        # Prüft ob ein laufender Trade zum Rider promoted werden soll.
        # Bedingungen: Bull Flag aktiv + Trade läuft + noch nicht promoted
        # Wenn promoted: nur noch EMA9 Trail — kein TP, kein Timeout
        bull_flag  = bool(last.get("signal_bull_flag", False))
        ema_9_p    = float(last.get("ema_9", 0.0) or 0.0)
        close_p    = float(last.get("close", 0.0) or 0.0)
        above_vwap_p = close_p > float(last.get("vwap", 0.0) or 0.0) > 0
        below_ema9_p = ema_9_p > 0 and close_p < ema_9_p
        dif_p      = float(last.get("dif", 0.0) or 0.0)
        dea_p      = float(last.get("dea", 0.0) or 0.0)
        adx_trend_p = bool(last.get("adx_trend", False))

        # Promotion Bedingungen:
        # 1. Bull Flag Signal aktiv
        # 2. Trade mindestens +0.25% im Plus
        # 3. Mindestens 5 Min gehalten
        # 4. DIF noch über DEA (Momentum intakt)
        # 5. Preis über VWAP
        # 6. Noch nicht promoted
        # 7. NICHT rsi_cross (Ausschluss 12.07.2026, siehe Kommentar unten)
        #
        # rsi_cross AUSGESCHLOSSEN aus der Promotion-Layer, 12.07.2026:
        # gepaarter Backtest-Vergleich (gleiche Entry-Signale, n=107, davon 27
        # promoted) zeigte durchgehend schlechteres Ergebnis MIT Promotion:
        # Netto +0.89% vs +1.18% (nur Trail), Median +0.80% vs +1.35%,
        # WR 81.5% vs 92.6%. Ursache: Promotion-Stop reagiert rein auf
        # Kerzen-Low, ohne Trend-Bestaetigung - Gegenteil des validierten
        # Trail (braucht Kurs UNTER Stufe UND below_ema9 GLEICHZEITIG).
        # Kein Urteil ueber die Promotion-Layer allgemein, nur rsi_cross-
        # spezifisch deaktiviert. Details: Memory-Eintrag "rsi_cross".
        should_promote = (
            bull_flag and
            pnl >= 0.25 and
            hold >= 5 and
            dif_p > dea_p and
            above_vwap_p and
            not trade.get("promoted", False) and
            strategy != "rsi_cross"
        )

        if should_promote:
            trade["promoted"] = True
            trade["promoted_at_pnl"] = pnl
            trade["promoted_at_hold"] = hold

            # Promotion darf Gewinne nicht ungeschuetzt lassen.
            _entry_price_p = float(trade.get("entry_price", 0.0) or 0.0)
            if _entry_price_p > 0:
                _old_stop_p = float(trade.get("hard_stop_price", 0.0) or 0.0)
                _protect_mult_p = 1.003
                if pnl >= 1.00:
                    _protect_mult_p = 1.005
                if pnl >= 1.50:
                    _protect_mult_p = 1.008

                _new_stop_p = _entry_price_p * _protect_mult_p
                if _new_stop_p > _old_stop_p:
                    trade["break_even_set"] = True
                    trade["hard_stop_price"] = _new_stop_p
                    _log(
                        f"promoted_protect pair={trade.get('pair')} strategy={strategy} "
                        f"pnl={pnl:.2f}% stop={_new_stop_p:.8f}"
                    )

            _log(f"PROMOTED pair={trade.get('pair')} strategy={strategy} pnl={pnl:.2f}% hold={hold:.1f}min")

        # Wenn promoted — Gewinnschutz zuerst nachziehen, dann EMA9/VWAP Regeln.
        if trade.get("promoted", False):
            _entry_price_p = float(trade.get("entry_price", 0.0) or 0.0)
            if _entry_price_p > 0:
                _old_stop_p = float(trade.get("hard_stop_price", 0.0) or 0.0)

                _protect_mult_p = 0.0
                if pnl >= 0.50:
                    _protect_mult_p = 1.001
                if pnl >= 1.00:
                    _protect_mult_p = 1.005
                if pnl >= 1.50:
                    _protect_mult_p = 1.008
                if pnl >= 2.00:
                    _protect_mult_p = 1.012

                if _protect_mult_p > 0:
                    _new_stop_p = _entry_price_p * _protect_mult_p
                    if _new_stop_p > _old_stop_p:
                        trade["break_even_set"] = True
                        trade["hard_stop_price"] = _new_stop_p
                        _log(
                            f"promoted_trail_protect pair={trade.get('pair')} strategy={strategy} "
                            f"pnl={pnl:.2f}% stop={_new_stop_p:.8f}"
                        )

            if below_ema9_p and pnl > 0.50:
                return f"{strategy}_promoted_exit"
            # Sicherheitsnetz: wenn VWAP verloren und im Verlust
            if not above_vwap_p and pnl < 0.10 and hold >= 20:
                return f"{strategy}_promoted_vwap_lost"
            # Maximaler Sicherheits-TP
            if pnl >= 30.0:
                return f"{strategy}_promoted_max_tp"
            return None  # Laufen lassen

        if strategy == "reversal":
            atr_pct = float(last.get('atr_pct', 0.008) or 0.008)
            tp_reversal = max(0.70, min(1.80, atr_pct * 180))
            if pnl >= tp_reversal:
                return "reversal_take_profit"
            vwap = float(last.get("vwap", 0.0) or 0.0)
            close = float(last.get("close", 0.0) or 0.0)
            ema_9 = float(last.get("ema_9", 0.0) or 0.0)
            above_vwap = close > vwap > 0
            below_ema9 = ema_9 > 0 and close < ema_9
            if below_ema9 and hold >= 10 and pnl > 0.20:
                return "reversal_trail_stop_ema9"
            if below_ema9 and pnl > 0.10:
                return "reversal_rebound_exit"
            if (not above_vwap) and hold >= 20 and pnl <= -0.10:
                return "reversal_lost_vwap"
            if hold >= 25 and pnl < 0.10:
                return "reversal_timeout"
            return None

        if strategy == "trend":
            atr_pct = float(last.get('atr_pct', 0.008) or 0.008)
            vwap = float(last.get("vwap", 0.0) or 0.0)
            close = float(last.get("close", 0.0) or 0.0)
            entry_price = float(trade.get("entry_price", 0.0) or 0.0)

            above_vwap = close > vwap > 0
            below_vwap = vwap > 0 and close < vwap

            # Trend-Funktion:
            # Frueher Trendaufbau / geordnete Trendfortsetzung.
            # Deshalb kein hektischer CVD-Exit und kein frueher EMA9-Trail.
            # Validiert gegen Risk-Pass-Trend-Signale:
            # hold48_safety avg_exit_pnl +1.3088%, current +0.4096%.

            # Break-Even Schutz ab +0.50%
            if pnl >= 0.50 and not trade.get("break_even_set", False):
                trade["break_even_set"] = True
                trade["hard_stop_price"] = entry_price * 1.001

            # Starker Trend-Gewinn: mitnehmen
            if pnl >= max(5.00, atr_pct * 500):
                return "trend_take_profit"

            # Echter Schaden: Trend-Signal weg, unter VWAP, nach etwas Laufzeit klar im Minus.
            # hold ist in Minuten. 24x 5m-Kerzen = 120 Minuten.
            if (not signal_trend) and below_vwap and hold >= 120 and pnl <= -1.00:
                return "trend_safety_lost"

            # Trend-Setup bekommt Luft.
            # hold ist in Minuten. 48x 5m-Kerzen = 240 Minuten = 4 Stunden.
            # Kein kurzer Timeout, sondern getesteter Trend-Zeit-Exit.
            if hold >= 240:
                return "trend_hold48_exit"

            return None
        if strategy == "pullback":
            vwap = float(last.get("vwap", 0.0) or 0.0)
            close = float(last.get("close", 0.0) or 0.0)
            ema_9 = float(last.get("ema_9", 0.0) or 0.0)
            atr_pct = float(last.get("atr_pct", 0.006) or 0.006)
            entry_price = float(trade.get("entry_price", 0.0) or 0.0)

            above_vwap = close > vwap > 0
            below_ema9 = ema_9 > 0 and close < ema_9

            # Pullback = Bounce-Strategie.
            # Max-Hold bleibt nur Notbremse, nicht Ziel-Haltedauer.
            if entry_price > 0:
                _old_stop_pb = float(trade.get("hard_stop_price", 0.0) or 0.0)
                _protect_mult_pb = 0.0

                if pnl >= 0.30:
                    _protect_mult_pb = 1.0005
                if pnl >= 0.60:
                    _protect_mult_pb = 1.0025
                if pnl >= 1.00:
                    _protect_mult_pb = 1.0050

                if _protect_mult_pb > 0:
                    _new_stop_pb = entry_price * _protect_mult_pb
                    if _new_stop_pb > _old_stop_pb:
                        trade["break_even_set"] = True
                        trade["hard_stop_price"] = _new_stop_pb
                        _log(
                            f"pullback_protect pair={trade.get('pair')} pnl={pnl:.2f}% "
                            f"stop={_new_stop_pb:.8f}"
                        )

            # Dynamischer Bounce-TP.
            # BODEN GESENKT 0.45 -> 0.30 am 27.07.2026. Befund (n=40 seit 13.07.):
            # der 0.45-Boden band bei 70% der Trades, weil atr_pct*130 bei
            # Median-ATR 0.00261 nur 0.339% ergibt - das "dynamische" Ziel war
            # faktisch ein Festwert von 0.45. Gleichzeitig lag der Median-MFE bei
            # 0.352%, also UNTER dem Ziel: nur 15% der Trades erreichten es
            # ueberhaupt. Folge: Exit-Effizienz 14.2% (schlechteste im System),
            # PnL-Median 0.050% bei MFE-Median 0.376% - die Trades liefen bis kurz
            # unters Ziel, drehten, und wurden bei ~null beendet.
            # 0.30 ist doppelt begruendet: (a) liegt unter dem Median-MFE, damit
            # erreichbar, und laesst die ATR-Skalierung wieder greifen statt sie zu
            # ueberschreiben; (b) 0.30% brutto - 0.15% Fee = 0.15% netto, genau das
            # Doppelte der Kosten. Tiefer waere zu duenn (0.25 -> nur 0.10% netto).
            # NICHT der Sweep-Bestwert als Begruendung: 0.25/0.30/0.35 lagen alle
            # aehnlich und alle deutlich besser als 0.40/0.45 - ein Plateau, der
            # Peak darin ist Rauschen (Lehre aus dem Liquiditaets-Fehlschlag 22.07.).
            # EINSCHRAENKUNG: im Sweep waren 55% der Trades unsicher, weil bei
            # gestoppten Trades die Reihenfolge von Hoch und Tief unbekannt ist.
            # Die mechanische Begruendung gilt unabhaengig davon.
            # ERWARTUNG: Anteil pullback_take_profit steigt deutlich, Effizienz
            # steigt von 14.2%. Passiert das nicht, greift der Patch nicht.
            tp_pullback = max(0.30, min(1.10, atr_pct * 130))
            if pnl >= tp_pullback:
                return "pullback_take_profit"

            # Bounce war da, EMA9 geht verloren -> Gewinn sichern.
            if below_ema9 and hold >= 10 and pnl > 0.35:
                return "pullback_trail_stop_ema9"

            # Rücksetzer scheitert.
            if (not signal_pullback) and (not above_vwap) and hold >= 25 and pnl <= -0.15:
                return "pullback_failed"

            # Qualitätsausstieg, nicht Max-Hold.
            if hold >= 45 and pnl < 0.05:
                return "pullback_stall_timeout"

            # Harte Notbremse / Max-Hold.
            if hold >= 75 and pnl < 0.10:
                return "pullback_timeout"

            return None

        if strategy == "breakout":
            atr_pct = float(last.get('atr_pct', 0.008) or 0.008)
            vwap = float(last.get("vwap", 0.0) or 0.0)
            close = float(last.get("close", 0.0) or 0.0)
            ema_9 = float(last.get("ema_9", 0.0) or 0.0)
            above_vwap = close > vwap > 0
            below_ema9 = ema_9 > 0 and close < ema_9
            entry_price = float(trade.get("entry_price", 0.0) or 0.0)
            # Break-Even: sobald +0.50% — Stop auf Entry gesetzt
            # Kein Hard-Stop mehr möglich nach diesem Punkt
            if pnl >= 0.50 and not trade.get("break_even_set", False):
                trade["break_even_set"] = True
                trade["hard_stop_price"] = entry_price * 1.001
            # Sicherheitsnetz Max-TP
            if pnl >= max(5.00, atr_pct * 500):
                return "breakout_take_profit"
            # EMA9 Trail ab 0.30% statt 0.20%
            if below_ema9 and hold >= 8 and pnl > 0.30:
                return "breakout_trail_stop_ema9"
            # CVD Exit: Vol steigt aber Preis fällt nach Entry
            cvd_5 = float(last.get("cvd_5", 0) or 0)
            vol_ratio = float(last.get("vol_ratio", 0) or 0)
            entry_vol = float(trade.get("entry_vol_ratio", 0) or 0)
            if cvd_5 < 0 and vol_ratio > max(entry_vol * 1.3, 1.5) and hold >= 5 and pnl < 0:
                return "breakout_cvd_exit"
            if (not above_vwap) and (not signal_breakout) and hold >= 15 and pnl <= -0.20:
                return "failed_breakout"
            if hold >= 60 and pnl < 0.10:
                return "breakout_timeout"
            return None

        if strategy == "momentum":
            # Momentum braucht schnellen Gewinnschutz.
            # 5m Setup + 1m Entry kann schnelle Pushes treffen, darf sie aber nicht wieder komplett abgeben.
            entry_price = float(trade.get("entry_price", 0.0) or 0.0)
            if entry_price > 0:
                _old_stop_m = float(trade.get("hard_stop_price", 0.0) or 0.0)

                _protect_mult_m = 0.0
                if pnl >= 0.50:
                    _protect_mult_m = 1.001
                if pnl >= 1.00:
                    _protect_mult_m = 1.005
                if pnl >= 1.50:
                    _protect_mult_m = 1.008
                if pnl >= 2.00:
                    _protect_mult_m = 1.012

                if _protect_mult_m > 0:
                    _new_stop_m = entry_price * _protect_mult_m
                    if _new_stop_m > _old_stop_m:
                        trade["break_even_set"] = True
                        trade["hard_stop_price"] = _new_stop_m
                        _log(
                            f"momentum_protect pair={trade.get('pair')} pnl={pnl:.2f}% "
                            f"stop={_new_stop_m:.8f}"
                        )

            # Momentum-Follow-through-Fail: validiert 29.06.2026, n=77 echte
            # Trades (RSI>=55 + min_pnl_5_15<=-0.40). Echte Summe -27.67%
            # gegen simuliert -1.20% bei Kappung = +26.47pp Verbesserung.
            # Ehrlich: 17/77 (22%) waeren am Ende noch Gewinner geworden
            # (v.a. AAVE +3.87%), Nettoeffekt aber klar positiv.
            entry_rsi_ft = float(trade.get("entry_rsi_14", 0.0) or 0.0)
            if hold >= 5 and hold <= 15 and entry_rsi_ft >= 55 and pnl <= -0.40:
                trade["momentum_followthrough_failed_entry_rsi"] = entry_rsi_ft
                trade["momentum_followthrough_failed_hold"] = hold
                trade["momentum_followthrough_failed_pnl"] = pnl
                return "momentum_followthrough_failed"

            # Dynamischer ATR-basierter Take-Profit
            atr_pct = float(last.get('atr_pct', 0.008) or 0.008)
            tp_target = max(0.80, min(2.00, atr_pct * 200))
            if pnl >= tp_target:
                return "momentum_take_profit"
            vwap = float(last.get("vwap", 0.0) or 0.0)
            close = float(last.get("close", 0.0) or 0.0)
            ema_9 = float(last.get("ema_9", 0.0) or 0.0)
            dif = float(last.get("dif", 0.0) or 0.0)
            above_vwap = close > vwap > 0
            below_ema9 = ema_9 > 0 and close < ema_9
            # DIF-Divergenz: Preis steigt aber DIF sinkt → Momentum stirbt
            entry_dif = float(trade.get("entry_dif", dif) or dif)
            highest = float(trade.get("highest_price", close) or close)
            dif_divergence = (close >= highest * 0.998) and (dif < entry_dif * 0.85) and pnl > 0.30
            # Bärischer Docht als Exit-Signal
            bearish_wick = bool(last.get('bearish_wick', False))
            if bearish_wick and pnl > 0.20:
                return 'momentum_bearish_wick_exit'
            if dif_divergence:
                return "momentum_dif_divergence"
            if below_ema9 and hold >= 10 and pnl > 0.20:
                return "momentum_trail_stop_ema9"
            if (not above_vwap) and (not signal_momentum) and hold >= 20 and pnl <= -0.20:
                return "momentum_fade"

            # Momentum darf nicht lange seitwaerts/leicht negativ gammeln.
            # Beispiel SOL/USDC 26.06.2026:
            # hold ~58m, pnl ~-0.20%, signal_momentum=False, kaum Volumen -> kept bis Timeout.
            if (not signal_momentum) and hold >= 45 and pnl < 0.15:
                return "momentum_stall_exit"

            if hold >= 90:
                return "momentum_timeout"
            return None

        if strategy == "trend_rider":
            ema_9 = float(last.get("ema_9", 0.0) or 0.0)
            close = float(last.get("close", 0.0) or 0.0)
            vwap = float(last.get("vwap", 0.0) or 0.0)
            entry_price = float(trade.get("entry_price", 0.0) or 0.0)
            atr_pct = float(last.get("atr_pct", trade.get("entry_atr_pct", 0.008)) or 0.008)

            below_ema9 = ema_9 > 0 and close < ema_9
            below_vwap = vwap > 0 and close < vwap

            tp_rider = max(2.20, min(4.50, atr_pct * 320.0))

            if pnl >= tp_rider:
                return "trend_rider_take_profit"

            if pnl >= 0.75 and not trade.get("break_even_set", False):
                trade["break_even_set"] = True
                trade["hard_stop_price"] = entry_price * 1.0015

            if below_ema9 and pnl > 0.70:
                return "trend_rider_trail_stop"

            if below_vwap and hold >= 30 and pnl <= -0.25:
                return "trend_rider_vwap_lost"

            if hold >= 120 and pnl < 0.25:
                return "trend_rider_stall_exit"

            if hold >= 180 and pnl < 0.40:
                return "trend_rider_timeout"

            return None

        if strategy == "squeeze_sniper":
            atr_pct = float(last.get("atr_pct", 0.008) or 0.008)
            tp_sniper = max(2.00, min(8.00, atr_pct * 500))
            close = float(last.get("close", 0.0) or 0.0)
            vwap = float(last.get("vwap", 0.0) or 0.0)
            ema_9 = float(last.get("ema_9", 0.0) or 0.0)
            entry_price = float(trade.get("entry_price", 0.0) or 0.0)
            below_ema9 = ema_9 > 0 and close < ema_9
            # Break-Even bei +1%
            if pnl >= 1.0 and not trade.get("break_even_set", False):
                trade["break_even_set"] = True
                trade["hard_stop_price"] = entry_price * 1.002
            # Max-TP — institutioneller Move kann groß werden
            if pnl >= tp_sniper:
                return "sniper_take_profit"
            # EMA9 Trail
            if below_ema9 and pnl > 0.50:
                return "sniper_trail_stop"
            # VWAP verloren + Verlust nach 15min
            if close < vwap and hold >= 15 and pnl < 0.0:
                return "sniper_vwap_lost"
            # Timeout 45min
            if hold >= 45 and pnl < 0.0:
                return "sniper_timeout"
            return None
        if strategy == "range":
            vwap = float(last.get("vwap", 0.0) or 0.0)
            close = float(last.get("close", 0.0) or 0.0)
            above_vwap = close > vwap > 0
            atr_pct_r = float(last.get('atr_pct', 0.008) or 0.008)
            # Range TP kleiner — Mean-Reversion ist kurz
            tp_range = max(0.35, min(0.80, atr_pct_r * 100))
            if pnl >= tp_range:
                return "range_take_profit"
            # Früher raus wenn VWAP überschritten
            if (not above_vwap) and pnl > 0.03:
                return "range_mean_revert_exit"
            # Schneller raus wenn Signal weg
            if (not signal_range) and hold >= 15 and pnl <= -0.15:
                return "range_failed"
            # Timeout halbiert — Range ist kurz
            if hold >= 25 and pnl < 0.05:
                return "range_timeout"
            return None

        if strategy == "rsi_cross":
            # Backtest-validiert 07.-08.07.2026: fixer -0.70% Stop schlug jede
            # ATR-Variante (1.0x-3.0x) beim Netto-Mittel. Gestufter, nach oben
            # eingefrorener Trail (Memory: "Gewinner brauchen Luft zum Atmen").
            # below_ema9 statt EMA1xEMA2-Crossdown als Live-Trigger (ema_1/ema_2
            # existieren nicht live, laut Memory Trigger praktisch gleichwertig -
            # der Trail-Mechanismus ist der Hebel, nicht der Trigger).
            entry_price = float(trade.get("entry_price", 0.0) or 0.0)
            close = float(last.get("close", 0.0) or 0.0)
            low = float(last.get("low", close) or close)
            high = float(last.get("high", close) or close)
            ema_9 = float(last.get("ema_9", 0.0) or 0.0)
            below_ema9 = ema_9 > 0 and close < ema_9

            if entry_price > 0 and low > 0:
                low_pnl = ((low / entry_price) - 1.0) * 100.0
                if low_pnl <= -0.70:
                    return "rsi_cross_hard_stop"

            # Eigener High-basierter MFE-Tracker (Backtest nutzte Kerzen-High,
            # nicht Close - highest_price ist systemweit Close-basiert und wird
            # von momentum mitgenutzt, hier bewusst nicht angetastet).
            if entry_price > 0:
                high_pnl = ((high / entry_price) - 1.0) * 100.0
                mfe = max(float(trade.get("rsi_cross_mfe", 0.0) or 0.0), high_pnl)
                trade["rsi_cross_mfe"] = mfe
            else:
                mfe = 0.0

            if mfe >= 2.00:
                floor = 1.30
            elif mfe >= 1.50:
                floor = 0.90
            elif mfe >= 1.00:
                floor = 0.50
            elif mfe >= 0.60:
                floor = 0.20
            elif mfe >= 0.30:
                floor = 0.00
            else:
                floor = None

            if floor is not None and hold >= 5 and pnl <= floor and below_ema9:
                return "rsi_cross_trail_breach"

            # TRAIL-GEKOPPELTER TIMER 27.07.2026: der starre 90min-Timer schnitt
            # Trades ab, die der validierte Trail noch haette laufen lassen.
            # BEFUND (n=102 seit Reset 12.07.): 40 Trades per max_hold beendet
            # (40.7% aller Trades), davon hatten 17 einen AKTIVEN Trail-Boden und
            # ALLE 17 lagen darueber - ihr Downside war begrenzt, die Uhr hat sie
            # trotzdem gestoppt. Median verschenkt: +0.197% (MFE 0.239% vs PnL
            # 0.072%). 30% dieser Trades hatten MFE>0.7%.
            # LOGIK: floor is not None bedeutet MFE>=0.30% erreicht, Trail scharf,
            # Verlust nach unten gedeckelt -> der validierte Mechanismus darf
            # entscheiden statt eines willkuerlichen Timers. Ohne aktiven Boden
            # bleibt es bei 90min wie bisher (unveraendert, kein Grund dort).
            # 240min ist eine grobe Kappe gegen Slot-Blockade (max_open_positions
            # =4), NICHT aus Daten optimiert - die Simulation dazu scheiterte an
            # der Kerzen-Reichweite (nur 2 von 40 Trades simulierbar, 38 wegen
            # Datenluecken verworfen). Bewusst als Kappe, nicht als Feinjustierung.
            # ERWARTUNG: Anteil rsi_cross_max_hold sinkt deutlich, rsi_cross_
            # trail_breach steigt entsprechend. Passiert das nicht, greift der
            # Patch nicht wie gedacht.
            if floor is not None:
                if hold >= 240:
                    return "rsi_cross_max_hold"
            elif hold >= 90:
                return "rsi_cross_max_hold"

            return None

        if hold >= 120 and pnl < 0.10:
            return "generic_timeout"

        return None

    def run(self) -> dict:
        state = self._load_json(
            self.open_trades_path,
            {
                "open_trades": [],
                "created": [],
                "skipped": [],
                "closed_trades": [],
                "closed_now": [],
                "closed_count_total": 0,
            },
        )

        open_trades = list(state.get("open_trades", []))
        created = list(state.get("created", []))
        skipped = list(state.get("skipped", []))
        closed_total = list(state.get("closed_trades", []))
        closed_now = []

        still_open = []

        for trade in open_trades:
            if trade.get("status") != "open":
                still_open.append(trade)
                continue

            last = self._load_last_feature_row(trade["pair"])
            if last is None:
                trade["monitor_status"] = "no_feature_data"
                still_open.append(trade)
                _log(f"exit pair={trade['pair']} status=kept reason=no_feature_data")
                continue

            last_price = float(last.get("close", 0.0) or 0.0)
            entry_price = float(trade.get("entry_price", 0.0) or 0.0)

            _raw_last = self._load_last_raw_candle_row(trade["pair"])
            if _raw_last:
                for _k in ("timestamp", "datetime", "open", "high", "low", "close", "volume"):
                    if _k in _raw_last and _raw_last.get(_k) not in (None, ""):
                        last[_k] = _raw_last[_k]
                trade["price_source_at_exit"] = "raw_candle_overlay"
                last_price = float(last.get("close", last_price) or last_price or 0.0)

            trade["last_price"] = last_price
            trade["highest_price"] = max(float(trade.get("highest_price", entry_price) or entry_price), last_price)
            trade["lowest_price"] = min(float(trade.get("lowest_price", entry_price) or entry_price), last_price)
            trade["hold_minutes"] = self._calc_hold_minutes(str(trade.get("entry_time_utc", "")))
            trade["pnl_pct"] = self._calc_pnl_pct(entry_price, last_price)
            trade["last_rsi_14"] = float(last.get("rsi_14", 0.0) or 0.0)
            trade["last_vol_ratio"] = float(last.get("vol_ratio", 0.0) or 0.0)

            exit_reason = self._decide_exit(trade, last)

            if exit_reason is None:
                # Gestaffelter Fächer: VWAP-Retest für zweite Tranche
                if trade.get('strategy') == 'momentum' and not trade.get('tranche_2_filled', False):
                    _vwap = float(last.get('vwap', 0.0) or 0.0)
                    _close = float(last.get('close', 0.0) or 0.0)
                    _entry = float(trade.get('entry_price', 0.0) or 0.0)
                    # Retest: Preis kam nach Entry auf VWAP zurück und ist jetzt wieder drüber
                    _prev_below = trade.get('was_below_vwap_after_entry', False)
                    if not _prev_below and _close < _vwap and float(trade.get('hold_minutes', 0)) > 3:
                        trade['was_below_vwap_after_entry'] = True
                    if _prev_below and _close > _vwap and not trade.get('tranche_2_ready', False):
                        trade['tranche_2_ready'] = True
                        _log(f"tranche_2_ready pair={trade['pair']} pnl={trade.get('pnl_pct', 0):.3f}%")
                trade["monitor_status"] = "kept_open"
                still_open.append(trade)
                _log(
                    f"exit pair={trade['pair']} status=kept "
                    f"pnl={trade.get('pnl_pct', 0.0):.3f}% hold={trade.get('hold_minutes', 0.0):.1f}m"
                )
                continue

            # Paper-Fill-Modell:
            # Wenn hard_stop_price ausloest, wird Paper zum Stop-Preis geschlossen,
            # nicht zum spaeteren Candle-Close.
            trade["feature_close_at_exit"] = last_price
            trade["effective_exit_price"] = last_price

            if exit_reason == "hard_stop_price":
                _hstop = float(trade.get("hard_stop_price", 0.0) or 0.0)
                if _hstop > 0:
                    trade["effective_exit_price"] = _hstop
                    trade["paper_fill_model"] = "hard_stop_price_fill"
                    trade["last_price_before_effective_exit"] = last_price
                    last_price = _hstop
                    trade["last_price"] = last_price
                    trade["pnl_pct"] = self._calc_pnl_pct(entry_price, last_price)

            if exit_reason == "hard_stop":
                _dyn_stop_pct = float(trade.get("dynamic_stop_pct", 0.0) or 0.0)
                if entry_price > 0 and _dyn_stop_pct < 0:
                    _dyn_stop_price = entry_price * (1.0 + (_dyn_stop_pct / 100.0))
                    if _dyn_stop_price > 0:
                        trade["effective_exit_price"] = _dyn_stop_price
                        trade["paper_fill_model"] = "dynamic_hard_stop_fill"
                        trade["last_price_before_effective_exit"] = last_price
                        trade["dynamic_stop_price"] = _dyn_stop_price
                        last_price = _dyn_stop_price
                        trade["last_price"] = last_price
                        trade["pnl_pct"] = self._calc_pnl_pct(entry_price, last_price)

            trade["status"] = "closed"
            trade["monitor_status"] = "closed"
            trade["exit_reason"] = exit_reason
            # Logging-Erweiterung 26.06.2026: Stop/Exit-Kontext fuer spaetere
            # Paper-Fill-Modell-Auswertung sichern (kein Verhaltens-Patch).
            trade["hard_stop_price_at_exit"] = trade.get("hard_stop_price", "")
            trade["break_even_set_at_exit"] = trade.get("break_even_set", "")
            trade["highest_price_at_exit"] = trade.get("highest_price", "")
            trade["lowest_price_at_exit"] = trade.get("lowest_price", "")
            trade["exit_open"] = last.get("open", "")
            trade["exit_high"] = last.get("high", "")
            trade["exit_low"] = last.get("low", "")
            trade["exit_close"] = last.get("close", "")
            if exit_reason == 'hard_stop':
                # Datencheck 01.07.2026:
                # Kein globaler Pair-Cooldown mehr nach hard_stop.
                # Grund: globale Locks blockieren auch profitable Reentries anderer Strategien.
                _log(f"cooldown_disabled pair={trade['pair']} strategy={trade.get('strategy')} reason=global_hard_stop_pair_lock_removed")
            trade["close_time_utc"] = datetime.now(timezone.utc).isoformat()
            # Exit-Snapshot: letzte abgeschlossene Kerze beim Exit
            try:
                import pandas as _pd, json as _json
                from pathlib import Path as _P
                _fp = _P("data/candles") / (trade["pair"].replace("/","_") + "_5m_features.csv")
                if _fp.exists():
                    _fdf = _pd.read_csv(_fp)
                    if len(_fdf) >= 2:
                        _row = _fdf.iloc[-2]
                        _snap_dir = _P("data/history/exit_snapshots")
                        _snap_dir.mkdir(parents=True, exist_ok=True)
                        _sid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + trade["pair"].replace("/","_")
                        _payload = {k: (None if _pd.isna(v) else (v.item() if hasattr(v,"item") else v)) for k,v in _row.items()}
                        _payload["_trade_id"] = trade.get("trade_id")
                        _payload["_exit_reason"] = trade.get("exit_reason","")
                        _payload["_pnl_pct"] = trade.get("pnl_pct")
                        _payload["_feature_close_at_exit"] = trade.get("feature_close_at_exit", "")
                        _payload["_effective_exit_price"] = trade.get("effective_exit_price", "")
                        _payload["_hard_stop_price"] = trade.get("hard_stop_price", "")
                        _payload["_hard_stop_trigger_low"] = trade.get("hard_stop_trigger_low", "")
                        _payload["_hard_stop_trigger_close"] = trade.get("hard_stop_trigger_close", "")
                        _payload["_paper_fill_model"] = trade.get("paper_fill_model", "")
                        _payload["_price_source_at_exit"] = trade.get("price_source_at_exit", "")
                        (_snap_dir / (_sid + ".json")).write_text(_json.dumps(_payload, default=str))
            except Exception as _e:
                _log(f"exit_snapshot_failed pair={trade.get('pair')} reason={_e}")
            trade["close_price"] = last_price

            closed_now.append(trade)
            log_close(trade)
            closed_total.append(trade)

            _log(
                f"exit pair={trade['pair']} status=closed reason={exit_reason} "
                f"pnl={trade.get('pnl_pct', 0.0):.3f}% hold={trade.get('hold_minutes', 0.0):.1f}m"
            )

            if self.telegram.enabled:
                try:
                    self.telegram.send(
                        f"🔴 CLOSE {trade['pair']}\n"
                        f"Strategy: {trade.get('strategy', '-')}\n"
                        f"Mode: {str(trade.get('execution_mode', 'paper')).upper()}\n"
                        f"Reason: {exit_reason}\n"
                        f"PnL: {float(trade.get('pnl_pct', 0.0) or 0.0):.3f}%\n"
                        f"Hold: {float(trade.get('hold_minutes', 0.0) or 0.0):.1f}m\n"
                        f"Entry: {float(trade.get('entry_price', 0.0) or 0.0):.6f}\n"
                        f"Last: {float(trade.get('last_price', 0.0) or 0.0):.6f}"
                    )
                except Exception as e:
                    _log(f"telegram_close_alert_failed pair={trade['pair']} error={e}")

        out = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "live_trading_enabled": state.get("live_trading_enabled", False),
            "private_api_ready": state.get("private_api_ready", False),
            "execution_mode": state.get("execution_mode", "paper"),
            "trading_paused": state.get("trading_paused", False),
            "free_usdc_snapshot": state.get("free_usdc_snapshot", 0.0),
            "open_trades": still_open,
            "created": created,
            "skipped": skipped,
            "open_count": len([t for t in still_open if t.get("status") == "open"]),
            "closed_trades": closed_total,
            "closed_now": closed_now,
            "closed_count_total": len(closed_total),
        }

        self._save_json(self.open_trades_path, out)

        _log(
            f"exit_summary open_count={out['open_count']} "
            f"closed_now={len(closed_now)} closed_total={out['closed_count_total']}"
        )

        return out
