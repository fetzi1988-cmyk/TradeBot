"""
ENDVISION-1h — Portfolio / Trade-Lebenszyklus mit ATR-Exit und echten Kosten.

Kernunterschiede zum 5m-System:
  - Kosten (round_trip_pct) sind HIER eingebaut und werden vom PnL abgezogen.
    Im 5m-System fehlte das komplett -> jeder PnL war brutto.
  - Exit ist ATR-relativ (pro Coin adaptiv), Bar-fuer-Bar, pessimistisch
    (Low vor High in derselben Kerze), identisch zur Backtest-Logik vom Aug 2026.
  - Positionsgroesse aus festem Kapitalrisiko (risk_per_trade_pct) und
    Stop-Abstand -> gleiches Dollar-Risiko ueber alle Coins, egal welche ATR.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Trade:
    pair: str
    strategy: str
    entry_ts: str
    entry_price: float
    atr_pct: float          # ATR in % zum Entry (fuer ATR-relative Stops)
    stake_usdc: float
    # Exit-Parameter (ATR-Faktoren, aus Config)
    init_stop_atr: float
    arm_mfe_atr: float
    trail_atr: float
    max_hold_bars: int
    # Laufzustand
    peak_ret_pct: float = 0.0
    armed: bool = False
    stop_ret_pct: float = field(init=False)   # aktueller Stop als Rendite vom Entry
    bars_held: int = 0
    # Ergebnis
    exit_ts: str | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    gross_pct: float | None = None
    net_pct: float | None = None
    mfe_pct: float = 0.0
    mae_pct: float = 0.0

    def __post_init__(self):
        # Initial-Stop: init_stop_atr * ATR unter dem Entry
        self.stop_ret_pct = -self.init_stop_atr * self.atr_pct

    def update_bar(self, high: float, low: float, close: float, ts: str,
                   cost_round_trip_pct: float) -> bool:
        """Verarbeitet eine 1h-Kerze. Gibt True zurueck, wenn der Trade schliesst.
        Renditen in % vom Entry. Pessimistisch: erst Stop (Low), dann Ziel/Hoch."""
        self.bars_held += 1
        lo_ret = (low / self.entry_price - 1.0) * 100.0
        hi_ret = (high / self.entry_price - 1.0) * 100.0
        self.mae_pct = min(self.mae_pct, lo_ret)
        self.mfe_pct = max(self.mfe_pct, hi_ret)

        # 1) Stop zuerst pruefen (pessimistisch)
        if lo_ret <= self.stop_ret_pct:
            self._close(self.stop_ret_pct, ts,
                        "trail_stop" if self.armed else "hard_stop",
                        cost_round_trip_pct)
            return True

        # 2) Hoch aktualisieren
        if hi_ret > self.peak_ret_pct:
            self.peak_ret_pct = hi_ret

        # 3) Trail scharfschalten, sobald MFE die arm-Schwelle erreicht
        if not self.armed and self.peak_ret_pct >= self.arm_mfe_atr * self.atr_pct:
            self.armed = True

        # 4) Trail nachziehen
        if self.armed:
            new_stop = self.peak_ret_pct - self.trail_atr * self.atr_pct
            if new_stop > self.stop_ret_pct:
                self.stop_ret_pct = new_stop

        # 5) Max. Haltedauer -> Exit zum Close
        if self.bars_held >= self.max_hold_bars:
            close_ret = (close / self.entry_price - 1.0) * 100.0
            self._close(close_ret, ts, "max_hold", cost_round_trip_pct)
            return True
        return False

    def _close(self, gross_ret_pct: float, ts: str, reason: str,
               cost_round_trip_pct: float):
        self.exit_ts = ts
        self.exit_reason = reason
        self.gross_pct = gross_ret_pct
        self.net_pct = gross_ret_pct - cost_round_trip_pct   # <-- Kosten hier
        self.exit_price = self.entry_price * (1.0 + gross_ret_pct / 100.0)


def position_size(capital_usdc: float, risk_per_trade_pct: float,
                  init_stop_atr: float, atr_pct: float) -> float:
    """Stake so, dass bei Stop genau risk_per_trade_pct des Kapitals verloren geht.
    stop_distance_pct = init_stop_atr * atr_pct  (in %)
    verlust_usdc = stake * stop_distance_pct/100  =!=  capital * risk/100
    -> stake = capital * risk / stop_distance_pct"""
    stop_distance_pct = init_stop_atr * atr_pct * 100.0
    if stop_distance_pct <= 0:
        return 0.0
    return capital_usdc * risk_per_trade_pct / stop_distance_pct
