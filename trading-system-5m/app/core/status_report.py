from __future__ import annotations

import json
import os
from pathlib import Path

import ccxt
from dotenv import load_dotenv

from app.core.control_state import ControlState


class StatusReport:
    def __init__(
        self,
        env_path: str = "/root/trading-system/.env",
        open_trades_path: str = "/root/trading-system/data/state/open_trades.json",
    ):
        self.env_path = Path(env_path)
        self.open_trades_path = Path(open_trades_path)

        load_dotenv(self.env_path)

        self.api_key = os.getenv("BINANCE_API_KEY", "").strip()
        self.api_secret = os.getenv("BINANCE_API_SECRET", "").strip()
        self.live_trading_enabled = os.getenv("LIVE_TRADING_ENABLED", "false").strip().lower() == "true"

        self.private_api_ready = bool(self.api_key and self.api_secret)
        self.control_state = ControlState()

    def _load_open_state(self) -> dict:
        if not self.open_trades_path.exists():
            return {
                "execution_mode": "paper",
                "open_trades": [],
                "open_count": 0,
                "private_api_ready": self.private_api_ready,
                "live_trading_enabled": self.live_trading_enabled,
                "free_usdc_snapshot": 0.0,
            }
        return json.loads(self.open_trades_path.read_text(encoding="utf-8"))

    def _get_live_free_usdc(self) -> float:
        if not self.private_api_ready:
            return 0.0
        try:
            ex = ccxt.binance({
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
            })
            bal = ex.fetch_balance()
            return float(bal.get("USDC", {}).get("free", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _open_trade_lines(self, open_trades: list[dict]) -> list[str]:
        lines = []
        for t in open_trades:
            if t.get("status") != "open":
                continue
            pair = t.get("pair", "-")
            strategy = t.get("strategy", "-")
            stake = float(t.get("stake_usdc", 0.0) or 0.0)
            pnl = float(t.get("pnl_pct", 0.0) or 0.0)
            hold = float(t.get("hold_minutes", 0.0) or 0.0)
            last_price = float(t.get("last_price", 0.0) or 0.0)
            lines.append(
                f"- {pair} | {strategy} | stake={stake:.2f} | pnl={pnl:.3f}% | hold={hold:.1f}m | last={last_price:.6f}"
            )
        return lines

    def build_status(self) -> str:
        state = self._load_open_state()

        execution_mode = str(state.get("execution_mode", "paper")).upper()
        free_usdc_snapshot = float(state.get("free_usdc_snapshot", 0.0) or 0.0)
        free_usdc_live = self._get_live_free_usdc()
        open_trades = state.get("open_trades", []) or []
        open_count = len([t for t in open_trades if t.get("status") == "open"])
        paused = self.control_state.is_paused()

        lines = []
        lines.append("📊 STATUS")
        lines.append(f"Mode: {execution_mode}")
        lines.append(f"Private API: {'READY' if self.private_api_ready else 'OFF'}")
        lines.append(f"Live Trading: {'ON' if self.live_trading_enabled else 'OFF'}")
        lines.append(f"Paused: {'YES' if paused else 'NO'}")
        lines.append(f"USDC Free (live): {free_usdc_live:.2f}")
        lines.append(f"USDC Snapshot: {free_usdc_snapshot:.2f}")
        lines.append(f"Open Trades: {open_count}")

        if open_count <= 0:
            lines.append("Open Positions: none")
            return "\n".join(lines)

        lines.append("Open Positions:")
        lines.extend(self._open_trade_lines(open_trades))
        return "\n".join(lines)

    def build_open(self) -> str:
        state = self._load_open_state()
        open_trades = state.get("open_trades", []) or []
        open_count = len([t for t in open_trades if t.get("status") == "open"])

        lines = ["📂 OPEN POSITIONS", f"Count: {open_count}"]
        if open_count <= 0:
            lines.append("none")
            return "\n".join(lines)

        lines.extend(self._open_trade_lines(open_trades))
        return "\n".join(lines)

    def build_balance(self) -> str:
        state = self._load_open_state()
        free_usdc_snapshot = float(state.get("free_usdc_snapshot", 0.0) or 0.0)
        free_usdc_live = self._get_live_free_usdc()

        return "\n".join([
            "💰 BALANCE",
            f"USDC Free (live): {free_usdc_live:.2f}",
            f"USDC Snapshot: {free_usdc_snapshot:.2f}",
            f"Private API: {'READY' if self.private_api_ready else 'OFF'}",
        ])

    def build_mode(self) -> str:
        state = self._load_open_state()
        execution_mode = str(state.get("execution_mode", "paper")).upper()
        paused = self.control_state.is_paused()

        return "\n".join([
            "⚙️ MODE",
            f"Execution Mode: {execution_mode}",
            f"Live Trading: {'ON' if self.live_trading_enabled else 'OFF'}",
            f"Private API: {'READY' if self.private_api_ready else 'OFF'}",
            f"Paused: {'YES' if paused else 'NO'}",
        ])

    def build_help(self) -> str:
        return "\n".join([
            "🤖 COMMANDS",
            "/status - kompletter Status",
            "/open - offene Trades",
            "/balance - freie USDC",
            "/mode - mode / pause / api",
            "/pause - neue Trades stoppen",
            "/resume - neue Trades wieder erlauben",
            "/help - diese Hilfe",
        ])

    def build(self) -> str:
        return self.build_status()
