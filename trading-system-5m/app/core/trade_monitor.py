from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}")


class TradeMonitor:
    def __init__(
        self,
        open_trades_path: str = "data/state/open_trades.json",
        routing_path: str = "data/state/routing.json",
    ):
        self.open_trades_path = Path(open_trades_path)
        self.routing_path = Path(routing_path)

    def _load_json(self, path: Path, default):
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_json(self, path: Path, data) -> None:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _minutes_open(self, entry_time_utc: str) -> float:
        entry_dt = datetime.fromisoformat(entry_time_utc)
        now_dt = datetime.now(timezone.utc)
        return round((now_dt - entry_dt).total_seconds() / 60.0, 1)

    def run(self) -> dict:
        state = self._load_json(self.open_trades_path, {"open_trades": []})
        routing = self._load_json(self.routing_path, {"routes": []})

        open_trades = state.get("open_trades", [])
        routes = routing.get("routes", [])
        route_map = {r["pair"]: r for r in routes}

        updated = []

        for trade in open_trades:
            if trade.get("status") != "open":
                updated.append(trade)
                continue

            pair = trade["pair"]
            route = route_map.get(pair)

            if not route:
                trade["monitor_status"] = "missing_route"
                updated.append(trade)
                _log(f"monitor pair={pair} skipped=missing_route")
                continue

            current_price = float(route.get("close", trade.get("last_price", 0.0)) or 0.0)
            entry_price = float(trade.get("entry_price", 0.0) or 0.0)

            trade["last_price"] = current_price
            trade["highest_price"] = max(float(trade.get("highest_price", current_price)), current_price)
            trade["lowest_price"] = min(float(trade.get("lowest_price", current_price)), current_price)
            trade["hold_minutes"] = self._minutes_open(trade["entry_time_utc"])

            if entry_price > 0:
                trade["pnl_pct"] = round(((current_price / entry_price) - 1.0) * 100.0, 3)
            else:
                trade["pnl_pct"] = 0.0

            trade["monitor_status"] = "updated"

            _log(
                f"monitor pair={pair} entry={entry_price:.6f} last={current_price:.6f} "
                f"pnl={trade['pnl_pct']:.3f}% hold={trade['hold_minutes']:.1f}m"
            )

            updated.append(trade)

        out = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "open_trades": updated,
            "created": state.get("created", []),
            "skipped": state.get("skipped", []),
            "open_count": len([t for t in updated if t.get("status") == "open"]),
        }

        self._save_json(self.open_trades_path, out)

        _log(f"monitor_summary open_count={out['open_count']}")
        return out
