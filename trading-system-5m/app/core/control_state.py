from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


class ControlState:
    def __init__(self, path: str = "/root/trading-system/data/state/control_state.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _default(self) -> dict:
        return {
            "trading_paused": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "system",
        }

    def load(self) -> dict:
        if not self.path.exists():
            data = self._default()
            self.save(data)
            return data
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            data = self._default()
            self.save(data)
            return data

    def save(self, payload: dict) -> None:
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def is_paused(self) -> bool:
        return bool(self.load().get("trading_paused", False))

    def pause(self, source: str = "telegram") -> dict:
        data = self.load()
        data["trading_paused"] = True
        data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        data["source"] = source
        self.save(data)
        return data

    def resume(self, source: str = "telegram") -> dict:
        data = self.load()
        data["trading_paused"] = False
        data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        data["source"] = source
        self.save(data)
        return data
