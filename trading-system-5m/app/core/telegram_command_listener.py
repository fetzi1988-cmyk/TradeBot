from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv

from app.core.telegram_notifier import TelegramNotifier
from app.core.status_report import StatusReport
from app.core.control_state import ControlState


def _log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}")


class TelegramCommandListener:
    def __init__(
        self,
        env_path: str = "/root/trading-system/.env",
        offset_path: str = "/root/trading-system/data/state/telegram_offset.json",
    ):
        load_dotenv(env_path)

        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = str(os.getenv("TELEGRAM_CHAT_ID", "").strip())
        self.enabled = bool(self.bot_token and self.chat_id)

        self.offset_path = Path(offset_path)
        self.offset_path.parent.mkdir(parents=True, exist_ok=True)

        self.notifier = TelegramNotifier(env_path=env_path)
        self.status_report = StatusReport(env_path=env_path)
        self.control_state = ControlState()

    def _load_offset(self) -> int:
        if not self.offset_path.exists():
            return 0
        try:
            data = json.loads(self.offset_path.read_text(encoding="utf-8"))
            return int(data.get("last_update_id", 0) or 0)
        except Exception:
            return 0

    def _save_offset(self, update_id: int) -> None:
        self.offset_path.write_text(
            json.dumps(
                {
                    "last_update_id": int(update_id),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _get_updates(self, offset: int) -> list[dict]:
        if not self.enabled:
            return []

        url = (
            f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
            f"?timeout=10&offset={offset}"
        )

        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if not data.get("ok"):
            return []

        return data.get("result", []) or []

    def _extract_text(self, upd: dict) -> tuple[int | None, str | None]:
        msg = upd.get("message", {}) or {}
        chat = msg.get("chat", {}) or {}
        text = msg.get("text")
        chat_id = chat.get("id")
        return chat_id, text

    def _handle_command(self, text: str) -> str | None:
        cmd = text.strip().lower()

        if cmd in {"/status", "status"}:
            return self.status_report.build_status()

        if cmd in {"/open", "open"}:
            return self.status_report.build_open()

        if cmd in {"/balance", "balance"}:
            return self.status_report.build_balance()

        if cmd in {"/mode", "mode"}:
            return self.status_report.build_mode()

        if cmd in {"/pause", "pause"}:
            self.control_state.pause(source="telegram")
            return "⏸ Trading paused.\nNeue Trades sind blockiert."

        if cmd in {"/resume", "resume"}:
            self.control_state.resume(source="telegram")
            return "▶️ Trading resumed.\nNeue Trades sind wieder erlaubt."

        if cmd in {"/help", "help"}:
            return self.status_report.build_help()

        if cmd.startswith("/"):
            return self.status_report.build_help()

        return None

    def process_once(self) -> dict:
        if not self.enabled:
            return {"ok": False, "reason": "telegram_not_configured", "processed": 0}

        last_update_id = self._load_offset()
        updates = self._get_updates(last_update_id + 1)

        processed = 0
        handled = []

        for upd in updates:
            update_id = int(upd.get("update_id", 0) or 0)
            chat_id, text = self._extract_text(upd)

            self._save_offset(update_id)

            if str(chat_id) != self.chat_id:
                continue
            if not text:
                continue

            response = self._handle_command(text)
            if response:
                self.notifier.send(response)
                processed += 1
                handled.append(text.strip())
                _log(f"telegram_command handled={text.strip()}")

        return {
            "ok": True,
            "processed": processed,
            "handled": handled,
            "updates_seen": len(updates),
        }
