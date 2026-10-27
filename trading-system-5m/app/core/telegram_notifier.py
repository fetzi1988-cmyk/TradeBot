from __future__ import annotations

import os
import json
import urllib.request
import urllib.parse
from datetime import datetime, timezone

from dotenv import load_dotenv


def _log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}")


class TelegramNotifier:
    def __init__(self, env_path: str = "/root/trading-system/.env"):
        load_dotenv(env_path)
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(self.bot_token and self.chat_id)

    def send(self, text: str) -> dict:
        if not self.enabled:
            return {"ok": False, "reason": "telegram_not_configured"}

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = urllib.parse.urlencode({
            "chat_id": self.chat_id,
            "text": text,
        }).encode()

        req = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        _log(f"telegram_send ok={data.get('ok')}")
        return data
