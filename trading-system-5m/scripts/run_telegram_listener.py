import sys
import time
from pathlib import Path

BASE_DIR = Path("/root/trading-system")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.telegram_command_listener import TelegramCommandListener

tl = TelegramCommandListener()
print("TELEGRAM_COMMAND_SERVICE_RUNNING")

while True:
    try:
        tl.process_once()
    except Exception as e:
        print("TG_SERVICE_ERROR", e)
    time.sleep(3)
