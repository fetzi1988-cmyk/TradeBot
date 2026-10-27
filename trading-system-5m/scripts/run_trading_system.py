import sys
from pathlib import Path

BASE_DIR = Path("/root/trading-system")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.runner import TradingSystemRunner

runner = TradingSystemRunner()
print("TRADING_SYSTEM_SERVICE_RUNNING")
runner.run_forever()
