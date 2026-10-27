"""Läuft alle 30 Min via Cron — holt Funding Rates für alle Watchlist Coins"""
import sys
sys.path.insert(0, "/root/trading-system")

import yaml
from app.core.macro_filter import fetch_macro_data, save_macro_state, get_btc_macro
from datetime import datetime, timezone

def main():
    pairs_cfg = yaml.safe_load(open("/root/trading-system/config/pairs.yaml"))
    watchlist = pairs_cfg if isinstance(pairs_cfg, list) else pairs_cfg.get("pairs", [])

    print(f"[{datetime.now(timezone.utc)}] Fetching macro data for {len(watchlist)} pairs...")

    btc = get_btc_macro()
    print(f"BTC Funding: {btc['funding_rate']:+.6f} → {btc['funding_class']}")

    macro = fetch_macro_data(watchlist)

    overheated = [p for p, d in macro.items() if d.get("funding_class") == "overheated"]
    capitulation = [p for p, d in macro.items() if d.get("funding_class") == "capitulation"]
    print(f"Overheated: {overheated}")
    print(f"Capitulation: {capitulation}")

    save_macro_state(macro)
    print("OK_MACRO_SAVED")

if __name__ == "__main__":
    main()
