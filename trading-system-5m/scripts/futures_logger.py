#!/usr/bin/env python3
"""Funding + Open Interest Logger — REINER LOGGER, parallel, kein Trading-Eingriff.
Laeuft alle 5min via Cron. Loggt OI-Velocity (Delta vs vorheriger Messung)."""
import ccxt, csv, json, os
from pathlib import Path
from datetime import datetime, timezone
import yaml, warnings; warnings.filterwarnings('ignore')

OUT   = Path("/root/trading-system/data/history/futures_flow.csv")
STATE = Path("/root/trading-system/data/state/futures_oi_last.json")
PAIRS = yaml.safe_load(open("/root/trading-system/config/pairs.yaml"))["pairs"]

def _log(m): print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] {m}")

fut = ccxt.binance({'options': {'defaultType': 'future'}})
fut.load_markets()
last_oi = json.loads(STATE.read_text()) if STATE.exists() else {}

new_file = not OUT.exists()
rows_written = 0
now = datetime.now(timezone.utc).isoformat()

with open(OUT, 'a', newline='') as f:
    w = csv.writer(f)
    if new_file:
        w.writerow(['ts_utc','pair','funding_rate','oi_amount','oi_value_usd',
                    'oi_delta_pct','price','price_delta_pct'])
    for p in PAIRS:
        base = p.split('/')[0]
        perp = f"{base}/USDT:USDT"
        if perp not in fut.markets: continue
        try:
            fr = fut.fetch_funding_rate(perp)
            funding = fr.get('fundingRate')
            price = fr.get('markPrice') or fr.get('indexPrice') or 0
            oi = fut.fetch_open_interest(perp)
            oi_amt = oi.get('openInterestAmount') or 0
            oi_val = oi.get('openInterestValue') or 0
            # Delta vs letzte Messung
            prev = last_oi.get(perp, {})
            oi_delta = ((oi_amt - prev.get('oi',oi_amt)) / prev.get('oi',oi_amt) * 100) if prev.get('oi') else 0
            px_delta = ((price - prev.get('px',price)) / prev.get('px',price) * 100) if prev.get('px') else 0
            w.writerow([now, p, funding, f"{oi_amt:.2f}", f"{oi_val:.0f}",
                        f"{oi_delta:+.3f}", f"{price:.8f}", f"{px_delta:+.3f}"])
            last_oi[perp] = {'oi': oi_amt, 'px': price}
            rows_written += 1
            # Auffaellige Kombis sofort loggen
            if abs(oi_delta) > 2.0 or abs(funding or 0) > 0.0005:
                _log(f"FLOW {p} funding={funding:+.5f} OI_delta={oi_delta:+.2f}% px_delta={px_delta:+.2f}%")
        except Exception as e:
            pass

STATE.write_text(json.dumps(last_oi))
_log(f"Fertig. {rows_written} Pairs geloggt.")
