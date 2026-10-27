"""
WebSocket Liquidation Collector — Bybit V5 allLiquidation Stream.
Eigenstaendiger Service, isoliert vom Bot. REINER LOGGER.
Binance-fstream ist von diesem Server geoblockt -> Bybit als Quelle.
Bybit-Coins matchen Binance-Pairs 1:1 (BTCUSDT, ETHUSDT, ...).
S=Sell -> Long-Liquidation | S=Buy -> Short-Liquidation
"""
import asyncio, json, time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
import aiohttp, csv, yaml

def _log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] liq {msg}", flush=True)

OUT_DIR  = Path("data/liquidations"); OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = Path("data/history/liquidations.csv")
PAIRS_FILE = Path("config/pairs.yaml")
URL = "wss://stream.bybit.com/v5/public/linear"

_window = defaultdict(lambda: deque())  # symbol -> deque[(ts, side, usd, price)]

def _load_symbols():
    try:
        pairs = yaml.safe_load(PAIRS_FILE.read_text())["pairs"]
    except:
        pairs = ["BTC/USDC","ETH/USDC","SOL/USDC"]
    # /USDC -> USDT-Perp Symbol (Bybit fuehrt USDT-Perps)
    syms = []
    for p in pairs:
        base = p.split("/")[0]
        syms.append(f"{base}USDT")
    return syms

def _record(sym, side, qty, price, ts):
    usd = qty * price
    _window[sym].append((ts, side, usd, price))
    if usd > 20000:
        _log(f"BIG {sym} {'LONG' if side=='Sell' else 'SHORT'}_liq ${usd:,.0f} @ {price}")

def _prune(now_ms):
    cutoff = now_ms - 5*60*1000
    for dq in _window.values():
        while dq and dq[0][0] < cutoff:
            dq.popleft()

async def _persist_loop():
    new = not CSV_PATH.exists()
    while True:
        await asyncio.sleep(30)
        _prune(int(time.time()*1000))
        ts = datetime.now(timezone.utc).isoformat()
        try:
            with open(CSV_PATH, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["ts_utc","pair","long_liq_usd_5m","short_liq_usd_5m","liq_count_5m","net_liq_usd_5m"]); new=False
                for sym, dq in list(_window.items()):
                    if not dq: continue
                    long_usd  = sum(u for _,s,u,_ in dq if s=="Sell")
                    short_usd = sum(u for _,s,u,_ in dq if s=="Buy")
                    if long_usd+short_usd < 500: continue
                    w.writerow([ts, sym, f"{long_usd:.0f}", f"{short_usd:.0f}", len(dq), f"{short_usd-long_usd:.0f}"])
                    (OUT_DIR / (sym + ".json")).write_text(json.dumps({
                        "pair": sym, "ts": ts,
                        "long_liq_usd_5m": round(long_usd), "short_liq_usd_5m": round(short_usd),
                        "liq_count_5m": len(dq), "net_liq_usd_5m": round(short_usd-long_usd)}))
        except Exception as e:
            _log(f"persist error: {e}")

async def _stream():
    while True:
        try:
            syms = _load_symbols()
            args = [f"allLiquidation.{s}" for s in syms]
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(URL, heartbeat=20) as ws:
                    # Bybit: max 10 args pro subscribe-call -> batchen
                    for i in range(0, len(args), 10):
                        await ws.send_json({"op":"subscribe","args":args[i:i+10]})
                        await asyncio.sleep(0.2)
                    _log(f"subscribed {len(args)} Bybit liquidation channels")
                    async for msg in ws:
                        if msg.type==aiohttp.WSMsgType.TEXT:
                            d=json.loads(msg.data)
                            if "iquidation" in str(d.get("topic","")):
                                for item in d.get("data", []):
                                    try:
                                        _record(item["s"], item["S"], float(item["v"]), float(item["p"]), int(item["T"]))
                                    except: pass
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            _log("ws closed, reconnect in 5s"); break
        except Exception as e:
            _log(f"stream error: {e}, reconnect in 5s")
        await asyncio.sleep(5)

async def _run():
    await asyncio.gather(_stream(), _persist_loop())

if __name__ == "__main__":
    _log("Bybit liquidation collector starting")
    asyncio.run(_run())
