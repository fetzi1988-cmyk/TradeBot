"""
WebSocket CVD Collector — Binance aggTrade Stream
Dynamische Pairs — erkennt Änderungen in pairs.yaml und reconnectet.
"""
import asyncio, json, threading, time, yaml
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

def _log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] cvd {msg}")

CVD_DIR      = Path("data/cvd")
PAIRS_FILE   = Path("config/pairs.yaml")
CVD_DIR.mkdir(parents=True, exist_ok=True)

_state: dict = defaultdict(lambda: {
    "cvd_candle": 0.0, "cvd_cum": 0.0,
    "buy_vol": 0.0, "sell_vol": 0.0,
    "trade_count": 0, "candle_start": None,
    "history": [],
})

def _load_pairs() -> list:
    try:
        return yaml.safe_load(PAIRS_FILE.read_text())["pairs"]
    except:
        return []

def _pair_to_symbol(pair): return pair.replace("/","").upper()
def _symbol_to_pair(sym):
    return (sym[:-4] + "/USDC") if sym.endswith("USDC") else sym

def _bucket(ts_ms): return (ts_ms // 300000) * 300000

def _load_history(pair):
    """Beim Start vorhandene CVD-History einlesen"""
    try:
        fname = CVD_DIR / (pair.replace("/","_") + "_cvd.json")
        if fname.exists():
            import json as _j
            data = _j.loads(fname.read_text())
            hist = data.get("history", [])
            if hist:
                _state[pair]["history"] = hist[-20:]
                _state[pair]["cvd_cum"] = float(data.get("cvd_cum", 0))
    except:
        pass

def _on_trade(symbol, price, qty, is_buyer_maker, ts_ms):
    pair = _symbol_to_pair(symbol)
    s = _state[pair]
    b = _bucket(ts_ms)
    if s["candle_start"] is not None and b != s["candle_start"]:
        s["history"].append({
            "ts": s["candle_start"],
            "cvd": round(s["cvd_candle"], 4),
            "buy_vol": round(s["buy_vol"], 4),
            "sell_vol": round(s["sell_vol"], 4),
            "trades": s["trade_count"],
        })
        s["history"] = s["history"][-20:]
        s["cvd_candle"] = s["buy_vol"] = s["sell_vol"] = s["trade_count"] = 0
    s["candle_start"] = b
    notional = price * qty
    if is_buyer_maker:
        s["cvd_candle"] -= notional; s["sell_vol"] += notional
    else:
        s["cvd_candle"] += notional; s["buy_vol"] += notional
    s["cvd_cum"] += notional * (1 if not is_buyer_maker else -1)
    s["trade_count"] += 1

def _save(pair):
    s = _state[pair]
    h = s["history"]
    cvd_5  = sum(x["cvd"] for x in h[-5:])
    cvd_10 = sum(x["cvd"] for x in h[-10:])

    # CVD Buildup Score — misst stille Akkumulation
    # Profis kaufen VOR dem Move: CVD steigt bei flachem/fallendem Preis
    buildup_score = 0.0
    cvd_trend_6 = 0.0
    price_range_pct = 0.0
    buy_sell_ratio = 0.0
    if len(h) >= 6:
        cvds = [x["cvd"] for x in h[-6:]]
        buys = [x["buy_vol"] for x in h[-6:]]
        sells = [x["sell_vol"] for x in h[-6:]]
        # CVD-Trend: positiv = Käufer bauen auf
        cvd_trend_6 = cvds[-1] - cvds[0]
        # Buy/Sell Ratio
        total_buy = sum(buys)
        total_sell = sum(sells) + 0.001
        buy_sell_ratio = round(total_buy / total_sell, 4)
        # Buildup: CVD steigt = +1, Buy/Sell > 1.5 = +1, CVD positiv 4/6 Kerzen = +1
        pos_candles = sum(1 for c in cvds if c > 0)
        score = 0
        if cvd_trend_6 > 0: score += 1
        if buy_sell_ratio > 1.5: score += 1
        if pos_candles >= 4: score += 1
        buildup_score = score  # 0-3, 3 = starke stille Akkumulation

    data = {
        "pair": pair,
        "ts": datetime.now(timezone.utc).isoformat(),
        "cvd_candle": round(s["cvd_candle"], 4),
        "cvd_5":  round(cvd_5, 4),
        "cvd_10": round(cvd_10, 4),
        "cvd_cum": round(s["cvd_cum"], 4),
        "buy_vol": round(s["buy_vol"], 4),
        "sell_vol": round(s["sell_vol"], 4),
        "cvd_rising":  cvd_5 > 0,
        "cvd_falling": cvd_5 < 0,
        "cvd_bear_div": False,
        "cvd_bull_div": False,
        "trade_count": s["trade_count"],
        "cvd_trend_6": round(cvd_trend_6, 4),
        "buy_sell_ratio": buy_sell_ratio,
        "buildup_score": buildup_score,
        "history": h[-10:],
    }
    (CVD_DIR / (pair.replace("/","_") + "_cvd.json")).write_text(
        json.dumps(data, indent=2))

async def _stream_pairs(pairs: list):
    if not pairs:
        await asyncio.sleep(5); return
    syms    = [_pair_to_symbol(p).lower() for p in pairs]
    streams = "/".join(f"{s}@aggTrade" for s in syms)
    url     = f"wss://stream.binance.com:9443/stream?streams={streams}"
    # History aus letztem Run laden
    for _p in pairs:
        _load_history(_p)
    for _p in pairs:
        _load_history(_p)
    _log(f"connect pairs={len(pairs)} history_loaded")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url, heartbeat=20) as ws:
                _log("connected")
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        d = json.loads(msg.data).get("data", {})
                        if d.get("e") == "aggTrade":
                            _on_trade(d["s"],float(d["p"]),
                                      float(d["q"]),bool(d["m"]),int(d["T"]))
                    elif msg.type in (aiohttp.WSMsgType.ERROR,
                                      aiohttp.WSMsgType.CLOSED):
                        _log("ws closed"); break
    except Exception as e:
        _log(f"stream error: {e}")

async def _run():
    current_pairs = []
    pairs_mtime   = 0.0
    save_task     = None
    stream_task   = None

    async def _save_loop(pairs):
        while True:
            await asyncio.sleep(10)
            for p in pairs:
                try: _save(p)
                except: pass

    while True:
        # Pairs-Änderung erkennen
        try:
            mtime = PAIRS_FILE.stat().st_mtime
        except:
            mtime = 0.0

        if mtime != pairs_mtime:
            new_pairs = _load_pairs()
            if set(new_pairs) != set(current_pairs):
                _log(f"pairs changed {len(current_pairs)}→{len(new_pairs)} — reconnect")
                # Alte Tasks canceln
                if stream_task and not stream_task.done():
                    stream_task.cancel()
                if save_task and not save_task.done():
                    save_task.cancel()
                current_pairs = new_pairs
                pairs_mtime   = mtime
                stream_task = asyncio.create_task(_stream_pairs(current_pairs))
                save_task   = asyncio.create_task(_save_loop(current_pairs))
            else:
                pairs_mtime = mtime

        await asyncio.sleep(30)  # Alle 30s auf Änderungen prüfen

def start():
    def _thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run())
    t = threading.Thread(target=_thread, daemon=True, name="cvd_collector")
    t.start()
    _log("thread started")
    return t
