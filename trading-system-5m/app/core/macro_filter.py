"""
Macro Filter — Funding Rate + Open Interest als Markt-Kontext
Holt Daten von Binance Perpetual Futures als Makro-Signal
für unsere Spot USDC Trades.
"""
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import ccxt


def _log(msg):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}")


USDC_TO_PERP = {
    "PEPE/USDC":  None,
    "MATIC/USDC": None,
}

STATE_PATH = Path("/root/trading-system/data/state/macro_filter.json")


def get_perp_symbol(spot_pair: str) -> str | None:
    if spot_pair in USDC_TO_PERP:
        return USDC_TO_PERP[spot_pair]
    coin = spot_pair.replace("/USDC", "")
    return f"{coin}/USDT:USDT"


def classify_funding(rate: float) -> str:
    if rate > 0.0005:    return "overheated"    # >0.05%/8h — zu viele Longs
    if rate > 0.0002:    return "bullish"        # normal bullish
    if rate > -0.0001:   return "neutral"        # ausgeglichen
    if rate > -0.0003:   return "bearish"        # bearish
    return "capitulation"                         # <-0.03% — Shorts dominieren


def fetch_macro_data(watchlist: list[str]) -> dict:
    """
    Holt Funding Rate + OI für alle Coins in der Watchlist.
    Gibt dict zurück: pair -> {funding_rate, funding_class, oi, oi_change}
    """
    exchange = ccxt.binance({"enableRateLimit": True})
    result = {}

    for spot_pair in watchlist:
        perp = get_perp_symbol(spot_pair)
        if not perp:
            result[spot_pair] = {
                "funding_rate": 0.0,
                "funding_class": "neutral",
                "oi": None,
                "available": False
            }
            continue

        try:
            fr = exchange.fetch_funding_rate(perp)
            rate = float(fr.get("fundingRate") or 0.0)
            fclass = classify_funding(rate)

            result[spot_pair] = {
                "funding_rate": rate,
                "funding_class": fclass,
                "oi": None,
                "available": True
            }
        except Exception as e:
            result[spot_pair] = {
                "funding_rate": 0.0,
                "funding_class": "neutral",
                "oi": None,
                "available": False,
                "error": str(e)
            }
        time.sleep(0.1)

    return result


def should_trade(spot_pair: str, strategy: str, macro_data: dict) -> tuple[bool, str]:
    """
    Entscheidet ob ein Trade erlaubt ist basierend auf Makro-Kontext.
    Returns: (allowed, reason)
    """
    data = macro_data.get(spot_pair, {})
    if not data.get("available", False):
        return True, "no_macro_data"

    rate = data["funding_rate"]
    fclass = data["funding_class"]

    # Überhitzter Markt — nur Reversal/Range erlaubt
    if fclass == "overheated":
        if strategy in ("trend", "breakout", "momentum", "trend_rider"):
            return False, f"funding_overheated_{rate:.6f}"
        return True, f"funding_overheated_but_{strategy}_ok"

    # Kapitulation — keine Shorts/Range, nur Reversal/Momentum
    if fclass == "capitulation":
        if strategy in ("range",):
            return False, f"funding_capitulation_skip_range"
        return True, f"funding_capitulation_{strategy}_ok"

    # Normal — alles erlaubt
    return True, f"funding_{fclass}"


def save_macro_state(macro_data: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "data": macro_data
    }
    STATE_PATH.write_text(json.dumps(state, indent=2))


def load_macro_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        state = json.loads(STATE_PATH.read_text())
        return state.get("data", {})
    except Exception:
        return {}


def get_btc_macro() -> dict:
    """BTC als Gesamt-Markt-Indikator"""
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        fr = exchange.fetch_funding_rate("BTC/USDT:USDT")
        rate = float(fr.get("fundingRate") or 0.0)
        return {
            "funding_rate": rate,
            "funding_class": classify_funding(rate),
        }
    except Exception:
        return {"funding_rate": 0.0, "funding_class": "neutral"}
