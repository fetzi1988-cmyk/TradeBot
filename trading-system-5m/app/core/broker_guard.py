from __future__ import annotations

import os
from datetime import datetime, timezone

import ccxt
from dotenv import load_dotenv


def _log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}")


class BrokerGuard:
    def __init__(self, exchange_name: str = "binance", env_path: str = "/root/trading-system/.env"):
        load_dotenv(env_path)

        self.exchange_name = exchange_name
        self.api_key = os.getenv("BINANCE_API_KEY", "").strip()
        self.api_secret = os.getenv("BINANCE_API_SECRET", "").strip()

        kwargs = {
            "enableRateLimit": True,
        }

        if self.api_key and self.api_secret and exchange_name.lower() == "binance":
            kwargs["apiKey"] = self.api_key
            kwargs["secret"] = self.api_secret

        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class(kwargs)
        self.markets = self.exchange.load_markets()

        _log(
            f"broker_guard_ready exchange={self.exchange_name} "
            f"markets={len(self.markets)} auth={'on' if (self.api_key and self.api_secret) else 'off'}"
        )

    def _filter_value(self, market: dict, filter_type: str, key: str):
        filters = market.get("info", {}).get("filters", []) or []
        for f in filters:
            if f.get("filterType") == filter_type and key in f:
                try:
                    return float(f.get(key))
                except Exception:
                    return None
        return None

    def get_symbol_rules(self, symbol: str) -> dict:
        market = self.exchange.market(symbol)

        min_qty = self._filter_value(market, "LOT_SIZE", "minQty")
        step_size = self._filter_value(market, "LOT_SIZE", "stepSize")
        tick_size = self._filter_value(market, "PRICE_FILTER", "tickSize")

        min_notional = self._filter_value(market, "NOTIONAL", "minNotional")
        if min_notional is None:
            min_notional = self._filter_value(market, "MIN_NOTIONAL", "minNotional")

        if min_qty is None:
            min_qty = float((market.get("limits", {}).get("amount", {}) or {}).get("min") or 0.0)

        if min_notional is None:
            min_notional = float((market.get("limits", {}).get("cost", {}) or {}).get("min") or 0.0)

        return {
            "symbol": symbol,
            "active": bool(market.get("active", False)),
            "spot": bool(market.get("spot", False)),
            "min_qty": float(min_qty or 0.0),
            "step_size": float(step_size or 0.0),
            "tick_size": float(tick_size or 0.0),
            "min_notional": float(min_notional or 0.0),
            "amount_precision": market.get("precision", {}).get("amount"),
            "price_precision": market.get("precision", {}).get("price"),
        }

    def preview_order(self, symbol: str, stake_usdc: float, entry_price: float) -> dict:
        rules = self.get_symbol_rules(symbol)

        raw_amount = (stake_usdc / entry_price) if entry_price > 0 else 0.0

        try:
            rounded_amount = float(self.exchange.amount_to_precision(symbol, raw_amount)) if raw_amount > 0 else 0.0
        except Exception:
            rounded_amount = 0.0

        try:
            rounded_price = float(self.exchange.price_to_precision(symbol, entry_price)) if entry_price > 0 else 0.0
        except Exception:
            rounded_price = 0.0

        final_notional = rounded_amount * rounded_price

        reason = None
        if not rules["active"]:
            reason = "market_inactive"
        elif not rules["spot"]:
            reason = "not_spot_market"
        elif rounded_amount <= 0:
            reason = "amount_rounds_to_zero"
        elif rules["min_qty"] > 0 and rounded_amount < rules["min_qty"]:
            reason = "below_min_qty"
        elif rules["min_notional"] > 0 and final_notional < rules["min_notional"]:
            reason = "below_min_notional"

        status = "ok" if reason is None else "blocked"

        out = {
            "symbol": symbol,
            "stake_usdc": round(float(stake_usdc), 2),
            "entry_price": float(entry_price),
            "raw_amount": float(raw_amount),
            "rounded_amount": float(rounded_amount),
            "rounded_price": float(rounded_price),
            "final_notional": float(final_notional),
            "status": status,
            "reason": reason,
            **rules,
        }

        _log(
            f"broker_preview symbol={symbol} status={status} "
            f"stake={stake_usdc:.2f} price={entry_price:.6f} "
            f"amount={rounded_amount} notional={final_notional:.2f} "
            f"reason={reason or '-'}"
        )

        return out
