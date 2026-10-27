"""
ENDVISION-1h — Live-Runner & Portfolio-Engine (Maker/Limit-Entry).

EIN Kopf, zwei Modi:

  --replay : spielt das GESAMTE Portfolio bar-fuer-bar ueber die vorhandene
             1h-Historie durch. Anders als der Trade-Mittelwert-Backtest
             respektiert er echtes Kapital, max_open_trades (Slot-Konkurrenz),
             realistische asymmetrische Kosten (Maker-Entry + Taker-Exit) und
             liefert die ECHTE Equity-Kurve, Max-Drawdown und Monatsrenditen.
             -> beweist, ob das KONTO Geld macht, nicht nur der Durchschnitts-Trade.

  --live   : derselbe Entscheidungs-Kopf als stuendlicher Tick. Liest die zuletzt
             GESCHLOSSENE 1h-Kerze, verwaltet offene Positionen und offene
             Limit-Orders in einer State-Datei. Paper (Default) loggt die Orders;
             live.enabled=true in der Config wuerde echte ccxt-Orders senden.
             Vor dem ersten Live-Tick MUSS der Replay sauber sein.

Aufruf (im trading-system, damit .venv + Daten da sind):
  .venv/bin/python3 /root/TradeBot/core/live_runner_1h.py --replay \
      --config /root/TradeBot/config/config_1h.yaml --candles data/candles
  .venv/bin/python3 /root/TradeBot/core/live_runner_1h.py --live \
      --config /root/TradeBot/config/config_1h.yaml --candles data/candles
"""
from __future__ import annotations
import sys, os, glob, json, argparse, warnings, time
warnings.filterwarnings("ignore")
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.portfolio_1h import Trade

# Strategien mit robuster, fill-realistischer Kante (Fill-Qualitaets-Test 2026-10).
# Bewusst NUR diese zwei live — der Rest war im Taker-Backtest nicht robust.
LIVE_STRATS = ["rsi_cross", "pullback"]
FILL_WINDOW = 2            # Kerzen, die die Limit-Order Zeit hat, gefuellt zu werden
MIN_PENETRATION_PCT = 0.05 # Fill zaehlt nur, wenn Preis >=0.05% unter das Limit handelt
                           # (keine fragilen 1-Tick-Wicks; aus fill_quality.py abgeleitet)

# Realistische, asymmetrische Kosten:
#   Entry = Maker-Limit  -> 0.10% (eine Seite, Binance-Spot-Maker)
#   Exit  = Taker-Market -> Trailing-Stop schlaegt gegen dich durch, also Market:
#           0.10% Fee + ~0.035% halber Spread + Slippage  ~= 0.15%
COST_ENTRY_PCT = 0.10
COST_EXIT_PCT = 0.15


def sized_stake(equity, risk_pct, init_stop_atr, atr_pct, max_frac):
    """Positionsgroesse: so, dass bei Initial-Stop genau risk_pct der Equity faellt.
    stop_distance_pct = init_stop_atr * atr_pct  (atr_pct ist in %-Einheiten, z.B. 0.8).
    stake = equity * risk_pct / stop_distance_pct, gedeckelt auf max_frac der Equity.
    (Fix gegenueber portfolio_1h.position_size, das faelschlich * 100 rechnete.)"""
    stop_distance_pct = init_stop_atr * atr_pct
    if stop_distance_pct <= 0:
        return 0.0
    stake = equity * risk_pct / stop_distance_pct
    return min(stake, equity * max_frac)


# ----------------------------------------------------------------------------- #
#  REPLAY — Portfolio-Simulation ueber die ganze Historie
# ----------------------------------------------------------------------------- #
def load_pairs(candles_dir):
    """Laedt alle *_1h_features.csv in Arrays + ts->index-Map."""
    out = {}
    for fp in sorted(glob.glob(os.path.join(candles_dir, "*_1h_features.csv"))):
        pair = os.path.basename(fp).replace("_1h_features.csv", "")
        try:
            d = pd.read_csv(fp)
        except Exception:
            continue
        if len(d) < 60 or "timestamp" not in d.columns:
            continue
        for c in ["close", "high", "low", "atr_pct"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.sort_values("timestamp").reset_index(drop=True)
        sigs = {s: (d[f"signal_{s}"].fillna(False).astype(bool).values
                    if f"signal_{s}" in d.columns else np.zeros(len(d), bool))
                for s in LIVE_STRATS}
        ts = d["timestamp"].astype("int64").values
        if len(ts) and np.median(ts) < 1e12:   # Sekunden -> Millisekunden
            ts = ts * 1000
        out[pair] = {
            "ts": ts, "idx": {int(t): i for i, t in enumerate(ts)},
            "close": d["close"].values, "high": d["high"].values,
            "low": d["low"].values, "atr": d["atr_pct"].values, "sig": sigs,
        }
    return out


def replay(cfg, candles_dir, cost_entry, cost_exit):
    rk = cfg["risk"]; ex = cfg["exit"]
    cap0 = float(rk.get("start_capital_usdc", 1000.0))
    risk_pct = float(rk.get("risk_per_trade_pct", 1.0))
    max_open = int(rk.get("max_open_trades", 6))
    max_frac = 1.0 / max_open          # Kapital gleichmaessig auf die Slots verteilen
    cost_rt = cost_entry + cost_exit

    P = load_pairs(candles_dir)
    if not P:
        return None
    timeline = sorted({int(t) for p in P.values() for t in p["ts"]})

    cash = cap0
    open_pos = []      # dicts: trade, pair, stake, entry_idx
    pending = []       # dicts: pair, strat, limit, atr, sig_ts, expire_ts
    closed = []
    equity_curve = []  # (ts, equity)
    peak = cap0; max_dd = 0.0

    for t in timeline:
        # 1) OFFENE Positionen mit der aktuellen Kerze fortschreiben
        still_open = []
        for pos in open_pos:
            pd_ = P[pos["pair"]]; i = pd_["idx"].get(t)
            if i is None or t <= pos["entry_ts"]:
                still_open.append(pos); continue
            done = pos["trade"].update_bar(pd_["high"][i], pd_["low"][i],
                                           pd_["close"][i], str(t), cost_rt)
            if done:
                pnl_frac = pos["trade"].net_pct / 100.0
                cash += pos["stake"] * (1.0 + pnl_frac)
                closed.append({"pair": pos["pair"], "strategy": pos["trade"].strategy,
                               "entry_ts": pos["trade"].entry_ts, "exit_ts": str(t),
                               "net_pct": pos["trade"].net_pct,
                               "reason": pos["trade"].exit_reason,
                               "pnl_usdc": pos["stake"] * pnl_frac})
            else:
                still_open.append(pos)
        open_pos = still_open

        # 2) OFFENE Limit-Orders pruefen (aelteste zuerst)
        still_pending = []
        for o in sorted(pending, key=lambda x: x["sig_ts"]):
            pd_ = P[o["pair"]]; i = pd_["idx"].get(t)
            if i is None or t <= o["sig_ts"]:
                if t < o["expire_ts"]:
                    still_pending.append(o)
                continue
            lo = pd_["low"][i]; limit = o["limit"]
            pen = (limit - lo) / limit * 100.0 if limit > 0 else -1
            if lo <= limit and pen >= MIN_PENETRATION_PCT:
                # Equity jetzt = cash + Marktwert offener Positionen
                mtm = sum(p["stake"] * (1.0 + (P[p["pair"]]["close"][P[p["pair"]]["idx"][t]]
                           / p["trade"].entry_price - 1.0)) if P[p["pair"]]["idx"].get(t) is not None
                          else p["stake"] for p in open_pos)
                equity = cash + mtm
                if len(open_pos) < max_open:
                    stake = sized_stake(equity, risk_pct, ex["init_stop_atr"],
                                        o["atr"], max_frac)
                    if stake >= 10.0 and stake <= cash:
                        tr = Trade(pair=o["pair"], strategy=o["strat"], entry_ts=str(t),
                                   entry_price=float(limit), atr_pct=float(o["atr"]),
                                   stake_usdc=stake,
                                   init_stop_atr=float(ex["init_stop_atr"]),
                                   arm_mfe_atr=float(ex["arm_mfe_atr"]),
                                   trail_atr=float(ex["trail_atr"]),
                                   max_hold_bars=int(ex["max_hold_bars"]))
                        cash -= stake
                        open_pos.append({"trade": tr, "pair": o["pair"],
                                         "stake": stake, "entry_ts": t})
                # gefuellt oder Slot/Cash voll -> Order ist erledigt (kein Re-Queue)
                continue
            if t < o["expire_ts"]:
                still_pending.append(o)
        pending = still_pending

        # 3) NEUE Signale dieser Kerze als Limit-Orders einstellen
        for pair, pd_ in P.items():
            i = pd_["idx"].get(t)
            if i is None:
                continue
            for s in LIVE_STRATS:
                if pd_["sig"][s][i] and pd_["atr"][i] > 0 and pd_["close"][i] > 0:
                    expire = pd_["ts"][min(i + FILL_WINDOW, len(pd_["ts"]) - 1)]
                    pending.append({"pair": pair, "strat": s, "limit": pd_["close"][i],
                                    "atr": pd_["atr"][i], "sig_ts": t, "expire_ts": expire})

        # 4) Equity sampeln + Drawdown
        mtm = 0.0
        for p in open_pos:
            j = P[p["pair"]]["idx"].get(t)
            px = P[p["pair"]]["close"][j] if j is not None else p["trade"].entry_price
            mtm += p["stake"] * (1.0 + (px / p["trade"].entry_price - 1.0))
        equity = cash + mtm
        equity_curve.append((t, equity))
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100.0)

    return {"cap0": cap0, "closed": pd.DataFrame(closed),
            "equity": equity_curve, "max_dd": max_dd,
            "cost_rt": cost_rt, "max_open": max_open}


def report_replay(res, label):
    c = res["closed"]; cap0 = res["cap0"]
    end_eq = res["equity"][-1][1] if res["equity"] else cap0
    ret = (end_eq / cap0 - 1.0) * 100.0
    dur_days = (res["equity"][-1][0] - res["equity"][0][0]) / 86400000.0 if res["equity"] else 0
    print("=" * 84)
    print(f"REPLAY  [{label}]  Round-Trip-Kosten={res['cost_rt']:.2f}%  Slots={res['max_open']}")
    print("=" * 84)
    print(f"  Start {cap0:.0f} USDC  ->  Ende {end_eq:.0f} USDC   "
          f"({ret:+.1f}% ueber {dur_days:.0f} Tage)")
    print(f"  Trades={len(c)}  Max-Drawdown={res['max_dd']:.1f}%")
    if len(c):
        wr = (c["net_pct"] > 0).mean() * 100
        print(f"  Netto-WR={wr:.1f}%  Summe-PnL={c['pnl_usdc'].sum():+.1f} USDC")
        c = c.copy(); c["m"] = pd.to_datetime(c["exit_ts"].astype("int64"), unit="ms", utc=True).dt.strftime("%Y-%m")
        mon = c.groupby("m")["pnl_usdc"].sum()
        print("  Monats-PnL (USDC): " + "  ".join(f"{m}:{v:+.0f}" for m, v in mon.items()))
        rc = c["reason"].value_counts().to_dict()
        print("  Exit-Gruende: " + "  ".join(f"{k}:{v}" for k, v in rc.items()))
    if dur_days > 0 and ret != 0:
        ann = ((end_eq / cap0) ** (365.0 / dur_days) - 1.0) * 100.0
        print(f"  annualisiert ~ {ann:+.0f}%  (grobe Orientierung, keine Prognose)")


# ----------------------------------------------------------------------------- #
#  LIVE — ein stuendlicher Tick (Paper-Default)
# ----------------------------------------------------------------------------- #
def live_tick(cfg, candles_dir, state_path):
    """Liest die zuletzt GESCHLOSSENE 1h-Kerze je Pair, verwaltet offene Orders
    und Positionen in state_path. Paper: loggt Entscheidungen. Live: ccxt (gated)."""
    live_on = bool(cfg.get("live", {}).get("enabled", False))
    mode = "LIVE (echte Orders!)" if live_on else "PAPER (nur Log)"
    st = {"open": [], "pending": [], "cash": cfg["risk"].get("start_capital_usdc", 1000.0),
          "closed": []}
    if os.path.exists(state_path):
        try:
            st = json.load(open(state_path))
        except Exception:
            pass

    ex = cfg["exit"]; rk = cfg["risk"]
    max_open = int(rk.get("max_open_trades", 6)); max_frac = 1.0 / max_open
    cost_rt = COST_ENTRY_PCT + COST_EXIT_PCT
    now = int(time.time() * 1000)
    actions = []

    files = sorted(glob.glob(os.path.join(candles_dir, "*_1h_features.csv")))
    latest = {}
    for fp in files:
        pair = os.path.basename(fp).replace("_1h_features.csv", "")
        try:
            d = pd.read_csv(fp).sort_values("timestamp").reset_index(drop=True)
        except Exception:
            continue
        if len(d) < 2:
            continue
        row = d.iloc[-2].copy()      # vorletzte = zuletzt GESCHLOSSENE Kerze (letzte laeuft noch)
        if int(row["timestamp"]) < 1e12:   # Sekunden -> Millisekunden
            row["timestamp"] = int(row["timestamp"]) * 1000
        latest[pair] = row

    # 1) offene Positionen mit der zuletzt geschlossenen Kerze managen
    still_open = []
    for pos in st["open"]:
        row = latest.get(pos["pair"])
        if row is None:
            still_open.append(pos); continue
        tr = Trade(pair=pos["pair"], strategy=pos["strategy"], entry_ts=pos["entry_ts"],
                   entry_price=pos["entry_price"], atr_pct=pos["atr_pct"], stake_usdc=pos["stake"],
                   init_stop_atr=ex["init_stop_atr"], arm_mfe_atr=ex["arm_mfe_atr"],
                   trail_atr=ex["trail_atr"], max_hold_bars=ex["max_hold_bars"])
        tr.peak_ret_pct = pos.get("peak_ret_pct", 0.0); tr.armed = pos.get("armed", False)
        tr.stop_ret_pct = pos.get("stop_ret_pct", tr.stop_ret_pct)
        tr.bars_held = pos.get("bars_held", 0)
        done = tr.update_bar(float(row["high"]), float(row["low"]), float(row["close"]),
                             str(int(row["timestamp"])), cost_rt)
        if done:
            pnl = pos["stake"] * tr.net_pct / 100.0
            st["cash"] += pos["stake"] + pnl
            actions.append(f"EXIT  {pos['pair']:12s} {tr.exit_reason:10s} net={tr.net_pct:+.2f}%  PnL={pnl:+.1f} USDC")
            st["closed"].append({"pair": pos["pair"], "net_pct": tr.net_pct, "reason": tr.exit_reason})
            # LIVE: hier Market-Sell senden
        else:
            pos.update(peak_ret_pct=tr.peak_ret_pct, armed=tr.armed,
                       stop_ret_pct=tr.stop_ret_pct, bars_held=tr.bars_held)
            still_open.append(pos)
    st["open"] = still_open

    # 2) offene Limit-Orders pruefen
    still_pending = []
    for o in sorted(st["pending"], key=lambda x: x["sig_ts"]):
        row = latest.get(o["pair"])
        if row is None:
            still_pending.append(o); continue
        lo = float(row["low"]); limit = o["limit"]
        pen = (limit - lo) / limit * 100.0 if limit > 0 else -1
        if lo <= limit and pen >= MIN_PENETRATION_PCT and len(st["open"]) < max_open:
            equity = st["cash"] + sum(p["stake"] for p in st["open"])
            stake = sized_stake(equity, rk["risk_per_trade_pct"], ex["init_stop_atr"], o["atr"], max_frac)
            if 10.0 <= stake <= st["cash"]:
                st["cash"] -= stake
                st["open"].append({"pair": o["pair"], "strategy": o["strat"],
                                   "entry_ts": str(int(row["timestamp"])), "entry_price": limit,
                                   "atr_pct": o["atr"], "stake": stake, "bars_held": 0,
                                   "peak_ret_pct": 0.0, "armed": False})
                actions.append(f"FILL  {o['pair']:12s} {o['strat']:10s} @ {limit:.6g}  stake={stake:.0f} USDC")
                # LIVE: hier war die Limit-Buy; Fill bestaetigen
            continue
        if now < o["expire_ts"]:
            still_pending.append(o)
        else:
            actions.append(f"CANCEL {o['pair']:12s} {o['strat']:10s} (nicht gefuellt)")
    st["pending"] = still_pending

    # 3) neue Signale der zuletzt geschlossenen Kerze als Limit-Orders
    for pair, row in latest.items():
        for s in LIVE_STRATS:
            col = f"signal_{s}"
            if col in row and bool(row[col]) and float(row.get("atr_pct", 0)) > 0:
                expire = int(row["timestamp"]) + FILL_WINDOW * 3600_000
                st["pending"].append({"pair": pair, "strat": s, "limit": float(row["close"]),
                                      "atr": float(row["atr_pct"]), "sig_ts": int(row["timestamp"]),
                                      "expire_ts": expire})
                actions.append(f"NEW   {pair:12s} {s:10s} Limit-Buy @ {float(row['close']):.6g}")
                # LIVE: hier create_limit_buy_order senden

    json.dump(st, open(state_path, "w"), indent=2)
    equity = st["cash"] + sum(p["stake"] for p in st["open"])
    print("=" * 72)
    print(f"ENDVISION-1h LIVE-TICK  [{mode}]  {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"offen={len(st['open'])}/{max_open}  pending={len(st['pending'])}  "
          f"cash={st['cash']:.0f}  equity~{equity:.0f} USDC")
    print("=" * 72)
    if actions:
        for a in actions:
            print("  " + a)
    else:
        print("  keine Aktion diese Stunde.")
    if not live_on:
        print("\n  PAPER-Modus: keine echten Orders gesendet. State:", state_path)


# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--config", default="config/config_1h.yaml")
    ap.add_argument("--candles", default="data/candles")
    ap.add_argument("--state", default="logs/live_state_1h.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    if args.replay:
        print("\n### Realistische Kosten (Maker-Entry 0.10% + Taker-Exit 0.15%)")
        res = replay(cfg, args.candles, COST_ENTRY_PCT, COST_EXIT_PCT)
        if res is None:
            print("Keine Feature-Daten gefunden."); return
        report_replay(res, "realistisch")
        print("\n### Optimistisch (beide Seiten Maker, 0.20%) — Obergrenze")
        res2 = replay(cfg, args.candles, 0.10, 0.10)
        report_replay(res2, "optimistisch")
        print("\nLesart: Die 'realistisch'-Zeile ist die ehrliche Erwartung. Positiver")
        print("Endstand + ertraeglicher Max-Drawdown = der Bot darf live (erst Paper).")
    elif args.live:
        os.makedirs(os.path.dirname(args.state) or ".", exist_ok=True)
        live_tick(cfg, args.candles, args.state)
    else:
        print("Modus waehlen: --replay  oder  --live")


if __name__ == "__main__":
    main()
