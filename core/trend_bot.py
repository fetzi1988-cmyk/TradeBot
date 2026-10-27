"""
AUFWIND — Trend-Reiter fuer Spot (Long-only).

Ziel in Fetzis Worten: den richtigen EINSTIEG fuer einen Halte-Trade treffen,
HALTEN solange es laeuft, am richtigen Punkt VERKAUFEN. Also kein Rein-Raus-
Scalp, sondern den Aufwaertstrend reiten.

EINSTIEG (der "richtige Moment zu kaufen"):
  - uebergeordneter Aufwaertstrend: EMA50 > EMA200
  - Momentum dreht nach oben: close kreuzt EMA20 von unten nach oben
  -> kaufen (Spot), wenn beides zutrifft und ATR vorhanden.

HALTEN & VERKAUFEN (den Zug reiten, am Trendbruch raus):
  - Chandelier-Stop: hoechstes Hoch seit Einstieg minus K_ATR * ATR(Einstieg).
    Zieht WEIT nach (trend-skaliert, kein enges Scalp-Stop) -> Gewinner laufen.
  - zusaetzlich harter Trendbruch: close < EMA50 -> raus.
  - Exit zum Stop-Preis (wenn getroffen) bzw. zum close (bei EMA50-Bruch).

Alles in PREIS-Einheiten gerechnet (ATR in Preis, Stops als Preis) — keine
Bruch/Prozent-Verwechslung moeglich. Kosten: Round-Trip 0.20% (Spot-Taker beide
Seiten, klein gegen Trendbewegungen). Positionsgroesse = Equity/max_positions.

Zwei Modi:
  --replay : spielt AUFWIND ueber die ganze Historie (Equity-Kurve, Drawdown,
             Monate) — EINMALIGE Selbstpruefung der neuen Logik.
  --live   : derselbe Kopf als Tick, liest die zuletzt geschlossene Kerze je
             Coin, trifft Kauf-/Halte-/Verkaufs-Entscheidungen, State in JSON.
             Paper-Default (nur Log); live.enabled=true waere echte Orders.

Aufruf auf dem Server:
  .venv/bin/python3 /root/TradeBot/core/trend_bot.py --replay --candles data/candles --tf 1h
  .venv/bin/python3 /root/TradeBot/core/trend_bot.py --live   --candles data/candles --tf 1h
"""
from __future__ import annotations
import sys, os, glob, json, argparse, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

# --- Parameter (bewusst robust, nicht in-sample optimiert) ---
EMA_FAST, EMA_MID, EMA_SLOW = 20, 50, 200
ATR_LEN = 14
K_ATR = 3.0                 # Chandelier-Weite (trend-skaliert)
COST_RT = 0.20              # Round-Trip in % (Spot-Taker beide Seiten)
MAX_POSITIONS = 8
START_CAPITAL = 1000.0


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().values


def atr_price(high, low, close, n=ATR_LEN):
    pc = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    return pd.Series(tr).rolling(n).mean().values


def load(candles_dir, tf):
    pat = os.path.join(candles_dir, f"*_{tf}_features.csv")
    files = sorted(glob.glob(pat)) or sorted(
        f for f in glob.glob(os.path.join(candles_dir, f"*_{tf}.csv"))
        if not f.endswith(f"_{tf}_features.csv"))
    P = {}
    for fp in files:
        pair = os.path.basename(fp).replace(f"_{tf}_features.csv", "").replace(f"_{tf}.csv", "")
        try:
            d = pd.read_csv(fp, usecols=lambda c: c in ("timestamp", "open", "high", "low", "close"))
        except Exception:
            continue
        if "close" not in d.columns or len(d) < EMA_SLOW + 30:
            continue
        for c in ("high", "low", "close"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.sort_values("timestamp").reset_index(drop=True)
        ts = d["timestamp"].astype("int64").values
        if len(ts) and np.median(ts) < 1e12:
            ts = ts * 1000
        close = d["close"].values.astype(float)
        high = d["high"].values.astype(float); low = d["low"].values.astype(float)
        ef, em, es = ema(close, EMA_FAST), ema(close, EMA_MID), ema(close, EMA_SLOW)
        at = atr_price(high, low, close)
        # Einstiegssignal je Bar (nur Vergangenheit): Trend auf + close kreuzt EMA20 hoch
        entry = np.zeros(len(close), bool)
        for i in range(1, len(close)):
            if (em[i] > es[i]) and (close[i] > ef[i]) and (close[i-1] <= ef[i-1]) and at[i] > 0:
                entry[i] = True
        P[pair] = dict(ts=ts, idx={int(t): k for k, t in enumerate(ts)},
                       close=close, high=high, low=low, ema_mid=em, atr=at,
                       entry=entry, n=len(close))
    return P


# --------------------------------------------------------------------------- #
#  REPLAY — Portfolio ueber die ganze Historie
# --------------------------------------------------------------------------- #
def replay(P):
    timeline = sorted({int(t) for p in P.values() for t in p["ts"]})
    cash = START_CAPITAL
    pos = {}   # pair -> dict(entry_price, stake, hh, stop, atr_entry)
    closed = []
    eqc = []
    peak = START_CAPITAL; maxdd = 0.0

    for t in timeline:
        # 1) offene Positionen managen
        for pair in list(pos.keys()):
            pd_ = P[pair]; i = pd_["idx"].get(t)
            if i is None:
                continue
            ps = pos[pair]
            ps["hh"] = max(ps["hh"], pd_["high"][i])
            ps["stop"] = max(ps["stop"], ps["hh"] - K_ATR * ps["atr_entry"])
            exit_price = None; reason = None
            if pd_["low"][i] <= ps["stop"]:
                exit_price = ps["stop"]; reason = "chandelier"
            elif pd_["close"][i] < pd_["ema_mid"][i]:
                exit_price = pd_["close"][i]; reason = "ema50_break"
            if exit_price is not None:
                gross = (exit_price / ps["entry_price"] - 1) * 100
                net = gross - COST_RT
                cash += ps["stake"] * (1 + net / 100)
                closed.append(dict(pair=pair, entry_ts=ps["entry_ts"], exit_ts=t,
                                   net_pct=net, reason=reason,
                                   pnl=ps["stake"] * net / 100))
                del pos[pair]

        # 2) neue Einstiege (freie Slots, aeltere Signale zufrieden)
        for pair, pd_ in P.items():
            if pair in pos or len(pos) >= MAX_POSITIONS:
                continue
            i = pd_["idx"].get(t)
            if i is None or not pd_["entry"][i]:
                continue
            equity = cash + sum(ps["stake"] for ps in pos.values())
            stake = min(equity / MAX_POSITIONS, cash)
            if stake < 10:
                continue
            ep = pd_["close"][i]
            cash -= stake
            pos[pair] = dict(entry_price=ep, entry_ts=t, stake=stake, hh=pd_["high"][i],
                             stop=pd_["high"][i] - K_ATR * pd_["atr"][i], atr_entry=pd_["atr"][i])

        # 3) Equity (mark-to-market)
        mtm = 0.0
        for pair, ps in pos.items():
            j = P[pair]["idx"].get(t)
            px = P[pair]["close"][j] if j is not None else ps["entry_price"]
            mtm += ps["stake"] * (px / ps["entry_price"])
        equity = cash + mtm
        eqc.append((t, equity))
        peak = max(peak, equity); maxdd = max(maxdd, (peak - equity) / peak * 100)

    return dict(closed=pd.DataFrame(closed), eq=eqc, maxdd=maxdd)


def report_replay(res):
    c = res["closed"]; eq = res["eq"]
    end = eq[-1][1] if eq else START_CAPITAL
    ret = (end / START_CAPITAL - 1) * 100
    days = (eq[-1][0] - eq[0][0]) / 86400000.0 if eq else 0
    ann = ((end / START_CAPITAL) ** (365.0 / days) - 1) * 100 if days > 0 else 0
    print("=" * 88)
    print(f"AUFWIND REPLAY  (Trend-Reiter, EMA50>EMA200 + EMA20-Cross, Chandelier {K_ATR}xATR)")
    print("=" * 88)
    print(f"  Start {START_CAPITAL:.0f} -> Ende {end:.0f} USDC  ({ret:+.1f}%, annualisiert ~{ann:+.0f}%)  "
          f"ueber {days:.0f} Tage")
    print(f"  Max-Drawdown = {res['maxdd']:.1f}%")
    if len(c):
        wr = (c["net_pct"] > 0).mean() * 100
        avg = c["net_pct"].mean()
        print(f"  Trades={len(c)}  Netto-WR={wr:.1f}%  mittl. Trade={avg:+.2f}%  Summe-PnL={c['pnl'].sum():+.0f} USDC")
        cc = c.copy(); cc["m"] = pd.to_datetime(cc["exit_ts"].astype("int64"), unit="ms", utc=True).dt.strftime("%Y-%m")
        mon = cc.groupby("m")["pnl"].sum()
        print("  Monats-PnL: " + "  ".join(f"{m}:{v:+.0f}" for m, v in mon.items()))
        print("  Exit-Gruende: " + "  ".join(f"{k}:{v}" for k, v in c['reason'].value_counts().items()))
    print("\n  Lesart: Trend-Reiter lebt von wenigen grossen Gewinnern (lange gehalten) und")
    print("  vielen kleinen Fehlstarts. Positiver Endstand = er faengt die Aufwaertstrends.")
    print("  Max-Drawdown = was im Seitwaerts-/Abwaertsmarkt an Whipsaw zurueckkommt (ehrlich).")


# --------------------------------------------------------------------------- #
#  LIVE — ein Tick (Paper-Default)
# --------------------------------------------------------------------------- #
def live_tick(P, cfg, state_path):
    live_on = bool(cfg.get("live", {}).get("enabled", False))
    st = {"pos": {}, "cash": START_CAPITAL}
    if os.path.exists(state_path):
        try: st = json.load(open(state_path))
        except Exception: pass
    actions = []
    # zuletzt GESCHLOSSENE Kerze je Coin = vorletzter Index
    last = {}
    for pair, pd_ in P.items():
        if pd_["n"] >= 2:
            last[pair] = pd_["n"] - 2

    # 1) Positionen managen
    for pair in list(st["pos"].keys()):
        if pair not in last:
            continue
        i = last[pair]; pd_ = P[pair]; ps = st["pos"][pair]
        ps["hh"] = max(ps["hh"], float(pd_["high"][i]))
        ps["stop"] = max(ps["stop"], ps["hh"] - K_ATR * ps["atr_entry"])
        ep = ps["entry_price"]; reason = None; xp = None
        if pd_["low"][i] <= ps["stop"]:
            xp = ps["stop"]; reason = "chandelier"
        elif pd_["close"][i] < pd_["ema_mid"][i]:
            xp = float(pd_["close"][i]); reason = "ema50_break"
        if xp is not None:
            net = (xp / ep - 1) * 100 - COST_RT
            st["cash"] += ps["stake"] * (1 + net / 100)
            actions.append(f"VERKAUF {pair:12s} {reason:12s} net={net:+.2f}%")
            del st["pos"][pair]

    # 2) neue Einstiege
    for pair, i in last.items():
        if pair in st["pos"] or len(st["pos"]) >= MAX_POSITIONS:
            continue
        pd_ = P[pair]
        if not pd_["entry"][i]:
            continue
        equity = st["cash"] + sum(p["stake"] for p in st["pos"].values())
        stake = min(equity / MAX_POSITIONS, st["cash"])
        if stake < 10:
            continue
        ep = float(pd_["close"][i])
        st["cash"] -= stake
        st["pos"][pair] = dict(entry_price=ep, stake=stake, hh=float(pd_["high"][i]),
                               stop=float(pd_["high"][i]) - K_ATR * float(pd_["atr"][i]),
                               atr_entry=float(pd_["atr"][i]))
        actions.append(f"KAUF    {pair:12s} @ {ep:.6g}  stake={stake:.0f} USDC  (Trend auf, EMA20-Cross)")

    json.dump(st, open(state_path, "w"), indent=2)
    equity = st["cash"] + sum(p["stake"] for p in st["pos"].values())
    print("=" * 76)
    print(f"AUFWIND LIVE-TICK  [{'LIVE' if live_on else 'PAPER'}]  {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"offen={len(st['pos'])}/{MAX_POSITIONS}  cash={st['cash']:.0f}  equity~{equity:.0f} USDC")
    print("=" * 76)
    for a in actions or ["  keine Aktion diese Kerze."]:
        print("  " + a if not a.startswith("  ") else a)
    if not live_on:
        print(f"\n  PAPER: keine echten Orders. State: {state_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--candles", default="data/candles")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--state", default="logs/aufwind_state.json")
    a = ap.parse_args()
    P = load(a.candles, a.tf)
    if not P:
        print(f"Keine {a.tf}-Daten in {a.candles}."); return
    if a.replay:
        report_replay(replay(P))
    elif a.live:
        os.makedirs(os.path.dirname(a.state) or ".", exist_ok=True)
        live_tick(P, {}, a.state)
    else:
        print("Modus waehlen: --replay oder --live")


if __name__ == "__main__":
    main()
