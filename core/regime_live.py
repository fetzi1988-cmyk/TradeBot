"""
REGIME-LIVE — der laufende Bot (Spot, Long-only).

Die einzige Strategie, die jeden ehrlichen Test ueberlebt hat:
  - Halte einen Gleichgewicht-Korb der Majors (BTC/ETH/SOL/BNB/XRP), SOLANGE
    BTC ueber seinem 150-Tage-Schnitt liegt ("risk-on").
  - Faellt BTC unter den 150-Tage-Schnitt -> alles in USDC ("risk-off"),
    Crash aussitzen.
Backtest 2022-2026 (regime_hold.py): +229% bei 35% statt 60% Drawdown.

Ein Tages-Tick: liest die BTC-Tageshistorie (data/daily), bestimmt das Regime,
haelt/verkauft den Korb, Zustand in JSON. Paper-Default (nur Log); live.enabled
in der Config waere echte Spot-Orders. Einmal pro Tag laufen lassen (Cron), nach
einem frischen collect_daily.

Aufruf auf dem Server (taeglich):
  .venv/bin/python3 /root/TradeBot/scripts/collect_daily.py 400
  .venv/bin/python3 /root/TradeBot/core/regime_live.py
"""
from __future__ import annotations
import sys, os, glob, json, time
import urllib.request, urllib.parse
import numpy as np
import pandas as pd

DAILY_DIR = "data/daily"
REGIME_MA_DAYS = 150
TOP_N = int(os.getenv("REGIME_TOPN", "5"))   # Korb-Breite (Default 5; eng gewinnt messbar; per REGIME_TOPN aenderbar)
FALLBACK = ["BTC_USDC", "ETH_USDC", "SOL_USDC", "BNB_USDC", "XRP_USDC"]
START_CAPITAL = 1000.0
STATE = sys.argv[1] if len(sys.argv) > 1 else "logs/regime_live_state.json"
ENV_PATH = "/root/trading-system/.env"   # gleiche Telegram-Creds wie das alte System


def _telegram(text):
    """Best-effort Telegram-Nachricht an denselben Chat wie das alte System.
    Liest TELEGRAM_BOT_TOKEN/CHAT_ID aus /root/trading-system/.env oder der Umgebung."""
    tok = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    cid = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if (not tok or not cid) and os.path.exists(ENV_PATH):
        try:
            for ln in open(ENV_PATH):
                ln = ln.strip()
                if ln.startswith("TELEGRAM_BOT_TOKEN="):
                    tok = ln.split("=", 1)[1].strip().strip('"').strip("'")
                elif ln.startswith("TELEGRAM_CHAT_ID="):
                    cid = ln.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    if not tok or not cid:
        print("  [Telegram] keine Creds (.env) gefunden -> nur Log.")
        return
    try:
        url = f"https://api.telegram.org/bot{tok}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": cid, "text": text}).encode()
        with urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST"), timeout=20) as r:
            ok = json.loads(r.read().decode()).get("ok")
        print(f"  [Telegram] gesendet ok={ok}")
    except Exception as e:
        print(f"  [Telegram] Fehler: {e}")


def last_close(key):
    for fp in (os.path.join(DAILY_DIR, f"{key}_1d.csv"),):
        if os.path.exists(fp):
            d = pd.read_csv(fp)
            if "close" in d and len(d):
                return float(pd.to_numeric(d["close"], errors="coerce").dropna().iloc[-1])
    return None


def select_universe(n=TOP_N):
    """Top-n handelbare Coins nach Median-Quote-Volumen (close*volume), nur Coins
    mit genug Historie (>=REGIME_MA_DAYS). USDT-Taktgeber ausgeschlossen."""
    rows = []
    for fp in glob.glob(os.path.join(DAILY_DIR, "*_1d.csv")):
        key = os.path.basename(fp).replace("_1d.csv", "")
        if key.endswith("_USDT"):
            continue
        try:
            d = pd.read_csv(fp, usecols=lambda c: c in ("close", "volume"))
        except Exception:
            continue
        if "close" not in d or "volume" not in d or len(d) < REGIME_MA_DAYS:
            continue
        cc = pd.to_numeric(d["close"], errors="coerce").dropna()
        if len(cc) < REGIME_MA_DAYS or cc.min() <= 0 or cc.max() / cc.min() < 1.5:
            continue    # Stablecoin/Peg -> raus (kein Aufwaertspotenzial)
        qv = (cc * pd.to_numeric(d["volume"], errors="coerce")).median()
        if np.isfinite(qv) and qv > 0:
            rows.append((key, qv))
    rows.sort(key=lambda x: -x[1])
    sel = [k for k, _ in rows[:n]]
    return sel if sel else FALLBACK


def btc_regime():
    """risk_on, letzter BTC-Close, 150-Tage-MA — aus der tiefsten BTC-Datei."""
    cand = [os.path.join(DAILY_DIR, f"{k}_1d.csv") for k in ("BTC_USDT", "BTC_USDC")]
    fp = next((p for p in cand if os.path.exists(p)), None)
    if fp is None:
        return None, None, None, "keine BTC-Tagesdatei"
    d = pd.read_csv(fp)
    c = pd.to_numeric(d["close"], errors="coerce").dropna()
    if len(c) < REGIME_MA_DAYS + 1:
        return None, None, None, f"nur {len(c)} BTC-Tage (<{REGIME_MA_DAYS+1})"
    ma = c.rolling(REGIME_MA_DAYS).mean()
    # Entscheidung aus der letzten abgeschlossenen Tageskerze
    close = float(c.iloc[-1]); ma_now = float(ma.iloc[-1])
    return (close > ma_now), close, ma_now, os.path.basename(fp)


def main():
    cfg_live = False
    st = {"invested": False, "cash": START_CAPITAL, "holdings": {}}
    if os.path.exists(STATE):
        try: st = json.load(open(STATE))
        except Exception: pass

    risk_on, bc, ma, src = btc_regime()
    print("=" * 76)
    print(f"REGIME-LIVE  [{'LIVE' if cfg_live else 'PAPER'}]  {time.strftime('%Y-%m-%d %H:%M')}")
    if risk_on is None:
        print(f"  Abbruch: {src}.  Erst collect_daily laufen lassen.")
        print("=" * 76); return
    print(f"  BTC={bc:.0f}  150-Tage-MA={ma:.0f}  ->  Regime: {'RISK-ON (halten)' if risk_on else 'RISK-OFF (Cash)'}   [{src}]")
    print("=" * 76)

    actions = []
    universe = select_universe(TOP_N)
    print(f"  Universum: Top {len(universe)} nach Volumen -> {', '.join(universe[:10])}{' ...' if len(universe) > 10 else ''}")
    prices = {k: last_close(k) for k in universe}
    avail = [k for k in universe if prices.get(k)]

    if risk_on and not st["invested"]:
        # Korb kaufen: Cash gleichmaessig auf verfuegbare Majors
        per = st["cash"] / len(avail)
        st["holdings"] = {k: per / prices[k] for k in avail}   # Einheiten
        st["entry"] = {k: prices[k] for k in avail}
        st["cash"] = 0.0; st["invested"] = True
        for k in avail:
            actions.append(f"KAUF    {k:10s} fuer {per:.0f} USDC  @ {prices[k]:.6g}")
    elif (not risk_on) and st["invested"]:
        # alles zu USDC
        val = sum(st["holdings"].get(k, 0) * prices[k] for k in avail if prices.get(k))
        for k in avail:
            if st["holdings"].get(k):
                actions.append(f"VERKAUF {k:10s} @ {prices[k]:.6g}")
        st["cash"] = val; st["holdings"] = {}; st["invested"] = False
    else:
        actions.append("HALTEN" if st["invested"] else "WARTEN in USDC (Regime risk-off)")

    # Equity (mark-to-market)
    mtm = st["cash"] + sum(st["holdings"].get(k, 0) * prices[k] for k in avail if prices.get(k))
    json.dump(st, open(STATE, "w"), indent=2, default=float)
    for a in actions:
        print("  " + a)
    pos = "Korb gehalten" if st["invested"] else "alles USDC"
    print(f"\n  Status: {pos}   equity~{mtm:.0f} USDC   State: {STATE}")
    if not cfg_live:
        print("  PAPER-Modus: keine echten Orders gesendet.")

    # Telegram-Rueckmeldung (gleicher Chat wie das alte System)
    mode = "LIVE" if cfg_live else "PAPER"
    regime_txt = "RISK-ON (Majors halten)" if risk_on else "RISK-OFF (alles USDC)"
    msg = [f"🤖 Regime-Bot [{mode}]  {time.strftime('%Y-%m-%d')}",
           f"BTC {bc:.0f}  vs  150-Tage-MA {ma:.0f}",
           f"→ {regime_txt}"]
    msg += actions
    msg.append(f"Status: {pos}, equity ~{mtm:.0f} USDC")
    _telegram("\n".join(msg))


if __name__ == "__main__":
    os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
    main()
