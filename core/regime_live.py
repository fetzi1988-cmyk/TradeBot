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
START_CAPITAL = float(os.getenv("REGIME_CAPITAL", "100"))   # Startkapital in USDC (klein; per REGIME_CAPITAL aenderbar, z.B. 50)
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

    term = []
    universe = select_universe(TOP_N)
    print(f"  Universum: Top {len(universe)} nach Volumen -> {', '.join(universe[:10])}{' ...' if len(universe) > 10 else ''}")
    prices = {k: last_close(k) for k in universe}
    avail = [k for k in universe if prices.get(k)]

    per = 0.0
    if risk_on and not st["invested"]:
        per = st["cash"] / len(avail)
        st["holdings"] = {k: per / prices[k] for k in avail}   # Einheiten
        st["entry"] = {k: prices[k] for k in avail}
        st["cash"] = 0.0; st["invested"] = True
        did = "buy"
        for k in avail:
            term.append(f"KAUF    {k:10s} fuer {per:.2f} USDC  @ {prices[k]:.6g}")
    elif (not risk_on) and st["invested"]:
        val = sum(st["holdings"].get(k, 0) * prices[k] for k in avail if prices.get(k))
        for k in list(st["holdings"]):
            term.append(f"VERKAUF {k:10s} @ {prices.get(k, 0):.6g}")
        st["cash"] = val; st["holdings"] = {}; st.pop("entry", None); st["invested"] = False
        did = "sell"
    else:
        did = "hold" if st["invested"] else "wait"
        term.append("HALTEN" if st["invested"] else "WARTEN in USDC")

    mtm = st["cash"] + sum(st["holdings"].get(k, 0) * prices[k] for k in avail if prices.get(k))
    json.dump(st, open(STATE, "w"), indent=2, default=float)
    for a in term:
        print("  " + a)
    print(f"\n  Status: {'Korb gehalten' if st['invested'] else 'alles USDC'}   equity~{mtm:.2f} USDC   State: {STATE}")
    if not cfg_live:
        print("  PAPER-Modus: keine echten Orders gesendet.")

    # --- Klare, deutsche Telegram-Meldung ---
    def de(x):
        return f"{x:,.0f}".replace(",", ".")   # 79970 -> 79.970
    held = [k.split("_")[0] for k in st["holdings"]]
    pnl = (mtm / START_CAPITAL - 1) * 100 if START_CAPITAL else 0
    L = [f"📊 Trading-Bot · {time.strftime('%d.%m.%Y %H:%M')} · {'ECHT-GELD' if cfg_live else 'PAPIER (Test)'}", ""]
    if risk_on:
        L.append("🟢 Markt-Ampel: GRÜN")
        L.append(f"BTC {de(bc)}$ liegt ÜBER dem 150-Tage-Schnitt ({de(ma)}$).")
        L.append("→ Wir sind im Markt und nehmen die Aufwärtsbewegung mit.")
    else:
        L.append("🔴 Markt-Ampel: ROT")
        L.append(f"BTC {de(bc)}$ liegt UNTER dem 150-Tage-Schnitt ({de(ma)}$).")
        L.append("→ Wir stehen in Sicherheit (USDC) und sitzen den Abschwung aus.")
    L.append("")
    if did == "buy":
        L.append(f"🟢 HEUTE GEKAUFT: Korb aus {len(avail)} Coins, je ~{per:.2f} USDC")
        L.append("→ " + ", ".join(k.split("_")[0] for k in avail))
    elif did == "sell":
        L.append("🔴 HEUTE VERKAUFT: Korb aufgelöst, alles zurück in USDC.")
    elif did == "hold":
        L.append("✅ HEUTE: keine Änderung — der Korb bleibt gehalten.")
    else:
        L.append("⏸️ HEUTE: keine Änderung — weiter in USDC, bis die Ampel grün wird.")
    L.append("")
    if st["invested"]:
        L.append(f"📦 Im Korb ({len(held)} Coins): {', '.join(held)}")
    else:
        L.append("📦 Aktuell: 100% USDC (nicht investiert)")
    L.append(f"💰 Depotwert: ~{mtm:.2f} USDC   (Start {START_CAPITAL:.0f} · {pnl:+.1f}%)")
    if not cfg_live:
        L += ["", "ℹ️ Testmodus — es wird noch kein echtes Geld bewegt."]
    _telegram("\n".join(L))


if __name__ == "__main__":
    os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
    main()
