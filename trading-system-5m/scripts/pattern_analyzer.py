"""
Pattern Analyzer — findet welche Indikatoren historisch
kurz VOR echten Breakouts/Reversals/Trends/Ranges wahr waren.
Lädt Daten direkt von Binance via ccxt.
"""
import sys
import time
import numpy as np
import pandas as pd
import datetime as dt

sys.path.insert(0, "/root/trading-system")
from app.core.feature_engine import FeatureEngine

# ─── Config ───────────────────────────────────────────────
COINS = ['0G/USDC', '1000CAT/USDC', '1000CHEEMS/USDC', '1000SATS/USDC', '1INCH/USDC', '1MBABYDOGE/USDC', '2Z/USDC', 'A/USDC', 'AAVE/USDC', 'ACH/USDC', 'ACT/USDC', 'ACX/USDC', 'ADA/USDC', 'AEVO/USDC', 'AIXBT/USDC', 'ALGO/USDC', 'ALLO/USDC', 'ALT/USDC', 'ANIME/USDC', 'APE/USDC', 'API3/USDC', 'APT/USDC', 'AR/USDC', 'ARB/USDC', 'ARKM/USDC', 'ASTER/USDC', 'AT/USDC', 'ATOM/USDC', 'AUCTION/USDC', 'AVAX/USDC', 'AVNT/USDC', 'AXS/USDC', 'BABY/USDC', 'BANANA/USDC', 'BANANAS31/USDC', 'BANK/USDC', 'BARD/USDC', 'BB/USDC', 'BCH/USDC', 'BEAMX/USDC', 'BERA/USDC', 'BFUSD/USDC', 'BIGTIME/USDC', 'BIO/USDC', 'BLUR/USDC', 'BMT/USDC', 'BNB/USDC', 'BOME/USDC', 'BONK/USDC', 'BREV/USDC', 'BROCCOLI714/USDC', 'BTC/USDC', 'C/USDC', 'CAKE/USDC', 'CATI/USDC', 'CETUS/USDC', 'CFG/USDC', 'CFX/USDC', 'CGPT/USDC', 'CHZ/USDC', 'CKB/USDC', 'COMP/USDC', 'COOKIE/USDC', 'COTI/USDC', 'COW/USDC', 'CRV/USDC', 'CVC/USDC', 'CVX/USDC', 'CYBER/USDC', 'DASH/USDC', 'DOGE/USDC', 'DOGS/USDC', 'DOLO/USDC', 'DOT/USDC', 'DYDX/USDC', 'DYM/USDC', 'EDEN/USDC', 'EGLD/USDC', 'EIGEN/USDC', 'ENA/USDC', 'ENJ/USDC', 'ENS/USDC', 'ENSO/USDC', 'EPIC/USDC', 'ERA/USDC', 'ESP/USDC', 'ETC/USDC', 'ETH/USDC', 'ETHFI/USDC', 'EUL/USDC', 'EUR/USDC', 'EURI/USDC', 'F/USDC', 'FDUSD/USDC', 'FET/USDC', 'FF/USDC', 'FIL/USDC', 'FLOKI/USDC', 'FLUX/USDC', 'FOGO/USDC', 'FORM/USDC', 'FUN/USDC', 'GALA/USDC', 'GIGGLE/USDC', 'GMT/USDC', 'GMX/USDC', 'GPS/USDC', 'GRT/USDC', 'GUN/USDC', 'HAEDAL/USDC', 'HBAR/USDC', 'HEI/USDC', 'HEMI/USDC', 'HIVE/USDC', 'HMSTR/USDC', 'HOLO/USDC', 'HOME/USDC', 'HUMA/USDC', 'HYPER/USDC', 'ICP/USDC', 'ILV/USDC', 'IMX/USDC', 'INIT/USDC', 'INJ/USDC', 'IO/USDC', 'IOTA/USDC', 'JTO/USDC', 'JUP/USDC', 'KAIA/USDC', 'KAITO/USDC', 'KAT/USDC', 'KERNEL/USDC', 'KITE/USDC', 'KMNO/USDC', 'LA/USDC', 'LAYER/USDC', 'LDO/USDC', 'LINEA/USDC', 'LINK/USDC', 'LISTA/USDC', 'LPT/USDC', 'LSK/USDC', 'LTC/USDC', 'LUNA/USDC', 'LUNC/USDC', 'MAGIC/USDC', 'MANTA/USDC', 'MANTRA/USDC', 'MASK/USDC', 'MAV/USDC', 'ME/USDC', 'MEME/USDC', 'MET/USDC', 'MINA/USDC', 'MIRA/USDC', 'MITO/USDC', 'MMT/USDC', 'MORPHO/USDC', 'MOVE/USDC', 'MUBARAK/USDC', 'NEAR/USDC', 'NEIRO/USDC', 'NEO/USDC', 'NEWT/USDC', 'NIGHT/USDC', 'NIL/USDC', 'NMR/USDC', 'NOM/USDC', 'NOT/USDC', 'NXPC/USDC', 'ONDO/USDC', 'ONT/USDC', 'OP/USDC', 'OPEN/USDC', 'OPN/USDC', 'ORCA/USDC', 'ORDI/USDC', 'OSMO/USDC', 'PARTI/USDC', 'PAXG/USDC', 'PENDLE/USDC', 'PENGU/USDC', 'PEOPLE/USDC', 'PEPE/USDC', 'PHA/USDC', 'PIXEL/USDC', 'PLUME/USDC', 'PNUT/USDC', 'POL/USDC', 'PROVE/USDC', 'PUMP/USDC', 'PUNDIX/USDC', 'PYTH/USDC', 'QNT/USDC', 'QTUM/USDC', 'RARE/USDC', 'RAY/USDC', 'RED/USDC', 'RENDER/USDC', 'RESOLV/USDC', 'REZ/USDC', 'ROBO/USDC', 'ROSE/USDC', 'RPL/USDC', 'RSR/USDC', 'RUNE/USDC', 'RVN/USDC', 'S/USDC', 'SAGA/USDC', 'SAHARA/USDC', 'SAND/USDC', 'SAPIEN/USDC', 'SEI/USDC', 'SENT/USDC', 'SHELL/USDC', 'SHIB/USDC', 'SIGN/USDC', 'SKL/USDC', 'SKY/USDC', 'SNX/USDC', 'SOL/USDC', 'SOLV/USDC', 'SOMI/USDC', 'SOPH/USDC', 'SPK/USDC', 'SSV/USDC', 'STEEM/USDC', 'STO/USDC', 'STRK/USDC', 'STX/USDC', 'SUI/USDC', 'SUSHI/USDC', 'SXT/USDC', 'SYN/USDC', 'SYRUP/USDC', 'T/USDC', 'TAO/USDC', 'THE/USDC', 'THETA/USDC', 'TIA/USDC', 'TLM/USDC', 'TNSR/USDC', 'TON/USDC', 'TOWNS/USDC', 'TRB/USDC', 'TREE/USDC', 'TRUMP/USDC', 'TRX/USDC', 'TST/USDC', 'TURBO/USDC', 'TURTLE/USDC', 'TUT/USDC', 'TWT/USDC', 'U/USDC', 'UMA/USDC', 'UNI/USDC', 'USD1/USDC', 'USDE/USDC', 'USUAL/USDC', 'VANA/USDC', 'VANRY/USDC', 'VELODROME/USDC', 'VET/USDC', 'VIRTUAL/USDC', 'W/USDC', 'WAL/USDC', 'WCT/USDC', 'WIF/USDC', 'WLD/USDC', 'WLFI/USDC', 'XAI/USDC', 'XAUT/USDC', 'XLM/USDC', 'XPL/USDC', 'XRP/USDC', 'XTZ/USDC', 'XVG/USDC', 'YB/USDC', 'YGG/USDC', 'ZAMA/USDC', 'ZBT/USDC', 'ZEC/USDC', 'ZEN/USDC', 'ZK/USDC', 'ZKC/USDC', 'ZKP/USDC', 'ZRO/USDC', '币安人生/USDC']
TIMEFRAME   = "5m"
LOOKBACK_DAYS = 180        # 90 Tage = ~25k Kerzen pro Coin
LOOKBACK_CANDLES = 5      # wie viele Kerzen VOR Event schauen

# ─── Pattern Definitionen ─────────────────────────────────
def find_breakouts(df):
    """Echter Breakout: +3% in 6 Kerzen (30min), hält danach"""
    events = []
    for i in range(6, len(df) - 6):
        move = (df["close"].iloc[i] - df["close"].iloc[i-6]) / df["close"].iloc[i-6]
        hold = (df["close"].iloc[i+6] - df["close"].iloc[i]) / df["close"].iloc[i]
        if move > 0.03 and hold > -0.01:
            events.append(i)
    return events

def find_reversals(df):
    """Echter Reversal: -5% in 12 Kerzen, dann +3% Recovery"""
    events = []
    for i in range(12, len(df) - 12):
        drop = (df["close"].iloc[i] - df["close"].iloc[i-12]) / df["close"].iloc[i-12]
        recover = (df["close"].iloc[i+12] - df["close"].iloc[i]) / df["close"].iloc[i]
        if drop < -0.05 and recover > 0.03:
            events.append(i)
    return events

def find_trends(df):
    """Echter Trend: +8% über 24 Kerzen (2h) gleichmäßig"""
    events = []
    for i in range(24, len(df) - 6):
        move = (df["close"].iloc[i] - df["close"].iloc[i-24]) / df["close"].iloc[i-24]
        # Gleichmäßigkeit: jede 6er-Gruppe sollte positiv sein
        chunks = [
            (df["close"].iloc[i-24+j*6+6] - df["close"].iloc[i-24+j*6]) /
             df["close"].iloc[i-24+j*6]
            for j in range(4)
        ]
        smooth = all(c > -0.005 for c in chunks)
        if move > 0.08 and smooth:
            events.append(i)
    return events

def find_momentum_setups(df):
    """Momentum: Vol-Spike > 3x dann Konsolidierung, Preis hält"""
    events = []
    for i in range(3, len(df) - 6):
        spike = df["volume"].iloc[i-1] / df["volume"].iloc[max(0,i-10):i-1].mean() if df["volume"].iloc[max(0,i-10):i-1].mean() > 0 else 0
        curr_vol = df["volume"].iloc[i] / df["volume"].iloc[max(0,i-10):i].mean() if df["volume"].iloc[max(0,i-10):i].mean() > 0 else 0
        price_hold = df["close"].iloc[i] > df["close"].iloc[i-1] * 0.99
        if spike > 3.0 and curr_vol < spike * 0.60 and price_hold:
            events.append(i)
    return events

def find_pullback_setups(df):
    """Pullback: Preis war stark, dann -2% bis -6% Rücksetzer, dreht wieder"""
    events = []
    for i in range(24, len(df) - 6):
        trend = (df["close"].iloc[i-24] < df["close"].iloc[i-6])
        drop = (df["close"].iloc[i] - df["close"].iloc[i-6]) / df["close"].iloc[i-6]
        recovery = (df["close"].iloc[i+6] - df["close"].iloc[i]) / df["close"].iloc[i]
        if trend and -0.06 < drop < -0.015 and recovery > 0.01:
            events.append(i)
    return events

def find_squeeze_setups(df):
    """Squeeze: BB eng für > 10 Kerzen, dann explosiver Ausbruch"""
    events = []
    close = df["close"]
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    bb_width = (2 * std / sma).fillna(0)
    bb_avg = bb_width.rolling(20).mean()
    
    for i in range(30, len(df) - 6):
        squeeze_count = sum(bb_width.iloc[i-12:i] < bb_avg.iloc[i-12:i] * 0.75)
        vol_avg = df["volume"].iloc[i-20:i].mean()
        vol_spike = df["volume"].iloc[i] / vol_avg if vol_avg > 0 else 0
        price_move = abs(df["close"].iloc[i] - df["close"].iloc[i-3]) / df["close"].iloc[i-3]
        if squeeze_count >= 8 and vol_spike > 2.5 and price_move > 0.02:
            events.append(i)
    return events

def find_ranges(df):
    """Echter Range: Preis bewegt sich < 1.5% über 24 Kerzen"""
    events = []
    for i in range(24, len(df) - 6):
        window = df["close"].iloc[i-24:i]
        spread = (window.max() - window.min()) / window.mean()
        if spread < 0.015:
            events.append(i)
    return events

# ─── Indikatoren die wir prüfen ───────────────────────────
INDICATORS = [
    "bb_squeeze", "adx_range", "adx_trend",
    "dif_cross_up_dea", "dif_above_dea",
    "above_vwap", "below_vwap",
    "obv_rising", "obv_up",
    "bullish_wick", "bearish_wick",
    "vol_spike_and_hold", "adx_squeeze_breakout",
    "stronger_than_btc", "vwap_reclaim",
    "swept_and_reclaimed", "l3_context",
]

NUMERIC_INDICATORS = {
    "rsi_14":     lambda v: v < 35,
    "rsi_14_mid": lambda v: 45 < v < 65,
    "rsi_14_ob":  lambda v: v > 65,
    "vol_ratio":  lambda v: v > 1.5,
    "vol_spike":  lambda v: v > 2.5,
    "atr_pct":    lambda v: v > 0.008,
    "bb_percent_low": lambda v: v < 0.20,
    "bb_percent_hi":  lambda v: v > 0.80,
    "adx_val":    lambda v: v > 25,
    "adx_low":    lambda v: v < 15,
    "stoch_os":   lambda v: v < 25,
    "stoch_ob":   lambda v: v > 75,
}

# ─── Daten laden ──────────────────────────────────────────
def load_binance(coin, days=90):
    import ccxt
    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601(
        (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days))
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(coin, TIMEFRAME, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break
        time.sleep(0.3)
    df = pd.DataFrame(all_ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["datetime"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.drop_duplicates("timestamp").reset_index(drop=True)

# ─── Analyse ──────────────────────────────────────────────
def analyze_pattern(features_df, event_indices, pattern_name, lookback=5):
    """Für jedes Event: welche Indikatoren waren in den X Kerzen vorher wahr?"""
    results = {ind: 0 for ind in INDICATORS}
    results.update({k: 0 for k in NUMERIC_INDICATORS})
    total = 0

    for idx in event_indices:
        start = max(0, idx - lookback)
        window = features_df.iloc[start:idx]
        if len(window) == 0:
            continue
        total += 1

        for ind in INDICATORS:
            if ind in features_df.columns:
                if window[ind].any():
                    results[ind] += 1

        for name, condition in NUMERIC_INDICATORS.items():
            col_map = {
                "rsi_14": "rsi_14", "rsi_14_mid": "rsi_14",
                "rsi_14_ob": "rsi_14", "vol_ratio": "vol_ratio",
                "vol_spike": "vol_ratio", "atr_pct": "atr_pct",
                "bb_percent_low": "bb_percent", "bb_percent_hi": "bb_percent",
                "adx_val": "adx", "adx_low": "adx",
                "stoch_os": "stoch_rsi_k", "stoch_ob": "stoch_rsi_k",
            }
            col = col_map.get(name)
            if col and col in features_df.columns:
                if window[col].apply(condition).any():
                    results[name] += 1

    if total == 0:
        return {}

    return {k: round(v / total * 100, 1) for k, v in results.items()}

# ─── Main ─────────────────────────────────────────────────
def find_bull_flag(df):
    """Bull Flag: starker Move + enge Konsolidierung + Fortsetzung"""
    events = []
    for i in range(12, len(df) - 6):
        # Fahnenstange: +4% in 6 Kerzen
        pole = (df["close"].iloc[i-6] - df["close"].iloc[i-12]) / df["close"].iloc[i-12]
        # Konsolidierung: max 2% Range in letzten 6 Kerzen
        window = df["close"].iloc[i-6:i]
        consolidation = (window.max() - window.min()) / window.mean()
        # Hält über 50% der Fahnenstange
        holds = df["close"].iloc[i] > df["close"].iloc[i-12] + (df["close"].iloc[i-6] - df["close"].iloc[i-12]) * 0.50
        if pole > 0.04 and consolidation < 0.02 and holds:
            events.append(i)
    return events

def find_wyckoff_spring(df):
    """Wyckoff Spring: Fake-Breakdown unter Support + sofortige Recovery"""
    events = []
    for i in range(20, len(df) - 6):
        support = df["low"].iloc[i-20:i-1].min()
        fake_break = df["low"].iloc[i] < support * 0.998
        recovery = df["close"].iloc[i] > support
        vol_avg = df["volume"].iloc[i-20:i].mean()
        vol_spike = df["volume"].iloc[i] > vol_avg * 1.5
        if fake_break and recovery and vol_spike:
            events.append(i)
    return events

def find_hidden_divergence(df):
    """Hidden Divergence: Preis höhere Tiefs, RSI auch → Trend intakt"""
    events = []
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))


def main():
    fe = FeatureEngine(candles_dir="/root/trading-system/data/candles")

    pattern_results = {
        "bull_flag":         {},
        "wyckoff_spring":    {},
        "hidden_divergence": {},
    }
    pattern_counts = {k: 0 for k in pattern_results}

    print(f"Lade {len(COINS)} Coins × {LOOKBACK_DAYS} Tage...\n")

    for coin in COINS:
        print(f"  → {coin}...", end=" ", flush=True)
        try:
            df = load_binance(coin, LOOKBACK_DAYS)
            features = fe.build_features(df)
            print(f"{len(df)} Kerzen", end=" | ", flush=True)

            patterns = {
            "bull_flag":         find_bull_flag(df),
            "wyckoff_spring":    find_wyckoff_spring(df),
            "hidden_divergence": find_hidden_divergence(df),
        }

            for pname, events in patterns.items():
                print(f"{pname}:{len(events)}", end=" ", flush=True)
                if not events:
                    continue
                res = analyze_pattern(features, events, pname, LOOKBACK_CANDLES)
                for k, v in res.items():
                    pattern_results[pname][k] = pattern_results[pname].get(k, 0) + v
                pattern_counts[pname] += 1

            print()
        except Exception as e:
            print(f"FEHLER: {e}")
        time.sleep(1)

    # ─── Ausgabe ──────────────────────────────────────────
    print("\n" + "="*65)
    print("ERGEBNIS — Indikator-Häufigkeit VOR echtem Pattern")
    print("(Prozent der Events wo Indikator in letzten 5 Kerzen wahr war)")
    print("="*65)

    for pname, results in pattern_results.items():
        if not results or pattern_counts[pname] == 0:
            continue
        avg = {k: round(v / pattern_counts[pname], 1) for k, v in results.items()}
        sorted_res = sorted(avg.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{'─'*65}")
        print(f"  {pname.upper()}")
        print(f"{'─'*65}")
        for ind, pct in sorted_res[:12]:  # Top 12
            bar = "█" * int(pct / 5)
            print(f"  {ind:<25} {pct:>6.1f}%  {bar}")

    print("\n" + "="*65)
    print("EMPFEHLUNG pro Pattern:")
    print("="*65)
    for pname, results in pattern_results.items():
        if not results or pattern_counts[pname] == 0:
            continue
        avg = {k: round(v / pattern_counts[pname], 1) for k, v in results.items()}
        top3 = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n  {pname.upper()}:")
        for i, (ind, pct) in enumerate(top3, 1):
            print(f"    #{i}: {ind} ({pct}%)")

if __name__ == "__main__":
    main()

