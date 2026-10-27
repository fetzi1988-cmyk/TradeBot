from __future__ import annotations
import json, numpy as np, pandas as pd, pickle
from pathlib import Path
from datetime import datetime, timezone
from hmmlearn.hmm import GaussianHMM

def _log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}")

MODEL_PATH  = Path("data/state/hmm_model.pkl")
REGIME_PATH = Path("data/state/hmm_regime.json")
N_STATES    = 5
BASKET      = ["data/candles/BTC_USDC_5m.csv",
               "data/candles/ETH_USDC_5m.csv",
               "data/candles/SOL_USDC_5m.csv"]

def _features(df):
    c   = df["close"]
    ret = np.log(c / c.shift(1)).fillna(0.0)
    vol = ret.rolling(20).std().fillna(0.0)
    vr  = (df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)).fillna(1.0)
    X   = np.column_stack([ret.values, vol.values, vr.values])
    mu, sd = X.mean(0), X.std(0)
    sd[sd==0] = 1.0
    return (X - mu) / sd

def _basket_features():
    Xs, used = [], []
    for p in BASKET:
        if not Path(p).exists(): continue
        try:
            df = pd.read_csv(p)
            if len(df) < 100: continue
            Xs.append(_features(df)); used.append(Path(p).stem)
        except: continue
    if not Xs: return None, []
    n = min(x.shape[0] for x in Xs)
    return np.mean(np.stack([x[-n:] for x in Xs], 0), 0), used

def _market_corr(lookback=100):
    rets = []
    for p in BASKET:
        if not Path(p).exists(): continue
        try:
            df = pd.read_csv(p, usecols=["close"]).tail(lookback+1)
            if len(df) < 20: continue
            rets.append(df["close"].pct_change().dropna().values[-lookback:])
        except: continue
    if len(rets) < 2: return 0.5
    n = min(len(r) for r in rets)
    M = np.corrcoef([r[-n:] for r in rets])
    vals = [M[i,j] for i in range(len(M)) for j in range(i+1,len(M)) if not np.isnan(M[i,j])]
    return float(np.mean(np.abs(vals))) if vals else 0.5

def _label(model):
    order = np.argsort(model.means_[:,0])
    n = len(order)
    return {int(order[0]):"crash", int(order[1]):"distribution",
            int(order[n//2]):"accumulation", int(order[n-2]):"exhaustion",
            int(order[n-1]):"trend"}

def train(candles_path="data/candles/BTC_USDC_5m.csv"):
    try:
        X, used = _basket_features()
        if X is None:
            df = pd.read_csv(candles_path)
            if len(df) < 100: return False
            X = _features(df); used = ["BTC_only"]
        mc = _market_corr()
        m  = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                         n_iter=1000, tol=1e-6, random_state=42)
        m.fit(X)
        lb = _label(m)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump({"model":m,"labels":lb,"market_corr":mc}, open(MODEL_PATH,"wb"))
        _log(f"hmm_train ok basket={used} market_corr={mc:.2f} labels={lb}")
        return True
    except Exception as e:
        _log(f"hmm_train ERR {e}"); return False

def _real_return(lookback=20):
    """Echte (nicht standardisierte) mittlere Log-Rendite/Kerze des Baskets."""
    rets = []
    for p in BASKET:
        if not Path(p).exists(): continue
        try:
            df = pd.read_csv(p, usecols=["close"]).tail(lookback+1)
            if len(df) < 2: continue
            r = np.log(df["close"] / df["close"].shift(1)).dropna()
            if len(r): rets.append(float(r.mean()))
        except: continue
    return float(np.mean(rets)) if rets else 0.0

def predict(candles_path="data/candles/BTC_USDC_5m.csv"):
    default = {"regime":"neutral","confidence":0.0,"valid":False,
               "market_corr":0.5,"gate_threshold_hard":0.65,
               "gate_threshold_exhaustion":0.90}
    # Nur neu trainieren wenn Modell fehlt oder älter als 24h
    import time
    needs_train = not MODEL_PATH.exists()
    if not needs_train:
        age_h = (time.time() - MODEL_PATH.stat().st_mtime) / 3600
        needs_train = age_h > 24
    if needs_train:
        if not train(candles_path): return default
    try:
        d  = pickle.load(open(MODEL_PATH,"rb"))
        m, lb, mc = d["model"], d["labels"], d.get("market_corr",0.5)
        X, _ = _basket_features()
        if X is None:
            df = pd.read_csv(candles_path)
            if len(df) < 50: return default
            X = _features(df)
        _, post = m.score_samples(X)
        lp   = post[-1]
        best = int(np.argmax(lp))
        reg  = lb.get(best,"neutral")
        conf = float(lp[best])
        probs = {lb.get(i,f"s{i}"):float(p) for i,p in enumerate(lp)}
        # Auto-Adjust: hohe Korrelation = Gate aggressiver
        if mc >= 0.70:   gh, ge = 0.80, 0.92
        elif mc <= 0.40: gh, ge = 0.90, 0.97
        else:
            t = (mc-0.40)/0.30
            gh = round(0.80 - t*0.25, 3)
            ge = round(0.97 - t*0.17, 3)
        real_ret = _real_return(lookback=20)
        raw_reg  = reg
        if reg == "crash" and real_ret > -0.0008:
            reg = "neutral"
        elif reg == "distribution" and real_ret > -0.0003:
            reg = "neutral"
        if reg != raw_reg:
            _log(f"hmm_override raw={raw_reg}->{reg} real_ret={real_ret:.5f}")
        result = {"regime":reg,"confidence":conf,"probabilities":probs,
                  "raw_regime":raw_reg,"real_ret":round(real_ret,6),
                  "valid":True,"market_corr":mc,
                  "gate_threshold_hard":gh,"gate_threshold_exhaustion":ge,
                  "timestamp_utc":datetime.now(timezone.utc).isoformat()}
        REGIME_PATH.write_text(json.dumps(result, indent=2))
        _log(f"hmm_predict regime={reg} conf={conf:.1%} corr={mc:.2f} gate_hard={gh}")
        return result
    except Exception as e:
        _log(f"hmm_predict ERR {e}"); return default

def get_allowed_strategies(regime, confidence):
    if confidence < 0.45:
        return {"trend","breakout","momentum","pullback",
                "reversal","range","squeeze_sniper","trend_rider"}
    mapping = {
        "trend":        {"trend","breakout","momentum","trend_rider","pullback","squeeze_sniper"},
        "accumulation": {"pullback","reversal","range","squeeze_sniper","momentum"},
        "exhaustion":   {"reversal","range","pullback"},
        "distribution": {"reversal","range","squeeze_sniper","trend_rider"},
        "crash":        {"reversal","range","squeeze_sniper"},
        "neutral":      {"trend","breakout","momentum","pullback",
                         "reversal","range","squeeze_sniper","trend_rider"},
    }
    return mapping.get(regime, mapping["neutral"])
