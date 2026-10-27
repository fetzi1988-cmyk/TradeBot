#!/usr/bin/env python3
"""TF-agnostischer Feature-Build. Erzeugt *_<tf>_features.csv aus *_<tf>.csv.
Basiert auf build_15m_features.py, nur der TF ist parametrisiert (--tf).
CVD-Felder auf >5m unvollstaendig (300_000-Bucket) - im Scan markiert."""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.feature_engine import FeatureEngine

CANDLES = Path("data/candles")

def main(tf="1h", limit=None):
    fe = FeatureEngine()
    files = sorted(CANDLES.glob(f"*_{tf}.csv"))
    files = [f for f in files if not f.name.endswith("_features.csv")]
    if limit: files = files[:limit]
    ok=0; fehler=[]
    for fp in files:
        pair_key = fp.name.replace(f"_{tf}.csv","")
        try:
            df = pd.read_csv(fp)
            if len(df) < 220:
                fehler.append((pair_key, f"nur {len(df)} Zeilen")); continue
            feats = fe.build_features(df)
            _cu = (feats["rsi_6"] > feats["rsi_14"]) & (feats["rsi_6"].shift(1) <= feats["rsi_14"].shift(1))
            _ab = feats["rsi_6"] > feats["rsi_14"]
            _s3 = _cu.shift(2).fillna(False) & _ab.shift(1).fillna(False) & _ab
            feats["signal_rsi_cross"] = (_s3 & (feats["rsi_6"] <= 50)).fillna(False)
            feats.to_csv(CANDLES / f"{pair_key}_{tf}_features.csv", index=False)
            ok+=1
        except Exception as e:
            fehler.append((pair_key, f"{type(e).__name__}: {e}"))
    print(f"[{tf}] Erzeugt: {ok} von {len(files)}")
    if fehler:
        print(f"[{tf}] Fehler bei {len(fehler)}:")
        for p,e in fehler[:8]: print(f"  {p}: {e}")
    return ok, fehler

if __name__ == "__main__":
    tf = "1h"; lim=None
    args=sys.argv[1:]
    for i,a in enumerate(args):
        if a=="--tf" and i+1<len(args): tf=args[i+1]
        if a=="--limit" and i+1<len(args): lim=int(args[i+1])
    main(tf, lim)
