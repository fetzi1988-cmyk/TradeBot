#!/usr/bin/env python3
"""
Erzeugt *_15m_features.csv aus *_15m.csv, parallel zum Live-5m-Pfad.
Faesst den laufenden Betrieb NICHT an - schreibt nur neue Dateien.

Zweck: Backtests auf 15m ermoeglichen. Begruendung (30.07.2026): sauberer
Vergleich bei gleichem Fenster und gleichen Daten (5m selbst zu 15m aggregiert,
n=54 Pairs) ergab Faktor 1.824 fuer die typische Kerzenbewegung - passt zur
Wurzel-Theorie (sqrt(3)=1.732). Damit steigt das Verhaeltnis MFE zu 0.15%-Fee
etwa von 3.0x auf 5.4x (breakout) bzw. 1.9x auf 3.5x (momentum).
ACHTUNG: der Bar-Range-Faktor 2.906 aus derselben Messung ist ueberhoeht
(Median einer schiefen Verteilung skaliert nicht linear) - nicht verwenden.
Zusatznutzen: 15m-Dateien haben 1000 Zeilen = ~10.4 Tage, die 5m-Dateien nur
420 = 1.45 Tage. Siebenfache Backtest-Abdeckung.
OFFENER PUNKT: build_features() enthaelt den CVD-Merge mit 5m-Bucket-Ausrichtung
(300_000 ms). Auf 15m-Daten mappt das leicht versetzt. Fuer erste Backtests
akzeptabel, CVD-abhaengige Felder dort aber mit Vorsicht lesen.
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.feature_engine import FeatureEngine

CANDLES = Path("data/candles")

def main(limit=None):
    fe = FeatureEngine()
    files = sorted(CANDLES.glob("*_15m.csv"))
    if limit:
        files = files[:limit]
    ok = 0
    fehler = []
    for fp in files:
        pair_key = fp.name.replace("_15m.csv", "")
        try:
            df = pd.read_csv(fp)
            if len(df) < 250:
                fehler.append((pair_key, f"nur {len(df)} Zeilen"))
                continue
            feats = fe.build_features(df)

            # signal_rsi_cross wird im Live-System erst in process_pair()
            # gesetzt, nicht in build_features() - hier nachgezogen, damit
            # alle neun Strategien testbar sind.
            # WICHTIG: OHNE den ATR-Deckel <=0.0030. Der wurde auf 5m-Daten
            # kalibriert; ATR-Werte sind auf 15m etwa 1.8x groesser, der Deckel
            # wuerde dort praktisch alles blockieren. Muss fuer 15m neu
            # abgeleitet werden. Gleiches gilt fuer alle anderen ATR-Schwellen
            # im System: pullback TP atr*130, momentum TP atr*200,
            # dynamic_stop atr*150, atr_stake_reference 0.0030.
            _cu = (feats["rsi_6"] > feats["rsi_14"]) & (feats["rsi_6"].shift(1) <= feats["rsi_14"].shift(1))
            _ab = feats["rsi_6"] > feats["rsi_14"]
            _s3 = _cu.shift(2).fillna(False) & _ab.shift(1).fillna(False) & _ab
            feats["signal_rsi_cross"] = (_s3 & (feats["rsi_6"] <= 50)).fillna(False)

            dst = CANDLES / f"{pair_key}_15m_features.csv"
            feats.to_csv(dst, index=False)
            ok += 1
        except Exception as e:
            fehler.append((pair_key, f"{type(e).__name__}: {e}"))
    print(f"Erzeugt: {ok} von {len(files)}")
    if fehler:
        print(f"Fehler bei {len(fehler)}:")
        for p, e in fehler[:10]:
            print(f"  {p}: {e}")
    return ok, fehler

if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(lim)
