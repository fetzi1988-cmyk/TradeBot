# ENDVISION-1h

Eigenstaendiges 1-Stunden-Trading-System, parallel zum 5m-ENDVISION auf panzer-vps-1.
Kein momentum (als Muenzwurf-Einstieg belegt und deaktiviert).

## Warum 1h
Zwei Monate Live-Auswertung des 5m-Systems (Okt 2026) zeigten: das Defizit ist
strukturell, nicht phasenabhaengig. Jede Strategie liegt net ~ -0.23% pro Trade,
also ziemlich genau eine Kostenrunde (0.235%). Die Einstiege sind brutto nahe
break-even; die Kosten kippen sie ins Minus, weil die 5m-Bewegungen zu klein sind.
Groesserer Zeitrahmen ist der einzige Hebel, der das Verhaeltnis Bewegung/Kosten
an der Wurzel aendert.

## Was hier anders ist als im 5m-System
- Kosten (0.235% Round-Trip) sind in der Engine eingebaut und werden vom PnL
  abgezogen. Im 5m-System fehlte das komplett (jeder PnL war brutto).
- ATR-Exit neu fuer 1h (die 5m-Schwellen waren auf 1h alle falsch).
- Backtest zuerst: dieselbe Engine beweist auf der Historie, ob 1h traegt,
  bevor live gehandelt wird.

## Dateien
- `config/config_1h.yaml` — Strategien, ATR-Exit-Parameter, Kosten, Risiko
- `core/portfolio_1h.py`   — Trade-Lebenszyklus, ATR-Trail, Kosten, Positionsgroesse
- `core/backtest_1h.py`    — Backtest ueber *_1h_features.csv (+ Monats-Split)
- `scripts/collect_1h_history.py` — tiefe 1h-Historie von Binance sammeln

## Ablauf
1. Tiefe Daten sammeln (auf dem Server, braucht Binance-Zugang + feature_engine):
   ```
   python3 scripts/collect_1h_history.py 3000
   python3 scripts/build_features_tf.py --tf 1h   # aus dem 5m-trading-system
   ```
2. *_1h_features.csv nach data/candles/ legen
3. Backtest:
   ```
   python3 core/backtest_1h.py config/config_1h.yaml
   ```

## Methodische Regeln (aus dem 5m-Projekt uebernommen)
- Nur harte Gates, keine Score-Modelle
- Min. 30-50 Trades vor einem Urteil
- Nur disjunkte Gruppen zaehlen als Robustheitsbeleg; Monats-Split Pflicht
- Einzelner Peak mit schlechteren Nachbarn = Ausschlag, nicht Plateau
- Backup/Commit vor jeder Aenderung
