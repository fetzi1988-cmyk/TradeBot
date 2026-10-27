# TradeBot — Playbook & gewonnene Erfahrungen

Festgehalten am 2026-10-27. Das ist der Wissensstand aus der kompletten
Untersuchung — was getestet wurde, was scheitert, was traegt, und die
Mess-Methodik, damit jede kuenftige Idee in Minuten gegen dieselbe strenge
Latte geprueft werden kann.

## Die Kernfrage
Kann ein systematischer Bot auf Krypto-Spot zuverlaessig Prozente machen mit
wenig Minus? Ausgangsthese (Fetzi): Indikatoren zeigen *wann*, nicht *wohin*.

## Was GETESTET und WIDERLEGT wurde (alles ehrlich gemessen)
1. **Direktionale Indikator-Signale (rsi_cross, pullback) auf 5m und 1h.**
   Rohkante ~+0.04-0.06 % ueber der Marktdrift -> kleiner als Gebuehren.
   Look-ahead ausgeschlossen (nur .shift/rolling). Signal sagt die Richtung
   NICHT vorher. Tot.
2. **Maker-Limit-Entry + ATR-Trail (1h).** Mit korrekten Stops: touch -0.14 %,
   close -0.64 % netto. Tot. (Das fruehe +0.27 %/+266 % war ein EINHEITEN-BUG:
   atr_pct ist ein BRUCH, wurde als Prozent gerechnet -> Stops 100x zu eng,
   ein Haar-Trigger-Scalper mit Stops unter dem Spread. Der integrity_check
   hat es gefangen. LEKTION: Einheiten immer messen.)
3. **Mean-Reversion / Market-Making-Scalp (1h & 5m), mit Regime-Filter.**
   ~-0.5 %/Trade, GEFILTERT = IMMER-AN = ZUFALL. Der Preis ist bei 1h/5m ein
   Random Walk; nichts zum Mean-Reverten. Tot.
4. **Trend-Reiter AUFWIND (EMA-Cross + Chandelier, 1h).** -75 % ueber 2 Jahre.
   Auf 1h in der Alt-Suppe zu Tode gewhipsawt. Tot.
5. **Alt-Korb halten (301 Coins, 4 Jahre Tagesdaten).** -78 %, 93 % Drawdown.
   Die meisten Alts bluten gegen BTC. Das "taegliche Gruen" ist eine
   Bullenmarkt-/Recency-Illusion; ueber den Zyklus verliert es.

## Was TRAEGT (das einzige mit echter, robuster positiver Erwartung)
**REGIME-HALTEN auf den Majors.** Gleichgewicht-Korb BTC/ETH/SOL/BNB/XRP,
gehalten solange BTC ueber seinem 150-Tage-Schnitt liegt, sonst komplett USDC.
Backtest 2022-2026 (saubere Tagesdaten, regime_hold.py):
- Majors halten: +239 %, Max-Drawdown 60 %.
- **Majors + Regime: +229 %, Max-Drawdown 35 %** (fast die ganze Rendite,
  Drawdown fast halbiert).
- BTC pur + Regime: +219 %, Max-Drawdown 25,6 % (niedrigster DD, simpelster).

Ehrliche Einordnung: Das ist **Beta auf die Majors + ein Trend-Filter**, keine
Alpha-Zauberei. "Halte die Ueberlebenden, tritt in tiefen Baerenphasen beiseite."
Der Filter reagiert verzoegert -> 25-35 % Restdrawdown, nicht null. Vier Jahre
Krypto waren ein starker Zyklus; Zukunft kann anders laufen. Reines Spot,
long-only, wenige Umschichtungen (kaum Gebuehren/Steuer).

## Der laufende Bot
- `core/regime_live.py` — Tages-Tick: BTC-Regime pruefen, Majors-Korb halten
  oder USDC. Paper-Default; echte Orders hinter live.enabled.
- `core/regime_hold.py` — der Backtest/Beweis (argv: dir tf ma-tage [whitelist]).
- `scripts/collect_daily.py` — tiefe Tagesdaten (argv: ziel [whitelist]).
- `scripts/daily_regime.sh` — Cron-Wrapper (collect + live, taeglich, Paper).

## GEPARKT (braucht Futures, vom Nutzer vorerst abgelehnt)
Funding-Rate-Arbitrage (`core/backtest_funding.py`, `scripts/collect_funding.py`)
— markt-neutral, Spot long + Perp short. Einsatzbereit, falls spaeter ein
USDS-M-Futures-Konto dazukommt. (Binance-Futures ist fuer DE-Retail unter MiCA
aktuell vermutlich gesperrt — vor Einsatz pruefen.)

## Mess-Methodik (die Latte fuer JEDE neue Idee)
- net-mean UND ohneT5 (Mittel ohne Top-5) beide positiv -> nicht tail-abhaengig.
- Monats-Split -> ueber Phasen stabil, nicht eine Gluecksphase.
- ZUFALLS-Kontrolle -> schlaegt die Idee zufaelliges Timing ueberhaupt?
- Kosten ehrlich (Maker 0.10 %/Leg, Taker 0.15 %, Round-Trip min. 0.20 %).
- Einheiten messen (integrity_check): Bruch vs. Prozent, Stop-Distanz vs. Bar.
- Voller Zyklus, nicht nur das Bullen-Fenster. Saubere, tiefe, aligned Daten.
- Kein Look-ahead (nur .shift/rolling/Vortag). Pessimistische Fills.
