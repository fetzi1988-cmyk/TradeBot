#!/usr/bin/env python3
"""
FUNDING-ARB — Backtest der delta-neutralen Funding-Arbitrage (Spot long + Perp short).

Position je Pair: Spot LONG und Perp SHORT in GLEICHER Notional N. Der Preis-PnL beider Beine
hebt sich auf (so modelliert: Preis-PnL des Hedges = 0). Uebrig bleibt das Funding:
  - Funding-Rate > 0 : Longs zahlen an Shorts -> unser Short KASSIERT  rate * N
  - Funding-Rate < 0 : Shorts zahlen an Longs -> unser Short ZAHLT     rate * N  (rate negativ!)
  Vorzeichen-Regel im Code: Funding-PnL = +rate * N  (negatives rate = Verlust). Kein Betrag, kein abs().

EINHEITEN (kritisch, an jeder Stelle gilt):
  - funding_rate  = BRUCH pro Funding-Intervall (Standard 8h).  0.0001 = 0.01 % pro 8h
                    = 0.0001 * 3 * 365 = 0.1095 = 10.95 % pro Jahr.
  - ALLE internen Rechnungen laufen in BRUECHEN (0.01 = 1 %). Prozent gibt es nur in der
    Ausgabe, und zwar ausschliesslich ueber pct(): Bruch -> "x.xx%" (Multiplikation mit 100).
  - Renditen sind RELATIV ZUR NOTIONAL EINES BEINS (N = 1.0). "Kapital" = Notional * CAPITAL_FACTOR
    (Spot-Bein voll bezahlt + Margin-Puffer fuers Perp-Short), siehe unten.
  - Annualisierung ZEITBASIERT (Summe / Tage * 365), nicht ueber "3 pro Tag" -> korrekt auch
    fuer Perps mit 4h-/1h-Funding-Intervall (Rate gilt je Abrechnung, wird je Abrechnung gebucht).

HALTELOGIK (kein Look-ahead):
  Signal fuer Abrechnung t nutzt NUR Raten bis t-1: Summe der Raten der letzten LOOKBACK_DAYS Tage
  (Bruch), hochgerechnet auf ein Jahr (sig_ann = Summe / LOOKBACK_DAYS * 365, Bruch pro Jahr).
    - flach  -> Einstieg, wenn sig_ann > ENTRY_ANN_MIN  (muss die Kosten amortisierbar machen)
    - gehalten -> bleibt, solange sig_ann > EXIT_ANN_MIN (= "Funding im Mittel positiv")
    - sonst Ausstieg (Auf-/Abbau kostet jeweils Taker + Slippage auf BEIDEN Beinen)
  Die Verzoegerung des Signals bedeutet: in Negativ-Phasen zahlt die Strategie, bis das Signal
  dreht. Das wird NICHT weggerechnet, sondern ausgewiesen (Abschnitt Negativ-Funding).
  Referenz "immer gehalten": ein Einstieg am Anfang, ein Ausstieg am Ende, nie umschalten.

Daten: data/funding/<KEY>_funding.csv (von scripts/collect_funding.py):
  timestamp (ms) | funding_rate (Bruch pro Intervall) | datetime (ISO-UTC)

Aufruf auf dem Server (im trading-system-Verzeichnis):
    cd /root/trading-system
    .venv/bin/python3 /root/TradeBot/core/backtest_funding.py                       # data/funding -> data/funding_backtest
    .venv/bin/python3 /root/TradeBot/core/backtest_funding.py data/funding out_dir  # eigene Ordner
"""
import sys, glob, os
import numpy as np
import pandas as pd

# ============================================================================================
# KONSTANTEN (alle in BRUECHEN, nicht Prozent; Kommentar nennt die Prozent-Lesart)
# ============================================================================================
# --- Handelskosten (Taker, je Seite, je Bein; Anteil der Notional des jeweiligen Beins) ------
SPOT_TAKER = 0.0010        # 0.10 % Binance-Spot-Taker (Standardtarif, ohne BNB-Rabatt)
FUT_TAKER = 0.0005         # 0.05 % Binance-USDS-M-Taker (Standardtarif)
SLIPPAGE_PER_LEG = 0.0002  # 0.02 % Slippage je Bein und Seite (pessimistische Annahme, NICHT gemessen;
                           # bei illiquiden Alts real eher mehr)
# Einstieg = Spot kaufen + Perp shorten; Ausstieg = Spot verkaufen + Perp zurueckkaufen.
ENTRY_COST = (SPOT_TAKER + FUT_TAKER) + 2 * SLIPPAGE_PER_LEG   # 0.15 % + 0.04 % = 0.19 % der Notional
EXIT_COST = ENTRY_COST                                         # 0.19 %; Round-Trip also 0.38 %

# --- Basis-/Rebalancing-Risiko (Abschlag, konservativ, laufende Haltezeit) -------------------
BASIS_HAIRCUT_ANN = 0.01   # 1 % pro Jahr der Notional, anteilig nur WAEHREND der Haltezeit.
                           # Platzhalter fuer Basis-Schwankung, Rebalancing, Spot-USDC vs. Perp-USDT-Mismatch.
                           # Annahme, nicht gemessen -> auf 0.0 setzen fuer reines Funding.

# --- Kapitalbedarf ---------------------------------------------------------------------------
PERP_MARGIN = 0.20         # 20 % der Notional als Margin-Puffer fuer den Perp-Short (Liquidationsschutz
                           # bei Rallyes). Kapital je Hedge = N * (1 + PERP_MARGIN) = N * 1.20.
CAPITAL_FACTOR = 1.0 + PERP_MARGIN
RISKFREE_ANN = 0.04        # 4 % pro Jahr: Vergleichszins (Stablecoin-Geldmarkt). Nur fuer die Lesart.

# --- Haltelogik ------------------------------------------------------------------------------
LOOKBACK_DAYS = 3          # Fenster fuer das Signal (Tage); Warmup = erste LOOKBACK_DAYS ohne Handel
ENTRY_ANN_MIN = 0.10       # Einstieg nur wenn hochgerechnete Funding-Rendite > 10 % p.a.
                           # (= 0.0001 pro 8h, der Binance-Basiswert; darunter frisst der Round-Trip das Funding)
EXIT_ANN_MIN = 0.0         # Ausstieg sobald das Signal <= 0 % p.a. ("im Mittel positiv" nicht mehr erfuellt)

# --- Daten-/Report-Parameter ------------------------------------------------------------------
MIN_EVENTS = 120           # weniger Abrechnungen im File -> Pair ueberspringen
SANITY_MAX_ABS_RATE = 0.05 # |Rate| > 0.05 (5 % pro Intervall) -> vermutlich Prozent statt Bruch -> Pair verwerfen
SANITY_MAX_MEDIAN_ABS = 0.002  # Median |Rate| > 0.002 (0.2 % pro 8h = ~219 % p.a. als TYPISCHER Wert) ist auf Binance
                           # unrealistisch (typisch 0.0001); Prozent-Werte wie "0.01" (= 0.01 %) rutschen sonst unter
                           # die Max-Schwelle durch -> ebenfalls verwerfen
FULL_MONTH_DAYS = 25       # Monate mit weniger Tagen gelten als Randmonate (nicht in der Stabilitaetszaehlung)
MIN_POS_MONTH_SHARE = 0.70 # Lesart: mind. 70 % der vollen Monate netto positiv
PRINT_TOP, PRINT_BOTTOM = 20, 10


# ============================================================================================
# Hilfsfunktionen
# ============================================================================================
def pct(x, nd=2):
    """Bruch -> Prozent-String. EINZIGE Stelle, an der mit 100 multipliziert wird."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x * 100.0:.{nd}f}%"


def selbsttest_einheiten():
    """Rechnet das Lehrbuch-Beispiel nach, bevor irgendetwas anderes passiert."""
    jahr_8h = 0.0001 * 3 * 365                     # 0.01 % pro 8h, 3 Abrechnungen/Tag, 365 Tage
    assert abs(jahr_8h - 0.1095) < 1e-12, "Einheiten-Selbsttest: 0.0001 pro 8h muss 10.95 % p.a. sein"
    assert abs(ENTRY_COST - 0.0019) < 1e-12 and abs(EXIT_COST - 0.0019) < 1e-12, \
        "Kosten-Selbsttest: Einstieg/Ausstieg muessen je 0.19 % sein (bei Default-Konstanten)"
    # Signal-Skalierung: 9 Abrechnungen a 0.0001 in 3 Tagen -> 0.0009/3*365 = 0.1095 p.a.
    sig = (9 * 0.0001) / 3 * 365
    assert abs(sig - 0.1095) < 1e-12, "Einheiten-Selbsttest: Signal-Hochrechnung falsch"


def load_funding(folder):
    """Liest alle *_funding.csv. Rueckgabe: dict KEY -> Series(rate, Index = UTC-Zeit)."""
    data = {}
    skipped = []
    for fp in sorted(glob.glob(os.path.join(folder, "*_funding.csv"))):
        name = os.path.basename(fp)
        if name.startswith("_"):
            continue
        key = name.replace("_funding.csv", "")
        try:
            df = pd.read_csv(fp)
            ts = pd.to_numeric(df["timestamp"], errors="coerce")
            fr = pd.to_numeric(df["funding_rate"], errors="coerce")   # BRUCH pro Intervall
        except Exception as e:
            skipped.append((key, f"nicht lesbar: {e}"))
            continue
        d = pd.DataFrame({"ts": ts, "rate": fr}).dropna()
        d = d.drop_duplicates("ts").sort_values("ts")
        if len(d) < MIN_EVENTS:
            skipped.append((key, f"nur {len(d)} Abrechnungen (< {MIN_EVENTS})"))
            continue
        mx = float(d["rate"].abs().max())
        if mx > SANITY_MAX_ABS_RATE:
            skipped.append((key, f"max |Rate| = {mx:.4f} -> sieht nach Prozent statt Bruch aus"))
            continue
        med_abs = float(d["rate"].abs().median())
        if med_abs > SANITY_MAX_MEDIAN_ABS:
            skipped.append((key, f"Median |Rate| = {med_abs:.4f} -> sieht nach Prozent statt Bruch aus"))
            continue
        idx = pd.to_datetime(d["ts"].astype("int64"), unit="ms", utc=True)
        data[key] = pd.Series(d["rate"].to_numpy(dtype=float), index=pd.DatetimeIndex(idx), name=key)
    return data, skipped


# ============================================================================================
# Simulation je Pair
# ============================================================================================
def simulate(rate, mode):
    """Simuliert ein Pair. mode = 'strategie' (Signal-Haltelogik) oder 'immer' (durchgehend gehalten).
    rate: Series, Bruch pro Abrechnung, Index = UTC-Zeitpunkt der Abrechnung.
    Rueckgabe: DataFrame (nur Auswertungszeitraum nach Warmup) mit Spalten
      rate    Funding-Rate dieser Abrechnung (Bruch)
      held    1.0 wenn die Position diese Abrechnung gehalten hat, sonst 0.0
      funding held * rate   (Bruch der Notional; negativ = bezahlt)
      entries 1.0 an Abrechnungen mit Positionsaufbau (Zaehler fuer Trades)
      cost    Ein-/Ausstiegskosten, die an dieser Abrechnung anfallen (Bruch der Notional, >= 0)
      haircut Basis-Abschlag (Bruch der Notional, >= 0)
      net     funding - cost - haircut
    oder None, falls zu wenig Daten."""
    idx = rate.index
    n = len(rate)
    r = rate.to_numpy(dtype=float)

    # Zeitabstand je Abrechnung in Tagen (fuer zeitanteiligen Haircut); erste = Median
    dt_days = pd.Series(idx).diff().dt.total_seconds().to_numpy() / 86400.0
    med = np.nanmedian(dt_days)
    dt_days = np.where(np.isfinite(dt_days), dt_days, med)

    # Signal: Summe der Raten der letzten LOOKBACK_DAYS Tage (Bruch), um 1 Abrechnung verschoben
    # -> fuer Abrechnung t sind nur Raten bis t-1 sichtbar (KEIN Look-ahead).
    window = pd.Timedelta(days=LOOKBACK_DAYS)
    roll_sum = rate.rolling(window).sum().shift(1).to_numpy()
    sig_ann = roll_sum / LOOKBACK_DAYS * 365.0          # Bruch pro Jahr (0.1095 = 10.95 % p.a.)
    # Warmup: Fenster muss voll gewesen sein (letzte sichtbare Abrechnung >= Start + LOOKBACK_DAYS)
    prev_ts = pd.Series(idx).shift(1)
    valid = (prev_ts >= idx[0] + window).to_numpy()
    sig_ann = np.where(valid, sig_ann, np.nan)

    first = int(np.argmax(valid)) if valid.any() else None
    if first is None or n - first < 30:
        return None

    held = np.zeros(n)
    cost = np.zeros(n)
    entries = np.zeros(n)                                # 1.0 an Abrechnungen, an denen aufgebaut wird
    pos = False
    for i in range(first, n):
        if mode == "immer":
            want = True
        else:
            s = sig_ann[i]                               # nur Info bis t-1
            want = (s > EXIT_ANN_MIN) if pos else (s > ENTRY_ANN_MIN)
        if want and not pos:
            cost[i] += ENTRY_COST                        # Aufbau: Spot kaufen + Perp shorten
            entries[i] = 1.0
        if pos and not want:
            cost[i] += EXIT_COST                         # Abbau: Spot verkaufen + Perp eindecken
        pos = want
        held[i] = 1.0 if pos else 0.0
    if pos:
        cost[n - 1] += EXIT_COST                         # offene Position am Ende zwangsweise schliessen

    funding = held * r                                   # +rate: Short kassiert; rate<0: Short zahlt
    haircut = held * BASIS_HAIRCUT_ANN * dt_days / 365.0 # Bruch der Notional, zeitanteilig
    ev = pd.DataFrame({"rate": r, "held": held, "funding": funding, "cost": cost,
                       "haircut": haircut, "dt_days": dt_days, "entries": entries}, index=idx).iloc[first:]
    ev["net"] = ev["funding"] - ev["cost"] - ev["haircut"]
    return ev


def monthly(series):
    """Summe je Kalendermonat (UTC) -> Series(Bruch der Notional), Index 'YYYY-MM'."""
    return series.groupby(series.index.strftime("%Y-%m")).sum()


def pair_stats(ev):
    """Kennzahlen eines Pairs (alle Renditen Bruch der Notional)."""
    span_days = (ev.index[-1] - ev.index[0]).total_seconds() / 86400.0 + float(ev["dt_days"].median())
    net = float(ev["net"].sum())
    cum = ev["net"].cumsum()
    neg = ev["rate"] < 0
    m = monthly(ev["net"])
    return {
        "span_days": span_days,
        "events": len(ev),
        "raw_sum": float(ev["rate"].sum()),                      # Summe aller Raten (so als waere immer gehalten, ohne Kosten)
        "gross": float(ev["funding"].sum()),                     # Funding waehrend Haltezeit, vor Kosten
        "cost": float(ev["cost"].sum()),
        "haircut": float(ev["haircut"].sum()),
        "net": net,
        "ann_net": net / span_days * 365.0,                      # Bruch p.a. auf Notional
        "ann_net_cap": net / span_days * 365.0 / CAPITAL_FACTOR, # Bruch p.a. auf Kapital
        "ann_gross": float(ev["funding"].sum()) / span_days * 365.0,
        "share_pos": float((ev["rate"] > 0).mean()),             # Anteil Abrechnungen mit Rate > 0
        "share_held": float(ev["held"].mean()),                  # Anteil Abrechnungen mit Position
        "neg_sum_all": float(ev.loc[neg, "rate"].sum()),         # Summe aller negativen Raten (<= 0)
        "neg_paid": float(ev.loc[neg, "funding"].sum()),         # davon tatsaechlich bezahlt (<= 0)
        "trades": int(ev["entries"].sum()),                      # Anzahl Halteperioden (je 1x Auf- und Abbau)
        "maxdd": float((cum - cum.cummax()).min()),              # <= 0, Bruch der Notional
        "months": m,
        "pos_months": int((m > 0).sum()),
        "n_months": int(len(m)),
    }


# ============================================================================================
# Portfolio (gleichgewichtet ueber alle Pairs mit Daten am jeweiligen Tag)
# ============================================================================================
def portfolio_daily(evs, col):
    """Tagesreihe des Portfolios fuer Spalte col: Mittelwert ueber die Pairs, die an dem Tag Daten haben
    (Gleichgewichtung der Notional). Rueckgabe: (Series Bruch der Notional/Tag, Series Anzahl Pairs)."""
    daily = {k: ev[col].resample("D").sum() for k, ev in evs.items()}
    df = pd.concat(daily, axis=1)
    return df.mean(axis=1, skipna=True), df.notna().sum(axis=1)


def portfolio_stats(evs):
    """Portfolio-Kennzahlen (Bruch der Notional; Mittel ueber Pairs, keine Wiederanlage)."""
    net_d, npairs = portfolio_daily(evs, "net")
    fund_d, _ = portfolio_daily(evs, "funding")
    cost_d, _ = portfolio_daily(evs, "cost")
    hair_d, _ = portfolio_daily(evs, "haircut")
    raw_d, _ = portfolio_daily(evs, "rate")
    days = len(net_d)
    cum = net_d.cumsum()
    m_net = net_d.groupby(net_d.index.strftime("%Y-%m")).sum()
    m_days = net_d.groupby(net_d.index.strftime("%Y-%m")).size()
    m_pairs = npairs.groupby(npairs.index.strftime("%Y-%m")).mean()
    total = float(net_d.sum())
    return {
        "days": days,
        "n_pairs": len(evs),
        "gross": float(fund_d.sum()), "cost": float(cost_d.sum()), "haircut": float(hair_d.sum()),
        "net": total,
        "ann_net": total / days * 365.0,
        "ann_net_cap": total / days * 365.0 / CAPITAL_FACTOR,
        "ann_gross": float(fund_d.sum()) / days * 365.0,
        "maxdd": float((cum - cum.cummax()).min()),
        "m_net": m_net, "m_days": m_days, "m_pairs": m_pairs,
        "raw_daily": raw_d, "net_daily": net_d,
    }


def month_stability(m_net, m_days):
    """Zaehlt positive Monate unter den vollen Monaten (>= FULL_MONTH_DAYS Tage). Randmonate nur mit Tagesanteil."""
    full = m_days[m_days >= FULL_MONTH_DAYS].index
    mm = m_net.loc[full] if len(full) else m_net
    return int((mm > 0).sum()), int(len(mm)), (float(mm.min()) if len(mm) else float("nan"))


# ============================================================================================
# Ausgabe
# ============================================================================================
def print_pair_table(rows):
    head = (f"{'Pair':<14}{'Tage':>6}{'pos.Int.':>9}{'gehalten':>9}{'Trades':>7}"
            f"{'brutto p.a.':>12}{'netto p.a.':>11}{'netto Kap.':>11}{'immer Kap.':>11}"
            f"{'Mon.+':>7}{'MaxDD':>8}")
    print(head)
    print("-" * len(head))
    for key, s, si in rows:
        print(f"{key:<14}{s['span_days']:>6.0f}{pct(s['share_pos'],0):>9}{pct(s['share_held'],0):>9}"
              f"{s['trades']:>7d}{pct(s['ann_gross']):>12}{pct(s['ann_net']):>11}{pct(s['ann_net_cap']):>11}"
              f"{pct(si['ann_net_cap']):>11}{s['pos_months']:>3d}/{s['n_months']:<3d}{pct(s['maxdd'],1):>8}")


def main(folder="data/funding", out_dir="data/funding_backtest"):
    selbsttest_einheiten()
    data, skipped = load_funding(folder)
    if not data:
        print(f"Keine verwertbaren *_funding.csv in {folder} gefunden. "
              f"Erst scripts/collect_funding.py ausfuehren.")
        for k, why in skipped[:10]:
            print(f"  uebersprungen {k}: {why}")
        sys.exit(1)

    print("=" * 100)
    print("FUNDING-ARB BACKTEST  (Spot long + Perp short, gleiche Notional, delta-neutral)")
    print("=" * 100)
    print("Einheiten: funding_rate = BRUCH pro Intervall (0.0001 = 0.01 % pro 8h = 10.95 % p.a.).")
    print("Alle Renditen unten in % der NOTIONAL eines Beins; 'Kap.' = bezogen auf Kapital = "
          f"Notional x {CAPITAL_FACTOR:.2f} (Spot voll + {pct(PERP_MARGIN,0)} Margin).")
    print(f"Kosten: Spot-Taker {pct(SPOT_TAKER)}, Perp-Taker {pct(FUT_TAKER)}, Slippage {pct(SLIPPAGE_PER_LEG)} je Bein "
          f"-> Einstieg {pct(ENTRY_COST)} + Ausstieg {pct(EXIT_COST)} der Notional je Halteperiode; "
          f"Basis-Abschlag {pct(BASIS_HAIRCUT_ANN)} p.a. waehrend der Haltezeit.")
    print(f"Haltelogik: Einstieg wenn Signal > {pct(ENTRY_ANN_MIN,0)} p.a., Ausstieg wenn <= {pct(EXIT_ANN_MIN,0)} p.a. "
          f"(Signal = Raten der letzten {LOOKBACK_DAYS} Tage bis t-1, hochgerechnet; kein Look-ahead).")
    print(f"Geladen: {len(data)} Pairs aus {folder}; uebersprungen: {len(skipped)}")
    for k, why in skipped[:10]:
        print(f"  uebersprungen {k}: {why}")

    # ---- Je Pair simulieren -------------------------------------------------------------------
    evs_s, evs_i, rows = {}, {}, []
    for key, rate in data.items():
        es = simulate(rate, "strategie")
        ei = simulate(rate, "immer")
        if es is None or ei is None:
            continue
        evs_s[key], evs_i[key] = es, ei
        rows.append((key, pair_stats(es), pair_stats(ei)))
    if not rows:
        print("Kein Pair mit genug Daten nach Warmup.")
        sys.exit(1)
    rows.sort(key=lambda r: r[1]["ann_net_cap"], reverse=True)

    print("\n" + "=" * 100)
    print("PRO PAIR  (Strategie = Signal-Haltelogik; 'immer Kap.' = durchgehend gehalten, netto auf Kapital)")
    print("=" * 100)
    print("brutto p.a. = Funding waehrend Haltezeit vor Kosten (auf Notional, ueber den ganzen Zeitraum "
          "gerechnet, auch Leerlauf-Zeit); netto = nach Kosten+Abschlag; Mon.+ = positive Monate/Monate.")
    if len(rows) > PRINT_TOP + PRINT_BOTTOM:
        print_pair_table(rows[:PRINT_TOP])
        print(f"  ... {len(rows) - PRINT_TOP - PRINT_BOTTOM} Pairs ausgeblendet (vollstaendig in pairs_summary.csv) ...")
        print_pair_table(rows[-PRINT_BOTTOM:])
    else:
        print_pair_table(rows)

    # ---- Portfolio ------------------------------------------------------------------------------
    ps = portfolio_stats(evs_s)
    pi = portfolio_stats(evs_i)
    print("\n" + "=" * 100)
    print(f"PORTFOLIO  ({ps['n_pairs']} Pairs gleichgewichtet, {ps['days']} Tage, keine Wiederanlage)")
    print("=" * 100)
    print(f"{'':<34}{'Strategie':>14}{'immer gehalten':>16}")
    def row(label, a, b):
        print(f"{label:<34}{a:>14}{b:>16}")
    row("Funding brutto (Summe, Notional)", pct(ps["gross"]), pct(pi["gross"]))
    row("- Ein-/Ausstiegskosten", pct(ps["cost"]), pct(pi["cost"]))
    row("- Basis-Abschlag", pct(ps["haircut"]), pct(pi["haircut"]))
    row("= netto (Summe, Notional)", pct(ps["net"]), pct(pi["net"]))
    row("brutto p.a. (Notional)", pct(ps["ann_gross"]), pct(pi["ann_gross"]))
    row("netto p.a. (Notional)", pct(ps["ann_net"]), pct(pi["ann_net"]))
    row("netto p.a. (Kapital)", pct(ps["ann_net_cap"]), pct(pi["ann_net_cap"]))
    row("Max. Drawdown (Notional)", pct(ps["maxdd"]), pct(pi["maxdd"]))
    print(f"(Vergleichszins ohne Risiko: {pct(RISKFREE_ANN)} p.a. auf Kapital)")

    print("\nMONATS-SPLIT PORTFOLIO (netto, Notional; p.a. = Monatswert auf Jahr hochgerechnet; * = Randmonat)")
    print(f"{'Monat':<9}{'Tage':>5}{'Pairs':>6}{'Strat. Mon.':>13}{'Strat. p.a.':>13}{'immer Mon.':>12}{'immer p.a.':>12}")
    for mon in ps["m_net"].index:
        d = int(ps["m_days"][mon])
        flag = "*" if d < FULL_MONTH_DAYS else " "
        sm = float(ps["m_net"][mon]); im = float(pi["m_net"].get(mon, np.nan))
        print(f"{mon:<8}{flag}{d:>5d}{ps['m_pairs'][mon]:>6.0f}{pct(sm):>13}{pct(sm / d * 365.0, 1):>13}"
              f"{pct(im):>12}{pct(im / d * 365.0, 1):>12}")
    pos_s, n_s, worst_s = month_stability(ps["m_net"], ps["m_days"])
    pos_i, n_i, worst_i = month_stability(pi["m_net"], pi["m_days"])
    print(f"Positive volle Monate: Strategie {pos_s}/{n_s} (schlechtester {pct(worst_s)}), "
          f"immer gehalten {pos_i}/{n_i} (schlechtester {pct(worst_i)})")

    # ---- Negativ-Funding -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("NEGATIV-FUNDING: WAS PASSIERT DA WIRKLICH?")
    print("=" * 100)
    neg_days = float((ps["raw_daily"] < 0).mean())
    avg_pos = float(np.mean([s["share_pos"] for _, s, _ in rows]))
    avg_neg_all = float(np.mean([s["neg_sum_all"] for _, s, _ in rows]))
    avg_neg_paid = float(np.mean([s["neg_paid"] for _, s, _ in rows]))
    print(f"Anteil Abrechnungen mit positivem Funding (Mittel ueber Pairs): {pct(avg_pos,1)}; "
          f"Tage, an denen das Pair-Mittel der Raten negativ war: {pct(neg_days,1)}")
    print(f"Summe aller NEGATIVEN Raten je Pair (Mittel, immer gehalten haette man das bezahlt): {pct(avg_neg_all)} der Notional")
    print(f"Davon tatsaechlich bezahlt von der Strategie (Signal-Verzoegerung):                  {pct(avg_neg_paid)} der Notional")
    vermieden = avg_neg_paid - avg_neg_all
    print(f"Durch Aussteigen vermieden:                                                          {pct(vermieden)} der Notional")
    print("Gegenrechnung: Aussteigen/Wiedereinsteigen kostet je Halteperiode "
          f"{pct(ENTRY_COST + EXIT_COST)} Round-Trip, und in der Flat-Zeit entgeht auch positives Funding.")
    print("Pairs mit der meisten bezahlten Negativ-Funding-Summe (Strategie):")
    worst = sorted(rows, key=lambda r: r[1]["neg_paid"])[:5]
    for key, s, si in worst:
        print(f"  {key:<14} bezahlt {pct(s['neg_paid'])}  (immer gehalten: {pct(si['neg_paid'])}),  "
              f"netto p.a. Kapital {pct(s['ann_net_cap'])}")

    # ---- Dateien -------------------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{
        "pair": k, "tage": round(s["span_days"], 1), "abrechnungen": s["events"],
        "anteil_pos_funding": round(s["share_pos"], 4), "anteil_gehalten": round(s["share_held"], 4),
        "trades": s["trades"],
        "ann_brutto_notional": round(s["ann_gross"], 6), "ann_netto_notional": round(s["ann_net"], 6),
        "ann_netto_kapital": round(s["ann_net_cap"], 6), "immer_ann_netto_kapital": round(si["ann_net_cap"], 6),
        "summe_raten_alle": round(s["raw_sum"], 6), "summe_brutto_gehalten": round(s["gross"], 6),
        "kosten": round(s["cost"], 6), "haircut": round(s["haircut"], 6), "summe_netto": round(s["net"], 6),
        "neg_summe_alle": round(s["neg_sum_all"], 6), "neg_bezahlt_strategie": round(s["neg_paid"], 6),
        "max_drawdown": round(s["maxdd"], 6), "monate_pos": s["pos_months"], "monate": s["n_months"],
    } for k, s, si in rows]).to_csv(os.path.join(out_dir, "pairs_summary.csv"), index=False)
    mp = pd.DataFrame({k: s["months"] for k, s, _ in rows}).sort_index()
    mp.round(6).to_csv(os.path.join(out_dir, "monthly_pairs_net_notional.csv"), index_label="monat")
    pd.DataFrame({"tage": ps["m_days"], "pairs": ps["m_pairs"].round(1),
                  "strategie_netto": ps["m_net"].round(6), "immer_netto": pi["m_net"].round(6)}) \
      .to_csv(os.path.join(out_dir, "monthly_portfolio.csv"), index_label="monat")
    print(f"\nCSV geschrieben nach {out_dir}/: pairs_summary.csv, monthly_pairs_net_notional.csv, monthly_portfolio.csv")

    # ---- Lesart --------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("LESART")
    print("=" * 100)
    print("Funding-Arb traegt NUR, wenn netto-annualisiert klar > 0 UND ueber Monate stabil, NACH Kosten.")
    print("Maschinelle Pruefung der obigen Zahlen (Strategie, Portfolio, auf Kapital):")
    ok_a = ps["ann_net_cap"] > RISKFREE_ANN
    ok_b = n_s > 0 and pos_s / n_s >= MIN_POS_MONTH_SHARE
    ok_c = worst_s > -0.5 * max(ps["ann_net"], 1e-9) / 12.0 if np.isfinite(worst_s) else False
    print(f"  [{'ja ' if ok_a else 'NEIN'}] netto p.a. (Kapital) {pct(ps['ann_net_cap'])} > Vergleichszins {pct(RISKFREE_ANN)}")
    print(f"  [{'ja ' if ok_b else 'NEIN'}] positive volle Monate {pos_s}/{n_s} >= {pct(MIN_POS_MONTH_SHARE,0)}")
    print(f"  [{'ja ' if ok_c else 'NEIN'}] schlechtester voller Monat {pct(worst_s)} nicht schlimmer als "
          f"-50 % eines durchschnittlichen Monats")
    verdict = "VIABEL (unter diesen Annahmen, in-sample)" if (ok_a and ok_b and ok_c) else "NICHT BELEGT"
    print(f"  => {verdict}")
    print("Vorsicht bei der Interpretation (alle Punkte machen das Ergebnis eher zu gut als zu schlecht):")
    print("  - In-sample, nur ~1-2 Jahre; Funding ist stark regimeabhaengig (Bullenphase = hohes Funding).")
    print("  - Survivorship: Universum = heute bekannte Pairs; delistete Perps fehlen.")
    print("  - Funding verschiedener Coins ist hoch korreliert -> Diversifikation traegt weniger als sie aussieht.")
    print("  - Nicht modelliert: Liquidationsrisiko des Short in Rallyes (Margin nachschiessen), Spot-USDC vs.")
    print("    Perp-USDT-Basis, Funding-Zahlung auf Mark-Preis statt Notional, Kapazitaet/Slippage bei Groesse, Steuern.")
    print("  - Slippage und Basis-Abschlag sind ANNAHMEN (Konstanten oben), nicht gemessen -> Sensitivitaet testen.")
    print("  - Viabel waere erst: netto klar ueber dem Vergleichszins, in der Mehrzahl der Monate, mit")
    print("    ueberlebbarem Worst-Month und einem Ergebnis, das gegen hoehere Kosten robust bleibt.")


if __name__ == "__main__":
    fo = sys.argv[1] if len(sys.argv) > 1 else "data/funding"
    od = sys.argv[2] if len(sys.argv) > 2 else "data/funding_backtest"
    main(fo, od)
