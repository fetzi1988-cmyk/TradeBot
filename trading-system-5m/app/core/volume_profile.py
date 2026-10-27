"""
Volume Profile — VPOC, VAH, VAL
Berechnet Liquiditätszonen aus OHLCV Kerzen.
Wissenschaftlich stärker als OBV allein.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone


def _log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}")


def compute(df: pd.DataFrame, n_bins: int = 100, value_area_pct: float = 0.70) -> dict:
    """
    Berechnet Volume Profile für einen DataFrame.
    
    Returns:
        vpoc: Price of Control — Preisniveau mit höchstem Volumen
        vah:  Value Area High — obere Grenze der 70% Value Area
        val:  Value Area Low  — untere Grenze der 70% Value Area
        poc_distance_pct: Abstand des aktuellen Preises vom VPOC in %
        in_value_area: ob Preis in der Value Area liegt
    """
    if len(df) < 20:
        # Sichtbarkeits-Fix 01.07.2026: frisch rotierte Pairs mit wenig
        # 5m-Historie wurden bisher stillschweigend vom VP-Gate ausgenommen
        # (poc_dist=0.0 Default, sieht aus wie 'am POC', ist aber 'kein
        # Profile berechenbar'). Jetzt sichtbar im Log statt stiller Skip.
        _log(f"vp_empty_insufficient_history rows={len(df)}")
        return _empty()

    try:
        price_min = float(df["low"].min())
        price_max = float(df["high"].max())
        current   = float(df["close"].iloc[-1])

        if price_max <= price_min:
            _log(f"vp_empty_flat_range price_min={price_min} price_max={price_max}")
            return _empty()

        # Volumen auf Preisniveaus verteilen
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        vol_at_price = np.zeros(n_bins)

        for _, row in df.iterrows():
            # Gleichmässige Verteilung des Kerzenvolumens über den Kerzenkörper
            low_idx  = np.searchsorted(bin_edges, row["low"],  side="left")
            high_idx = np.searchsorted(bin_edges, row["high"], side="right")
            low_idx  = max(0, min(low_idx,  n_bins - 1))
            high_idx = max(0, min(high_idx, n_bins))
            if high_idx > low_idx:
                vol_per_bin = row["volume"] / (high_idx - low_idx)
                vol_at_price[low_idx:high_idx] += vol_per_bin

        # Bin-Mittelpunkte
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

        # VPOC = Bin mit höchstem Volumen
        vpoc_idx = int(np.argmax(vol_at_price))
        vpoc = float(bin_mids[vpoc_idx])

        # Value Area: 70% des Gesamtvolumens um VPOC
        total_vol = vol_at_price.sum()
        target_vol = total_vol * value_area_pct

        # Erweitere von VPOC nach oben und unten
        above = list(range(vpoc_idx + 1, n_bins))
        below = list(range(vpoc_idx - 1, -1, -1))

        included = {vpoc_idx}
        accumulated = vol_at_price[vpoc_idx]
        ai, bi = 0, 0

        while accumulated < target_vol:
            add_above = vol_at_price[above[ai]] if ai < len(above) else 0
            add_below = vol_at_price[below[bi]] if bi < len(below) else 0

            if add_above == 0 and add_below == 0:
                break
            if add_above >= add_below and ai < len(above):
                included.add(above[ai])
                accumulated += add_above
                ai += 1
            elif bi < len(below):
                included.add(below[bi])
                accumulated += add_below
                bi += 1
            else:
                break

        va_prices = [bin_mids[i] for i in included]
        vah = float(max(va_prices))
        val = float(min(va_prices))

        poc_dist = (current - vpoc) / vpoc * 100 if vpoc > 0 else 0.0
        in_va    = val <= current <= vah

        return {
            "vpoc":             vpoc,
            "vah":              vah,
            "val":              val,
            "poc_distance_pct": poc_dist,
            "in_value_area":    in_va,
            "above_vah":        current > vah,
            "below_val":        current < val,
            "valid":            True
        }

    except Exception as e:
        _log(f"volume_profile FEHLER: {e}")
        return _empty()


def _empty() -> dict:
    return {
        "vpoc": 0.0, "vah": 0.0, "val": 0.0,
        "poc_distance_pct": 0.0, "in_value_area": True,
        "above_vah": False, "below_val": False,
        "valid": False
    }


def compute_for_pair(pair: str, candles_dir: str = "data/candles",
                     lookback: int = 500) -> dict:
    """Berechnet Volume Profile für ein Pair aus lokalen Kerzen"""
    fname = pair.replace("/", "_") + "_5m.csv"
    path  = Path(candles_dir) / fname

    if not path.exists():
        return _empty()

    df = pd.read_csv(path).tail(lookback)
    if len(df) < 20:
        return _empty()

    return compute(df)


def add_vp_features(df: pd.DataFrame, lookback: int = 200) -> pd.DataFrame:
    """
    Fügt Volume Profile Features zu einem Feature DataFrame hinzu.

    Live-Speed-Fix:
    Früher wurde fuer JEDE Zeile ein eigenes Rolling Volume Profile berechnet.
    Bei 420 Kerzen waren das 420 teure Profile pro Pair.

    Fuer Live-Routing brauchen wir vor allem die letzten abgeschlossenen Kerzen.
    Deshalb:
    - Kontextfenster bleibt lookback=200
    - berechnet werden nur die letzten VP_LIVE_TARGET_ROWS Zeilen
    - aeltere Zeilen bekommen neutrale Defaults
    """
    out = df.copy()
    n = len(df)

    try:
        import os as _os
        target_rows = int(_os.environ.get("VP_LIVE_TARGET_ROWS", "60"))
    except Exception:
        target_rows = 60

    target_rows = max(3, min(int(target_rows), n if n > 0 else 3))

    vpoc_list = [0.0] * n
    vah_list = [0.0] * n
    val_list = [0.0] * n
    poc_dist_list = [0.0] * n
    in_va_list = [True] * n
    above_vah_list = [False] * n
    below_val_list = [False] * n

    calc_start = max(0, n - target_rows)

    for i in range(calc_start, n):
        start = max(0, i - lookback + 1)
        window = df.iloc[start:i+1]
        vp = compute(window)

        vpoc_list[i] = vp["vpoc"]
        vah_list[i] = vp["vah"]
        val_list[i] = vp["val"]
        poc_dist_list[i] = vp["poc_distance_pct"]
        in_va_list[i] = vp["in_value_area"]
        above_vah_list[i] = vp["above_vah"]
        below_val_list[i] = vp["below_val"]

    out["vp_vpoc"]       = vpoc_list
    out["vp_vah"]        = vah_list
    out["vp_val"]        = val_list
    out["vp_poc_dist"]   = poc_dist_list
    out["vp_in_va"]      = in_va_list
    out["vp_above_vah"]  = above_vah_list
    out["vp_below_val"]  = below_val_list
    return out

