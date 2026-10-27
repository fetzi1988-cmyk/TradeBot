import ccxt, yaml, time
from pathlib import Path

PAIRS_FILE   = Path("/root/trading-system/config/pairs.yaml")
MIN_VOL_USDC = 3_000_000   # mind. 3M USDC 24h Volumen
BLACKLIST    = {"币安人生/USDC","FDUSD/USDC","USDE/USDC","PAXG/USDC",
                "XAUT/USDC","USD1/USDC","EURI/USDC","EUR/USDC",
                "BFUSD/USDC","TON/USDC","PENDLE/USDC"}

def main():
    ex = ccxt.binance({"enableRateLimit": True})
    tickers = ex.fetch_tickers()

    # Alle USDC Pairs mit ausreichend Volumen
    candidates = []
    for symbol, t in tickers.items():
        if not symbol.endswith("/USDC"): continue
        if symbol in BLACKLIST: continue
        vol = float(t.get("quoteVolume") or 0)
        if vol >= MIN_VOL_USDC:
            candidates.append(symbol)

    candidates = sorted(set(candidates))

    # Aktuelle pairs.yaml laden
    current = yaml.safe_load(PAIRS_FILE.read_text()) or {"pairs": []}
    existing = set(current.get("pairs", []))

    new_pairs = [c for c in candidates if c not in existing]
    removed   = [p for p in existing if p not in candidates and p != "币安人生/USDC"]

    if new_pairs:
        print(f"NEU: {new_pairs}")
        current["pairs"] = sorted(existing | set(candidates))

    if removed:
        print(f"ZU WENIG VOLUMEN (behalten): {removed}")

    # Immer Blacklist enforzen
    current["pairs"] = [p for p in current["pairs"] if p not in BLACKLIST]

    PAIRS_FILE.write_text(
        "pairs:\n" + "".join(f"- {p}\n" for p in sorted(current["pairs"]))
    )
    print(f"pairs.yaml: {len(current['pairs'])} Pairs total")

if __name__ == "__main__":
    main()
