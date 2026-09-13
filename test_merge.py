import json
from pathlib import Path

DATE = "2025-01-02"

daily_file = Path(f"data/raw/daily_{DATE}.json")
universe_file = Path(f"data/raw/universe_{DATE}.json")

# Load cached Daily Market Summary
with open(daily_file, "r") as f:
    daily_data = json.load(f)

daily_results = daily_data.get("results", [])

# Load cached historical common-stock universe
with open(universe_file, "r") as f:
    universe_results = json.load(f)

# Create a set of common-stock tickers
common_stock_tickers = {
    row["ticker"]
    for row in universe_results
    if row.get("ticker") is not None
}

# Keep only Daily Market Summary rows whose ticker is a common stock
common_stock_daily = [
    row
    for row in daily_results
    if row.get("T") in common_stock_tickers
]

print("Daily market securities:", len(daily_results))
print("Historical common stocks:", len(common_stock_tickers))
print("Common stocks with market data:", len(common_stock_daily))