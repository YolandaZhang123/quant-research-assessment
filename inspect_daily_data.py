import json
from pathlib import Path

DATE = "2025-01-02"

daily_file = Path(f"data/raw/daily_{DATE}.json")
universe_file = Path(f"data/raw/universe_{DATE}.json")

# Load cached data
with open(daily_file, "r") as f:
    daily_data = json.load(f)

with open(universe_file, "r") as f:
    universe_data = json.load(f)

daily_results = daily_data.get("results", [])

common_tickers = {
    row["ticker"]
    for row in universe_data
    if row.get("ticker") is not None
}

# Keep common stocks only
stocks = [
    row for row in daily_results
    if row.get("T") in common_tickers
]

print("Common stocks with daily data:", len(stocks))

# -------------------------
# Basic data-quality checks
# -------------------------

missing_close = sum(
    1 for row in stocks
    if row.get("c") is None
)

missing_volume = sum(
    1 for row in stocks
    if row.get("v") is None
)

nonpositive_close = sum(
    1 for row in stocks
    if row.get("c") is not None and row["c"] <= 0
)

zero_volume = sum(
    1 for row in stocks
    if row.get("v") == 0
)

price_below_1 = sum(
    1 for row in stocks
    if row.get("c") is not None and row["c"] < 1
)

price_below_5 = sum(
    1 for row in stocks
    if row.get("c") is not None and row["c"] < 5
)

print("\nDATA QUALITY")
print("Missing close:", missing_close)
print("Missing volume:", missing_volume)
print("Non-positive close:", nonpositive_close)
print("Zero volume:", zero_volume)

print("\nLOW-PRICE STOCKS")
print("Close < $1:", price_below_1)
print("Close < $5:", price_below_5)