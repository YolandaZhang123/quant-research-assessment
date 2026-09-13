from pathlib import Path
import json

data_dir = Path("data/raw/daily")

files = sorted(data_dir.glob("daily_*.json"))

total_files = len(files)
trading_days = 0
empty_days = []

stock_counts = []

for file in files:
    with open(file, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    count = len(results)

    date = file.stem.replace("daily_", "")

    if count == 0:
        empty_days.append(date)
    else:
        trading_days += 1
        stock_counts.append(count)

print("Cached weekday files:", total_files)
print("Days with market data:", trading_days)
print("Days with no market data:", len(empty_days))

print("\nDates with no market data:")
for date in empty_days:
    print(date)

if stock_counts:
    print("\nStocks per active trading day")
    print("Minimum:", min(stock_counts))
    print("Maximum:", max(stock_counts))
    print("Average:", round(sum(stock_counts) / len(stock_counts), 1))