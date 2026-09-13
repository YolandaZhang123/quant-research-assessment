from pathlib import Path
import json
import pandas as pd

# --------------------------------
# Settings
# --------------------------------

REFERENCE_DATE = "2025-01-02"

daily_dir = Path("data/raw/daily")
universe_file = Path(f"data/raw/universe_{REFERENCE_DATE}.json")

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------
# 1. Load common-stock universe
# --------------------------------

with open(universe_file, "r") as f:
    universe_data = json.load(f)

common_tickers = {
    row["ticker"]
    for row in universe_data
    if row.get("ticker") is not None
}

# --------------------------------
# 2. Combine daily market files
# --------------------------------

rows = []

for file in sorted(daily_dir.glob("daily_*.json")):

    date = file.stem.replace("daily_", "")

    with open(file, "r") as f:
        data = json.load(f)

    results = data.get("results", [])

    # Skip market holidays / closures
    if not results:
        continue

    for row in results:

        ticker = row.get("T")

        # Prototype only:
        # use 2025-01-02 common-stock universe
        if ticker not in common_tickers:
            continue

        rows.append({
            "date": date,
            "ticker": ticker,
            "close": row.get("c"),
            "volume": row.get("v")
        })

df = pd.DataFrame(rows)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["ticker", "date"]
).reset_index(drop=True)

print("Rows:", len(df))
print("Unique stocks:", df["ticker"].nunique())
print("Trading days:", df["date"].nunique())

# --------------------------------
# 3. Create global trading-day index
# --------------------------------

trading_dates = sorted(df["date"].unique())

date_to_index = {
    date: i
    for i, date in enumerate(trading_dates)
}

df["day_index"] = df["date"].map(date_to_index)

# --------------------------------
# 4. Daily return
# --------------------------------

df["prev_close"] = (
    df.groupby("ticker")["close"]
      .shift(1)
)

df["prev_day_index"] = (
    df.groupby("ticker")["day_index"]
      .shift(1)
)

# Only calculate return if previous observation
# was the immediately previous trading day

valid_previous_day = (
    df["day_index"] - df["prev_day_index"] == 1
)

df["daily_return"] = (
    df["close"] / df["prev_close"] - 1
).where(valid_previous_day)

# --------------------------------
# 5. Abnormal Volume
# --------------------------------

# Average volume over PREVIOUS 20 observations.
# Signal day's own volume is NOT included.

df["avg_volume_20d"] = (
    df.groupby("ticker")["volume"]
      .transform(
          lambda x: x.shift(1)
                     .rolling(20, min_periods=20)
                     .mean()
      )
)

df["avol"] = (
    df["volume"] / df["avg_volume_20d"]
)

# --------------------------------
# 6. Dollar-volume liquidity measure
# --------------------------------

df["dollar_volume"] = (
    df["close"] * df["volume"]
)

df["avg_dollar_volume_20d"] = (
    df.groupby("ticker")["dollar_volume"]
      .transform(
          lambda x: x.shift(1)
                     .rolling(20, min_periods=20)
                     .mean()
      )
)

# --------------------------------
# 7. Exact future returns
# --------------------------------

for horizon in [1, 3, 5]:

    future = df[
        ["ticker", "day_index", "close"]
    ].copy()

    future["day_index"] = (
        future["day_index"] - horizon
    )

    future = future.rename(
        columns={
            "close": f"future_close_{horizon}d"
        }
    )

    df = df.merge(
        future,
        on=["ticker", "day_index"],
        how="left"
    )

    df[f"future_return_{horizon}d"] = (
        df[f"future_close_{horizon}d"]
        / df["close"]
        - 1
    )

# --------------------------------
# 8. Save processed prototype
# --------------------------------

output_file = (
    output_dir / "prototype_features.csv"
)

df.to_csv(output_file, index=False)

# --------------------------------
# 9. Print ONLY summary information
# --------------------------------

print("\nFEATURE AVAILABILITY")

print(
    "Daily return available:",
    df["daily_return"].notna().sum()
)

print(
    "AVOL available:",
    df["avol"].notna().sum()
)

print(
    "Future 1D available:",
    df["future_return_1d"].notna().sum()
)

print(
    "Future 3D available:",
    df["future_return_3d"].notna().sum()
)

print(
    "Future 5D available:",
    df["future_return_5d"].notna().sum()
)

print("\nSaved processed prototype.")