import pandas as pd
import numpy as np

# --------------------------------
# 1. Load prototype features
# --------------------------------

df = pd.read_csv(
    "data/processed/prototype_features.csv",
    parse_dates=["date"]
)

sample = df.dropna(
    subset=[
        "daily_return",
        "avol",
        "future_return_1d",
        "future_return_3d",
        "future_return_5d"
    ]
).copy()

# --------------------------------
# 2. Rank stocks by daily return
# --------------------------------

sample["return_rank"] = (
    sample.groupby("date")["daily_return"]
          .rank(pct=True)
)

positive = sample[
    sample["return_rank"] >= 0.80
].copy()

negative = sample[
    sample["return_rank"] <= 0.20
].copy()

# --------------------------------
# 3. Split each group by daily AVOL median
# --------------------------------

for group in [positive, negative]:

    group["avol_median"] = (
        group.groupby("date")["avol"]
             .transform("median")
    )

    group["volume_group"] = (
        group["avol"] >= group["avol_median"]
    )

# --------------------------------
# 4. Create daily High-Low spreads
# --------------------------------

def get_daily_spreads(data, horizon):

    col = f"future_return_{horizon}d"

    daily = (
        data.groupby(["date", "volume_group"])[col]
            .mean()
            .unstack()
            .rename(
                columns={
                    False: "Low_AVOL",
                    True: "High_AVOL"
                }
            )
            .dropna()
    )

    daily["spread"] = (
        daily["High_AVOL"]
        - daily["Low_AVOL"]
    )

    return daily["spread"]

# --------------------------------
# 5. Evaluate uncertainty
# --------------------------------

def summarize(spreads):

    n = len(spreads)

    mean = spreads.mean()
    std = spreads.std(ddof=1)

    se = std / np.sqrt(n)

    t_stat = mean / se if se != 0 else np.nan

    median = spreads.median()

    pct_positive = (
        (spreads > 0).mean() * 100
    )

    return {
        "days": n,
        "mean": mean * 100,
        "median": median * 100,
        "std": std * 100,
        "se": se * 100,
        "t_stat": t_stat,
        "positive_days": pct_positive
    }

# --------------------------------
# 6. Print results
# --------------------------------

for name, group in [
    ("POSITIVE MOVERS", positive),
    ("NEGATIVE MOVERS", negative)
]:

    print(f"\n{name}")

    for horizon in [1, 3, 5]:

        spreads = get_daily_spreads(
            group,
            horizon
        )

        s = summarize(spreads)

        print(f"\n{horizon}D")
        print("Number of days:", s["days"])
        print("Mean spread:", round(s["mean"], 4), "%")
        print("Median spread:", round(s["median"], 4), "%")
        print("Daily spread SD:", round(s["std"], 4), "%")
        print("Standard error:", round(s["se"], 4), "%")
        print("Naive t-stat:", round(s["t_stat"], 3))
        print(
            "Days with positive High-Low spread:",
            round(s["positive_days"], 1),
            "%"
        )