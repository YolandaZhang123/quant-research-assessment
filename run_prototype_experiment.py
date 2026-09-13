import pandas as pd

# --------------------------------
# 1. Load processed prototype
# --------------------------------

df = pd.read_csv(
    "data/processed/prototype_features.csv",
    parse_dates=["date"]
)

# Need return + AVOL
sample = df.dropna(
    subset=[
        "daily_return",
        "avol",
        "future_return_1d",
        "future_return_3d",
        "future_return_5d"
    ]
).copy()

print("Usable observations:", len(sample))

# --------------------------------
# 2. Daily return ranks
# --------------------------------

sample["return_rank"] = (
    sample.groupby("date")["daily_return"]
          .rank(pct=True)
)

# Top 20% / Bottom 20%
positive = sample[
    sample["return_rank"] >= 0.80
].copy()

negative = sample[
    sample["return_rank"] <= 0.20
].copy()

# --------------------------------
# 3. Within each mover group,
#    split AVOL at the DAILY median
# --------------------------------

positive["avol_median"] = (
    positive.groupby("date")["avol"]
            .transform("median")
)

negative["avol_median"] = (
    negative.groupby("date")["avol"]
            .transform("median")
)

positive["volume_group"] = (
    positive["avol"] >= positive["avol_median"]
)

negative["volume_group"] = (
    negative["avol"] >= negative["avol_median"]
)

# True = High AVOL
# False = Low AVOL

# --------------------------------
# 4. Calculate DAILY portfolio returns
# --------------------------------

def daily_group_returns(data, horizon):

    col = f"future_return_{horizon}d"

    return (
        data.groupby(
            ["date", "volume_group"]
        )[col]
        .mean()
        .unstack()
        .rename(
            columns={
                False: "Low_AVOL",
                True: "High_AVOL"
            }
        )
    )

# --------------------------------
# 5. Print summary spreads
# --------------------------------

for name, group in [
    ("POSITIVE MOVERS", positive),
    ("NEGATIVE MOVERS", negative)
]:

    print(f"\n{name}")

    for horizon in [1, 3, 5]:

        daily = daily_group_returns(
            group,
            horizon
        )

        daily["spread"] = (
            daily["High_AVOL"]
            - daily["Low_AVOL"]
        )

        print(f"\n{horizon}D")

        print(
            "Low AVOL avg:",
            round(
                daily["Low_AVOL"].mean() * 100,
                4
            ),
            "%"
        )

        print(
            "High AVOL avg:",
            round(
                daily["High_AVOL"].mean() * 100,
                4
            ),
            "%"
        )

        print(
            "High - Low spread:",
            round(
                daily["spread"].mean() * 100,
                4
            ),
            "%"
        )