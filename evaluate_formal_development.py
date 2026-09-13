import pandas as pd
import numpy as np
import statsmodels.api as sm


# ============================================================
# 1. Load formal development results
# ============================================================

df = pd.read_csv(
    "results/formal_development_daily_spreads.csv",
    parse_dates=["signal_date"]
)

print("Rows loaded:", len(df))
print("Signal dates:", df["signal_date"].nunique())


# ============================================================
# 2. Newey-West / HAC mean test
# ============================================================

def hac_mean_test(series, maxlags):

    x = series.dropna().astype(float)

    # Regression:
    # spread_t = constant + error_t
    #
    # The constant is simply the mean spread.
    X = np.ones((len(x), 1))

    model = sm.OLS(
        x.values,
        X
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": maxlags
        }
    )

    mean = model.params[0]
    hac_se = model.bse[0]
    hac_t = model.tvalues[0]

    return mean, hac_se, hac_t


# ============================================================
# 3. Evaluate each mover × horizon
# ============================================================

summary_rows = []

for mover in ["positive", "negative"]:

    print("\n" + "=" * 55)
    print(mover.upper(), "MOVERS")
    print("=" * 55)

    for horizon in [1, 3, 5]:

        subset = df[
            (df["mover"] == mover)
            & (df["horizon"] == horizon)
        ].copy()

        spreads = subset["spread"]

        n = len(spreads)

        mean = spreads.mean()
        median = spreads.median()
        std = spreads.std(ddof=1)

        naive_se = std / np.sqrt(n)
        naive_t = (
            mean / naive_se
            if naive_se > 0
            else np.nan
        )

        # Overlapping k-day returns imply
        # up to k-1 overlap lags
        maxlags = horizon - 1

        hac_mean, hac_se, hac_t = (
            hac_mean_test(
                spreads,
                maxlags=maxlags
            )
        )

        # ----------------------------------------------------
        # Price-move-only baseline
        #
        # Weighted combination of High / Low AVOL portfolios
        # ----------------------------------------------------

        total_n = (
            subset["n_high"]
            + subset["n_low"]
        )

        baseline_return = (
            (
                subset["high_avol_return"]
                * subset["n_high"]
            )
            +
            (
                subset["low_avol_return"]
                * subset["n_low"]
            )
        ) / total_n

        baseline_mean = baseline_return.mean()

        # ----------------------------------------------------
        # Directional consistency
        # ----------------------------------------------------

        if mover == "positive":

            # Hypothesis predicts High - Low > 0
            directional_pct = (
                (spreads > 0).mean()
                * 100
            )

            expected_sign = "> 0"

        else:

            # For negative movers, stronger continuation
            # means High AVOL should be MORE negative:
            # High - Low < 0
            directional_pct = (
                (spreads < 0).mean()
                * 100
            )

            expected_sign = "< 0"

        print(f"\n{horizon}D")

        print(
            "Price-move-only baseline:",
            round(
                baseline_mean * 100,
                4
            ),
            "%"
        )

        print(
            "Low AVOL return:",
            round(
                subset["low_avol_return"].mean()
                * 100,
                4
            ),
            "%"
        )

        print(
            "High AVOL return:",
            round(
                subset["high_avol_return"].mean()
                * 100,
                4
            ),
            "%"
        )

        print(
            "High - Low spread:",
            round(mean * 100, 4),
            "%"
        )

        print(
            "Median spread:",
            round(median * 100, 4),
            "%"
        )

        print(
            "Naive t-stat:",
            round(naive_t, 3)
        )

        print(
            "HAC SE:",
            round(hac_se * 100, 4),
            "%"
        )

        print(
            "HAC t-stat:",
            round(hac_t, 3)
        )

        print(
            f"Days with expected sign ({expected_sign}):",
            round(directional_pct, 1),
            "%"
        )

        summary_rows.append({
            "mover": mover,
            "horizon": horizon,
            "n_days": n,
            "baseline_return_pct":
                baseline_mean * 100,
            "low_avol_return_pct":
                subset["low_avol_return"].mean()
                * 100,
            "high_avol_return_pct":
                subset["high_avol_return"].mean()
                * 100,
            "spread_pct":
                mean * 100,
            "median_spread_pct":
                median * 100,
            "naive_t":
                naive_t,
            "hac_se_pct":
                hac_se * 100,
            "hac_t":
                hac_t,
            "directional_pct":
                directional_pct
        })


# ============================================================
# 4. Save summary
# ============================================================

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    "results/formal_development_summary.csv",
    index=False
)

print(
    "\nSaved formal development summary."
)