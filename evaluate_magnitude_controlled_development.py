import pandas as pd
import numpy as np
import statsmodels.api as sm


# ============================================================
# 1. Load magnitude-controlled development results
# ============================================================

df = pd.read_csv(
    "results/magnitude_controlled_development.csv",
    parse_dates=["signal_date"]
)

print("Rows loaded:", len(df))
print(
    "Unique signal dates:",
    df["signal_date"].nunique()
)


# ============================================================
# 2. HAC / Newey-West mean test
# ============================================================

def hac_mean_test(series, maxlags):

    x = series.dropna().astype(float)

    X = np.ones(
        (len(x), 1)
    )

    model = sm.OLS(
        x.values,
        X
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": maxlags
        }
    )

    return (
        model.params[0],
        model.bse[0],
        model.tvalues[0]
    )


# ============================================================
# 3. Evaluate mover × horizon
# ============================================================

summary_rows = []

for mover in [
    "positive",
    "negative"
]:

    print(
        "\n========================================"
    )

    print(
        mover.upper(),
        "MOVERS"
    )

    print(
        "========================================"
    )

    for horizon in [1, 3, 5]:

        x = df[
            (df["mover"] == mover)
            & (df["horizon"] == horizon)
        ].copy()

        spread = x["spread"]

        n = len(spread)

        mean_spread = spread.mean()
        median_spread = spread.median()

        # Simple controlled average of the
        # two AVOL portfolios
        controlled_baseline = (
            (
                x["low_avol_return"]
                +
                x["high_avol_return"]
            )
            / 2
        ).mean()

        maxlags = horizon - 1

        mean_hac, hac_se, hac_t = (
            hac_mean_test(
                spread,
                maxlags
            )
        )

        # Original continuation hypothesis:
        #
        # Positive movers:
        # High AVOL should outperform
        #
        # Negative movers:
        # High AVOL should underperform
        if mover == "positive":

            expected_pct = (
                (spread > 0).mean()
                * 100
            )

            expected_sign = "> 0"

        else:

            expected_pct = (
                (spread < 0).mean()
                * 100
            )

            expected_sign = "< 0"

        signal_difference = (
            (
                x[
                    "high_signal_return"
                ]
                -
                x[
                    "low_signal_return"
                ]
            ).mean()
        )

        print(f"\n{horizon}D")

        print(
            "Observations:",
            n
        )

        print(
            "Controlled baseline:",
            round(
                controlled_baseline
                * 100,
                4
            ),
            "%"
        )

        print(
            "Low AVOL future return:",
            round(
                x[
                    "low_avol_return"
                ].mean()
                * 100,
                4
            ),
            "%"
        )

        print(
            "High AVOL future return:",
            round(
                x[
                    "high_avol_return"
                ].mean()
                * 100,
                4
            ),
            "%"
        )

        print(
            "High - Low spread:",
            round(
                mean_spread * 100,
                4
            ),
            "%"
        )

        print(
            "Median spread:",
            round(
                median_spread * 100,
                4
            ),
            "%"
        )

        print(
            "HAC SE:",
            round(
                hac_se * 100,
                4
            ),
            "%"
        )

        print(
            "HAC t-stat:",
            round(
                hac_t,
                3
            )
        )

        print(
            f"Days with original expected sign "
            f"({expected_sign}):",
            round(
                expected_pct,
                1
            ),
            "%"
        )

        print(
            "Signal-day High - Low difference:",
            round(
                signal_difference
                * 100,
                4
            ),
            "%"
        )

        summary_rows.append({
            "mover": mover,
            "horizon": horizon,
            "n_days": n,

            "controlled_baseline_pct":
                controlled_baseline
                * 100,

            "low_avol_return_pct":
                x[
                    "low_avol_return"
                ].mean()
                * 100,

            "high_avol_return_pct":
                x[
                    "high_avol_return"
                ].mean()
                * 100,

            "spread_pct":
                mean_spread * 100,

            "median_spread_pct":
                median_spread * 100,

            "hac_se_pct":
                hac_se * 100,

            "hac_t":
                hac_t,

            "expected_sign_pct":
                expected_pct,

            "signal_difference_pct":
                signal_difference * 100
        })


# ============================================================
# 4. Save summary
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    "results/"
    "magnitude_controlled_development_summary.csv",
    index=False
)

print(
    "\nSaved magnitude-controlled "
    "development summary."
)