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

# Remaining signal-day imbalance
df["signal_gap"] = (
    df["high_signal_return"]
    - df["low_signal_return"]
)


# ============================================================
# 2. Continuous-control regression
#
# future spread
# = alpha
# + gamma * remaining signal-day return gap
# + error
#
# alpha = expected High-Low future spread
#         when signal-day return gap = 0
# ============================================================

summary_rows = []

for mover in ["positive", "negative"]:

    print("\n========================================")
    print(mover.upper(), "MOVERS")
    print("========================================")

    for horizon in [1, 3, 5]:

        x = df[
            (df["mover"] == mover)
            & (df["horizon"] == horizon)
        ].copy()

        y = x["spread"]
        signal_gap = x["signal_gap"]

        # Add regression intercept
        X = sm.add_constant(signal_gap)

        # HAC / Newey-West
        model = sm.OLS(
            y,
            X
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": horizon - 1
            }
        )

        alpha = model.params["const"]
        alpha_se = model.bse["const"]
        alpha_t = model.tvalues["const"]

        gamma = model.params["signal_gap"]
        gamma_t = model.tvalues["signal_gap"]

        raw_mean = y.mean()

        print(f"\n{horizon}D")

        print(
            "Raw High - Low spread:",
            round(raw_mean * 100, 4),
            "%"
        )

        print(
            "Mean signal-day gap:",
            round(signal_gap.mean() * 100, 4),
            "%"
        )

        print(
            "Signal-gap P5 / P95:",
            round(
                signal_gap.quantile(0.05) * 100,
                4
            ),
            "%",
            "/",
            round(
                signal_gap.quantile(0.95) * 100,
                4
            ),
            "%"
        )

        print(
            "Days signal gap >= 0:",
            round(
                (signal_gap >= 0).mean()
                * 100,
                1
            ),
            "%"
        )

        print(
            "Adjusted spread at signal gap = 0:",
            round(alpha * 100, 4),
            "%"
        )

        print(
            "Adjusted HAC SE:",
            round(alpha_se * 100, 4),
            "%"
        )

        print(
            "Adjusted HAC t-stat:",
            round(alpha_t, 3)
        )

        print(
            "Signal-gap coefficient:",
            round(gamma, 4)
        )

        print(
            "Signal-gap coefficient t-stat:",
            round(gamma_t, 3)
        )

        print(
            "R-squared:",
            round(model.rsquared, 4)
        )

        summary_rows.append({
            "mover": mover,
            "horizon": horizon,
            "raw_spread_pct":
                raw_mean * 100,
            "mean_signal_gap_pct":
                signal_gap.mean() * 100,
            "adjusted_spread_pct":
                alpha * 100,
            "adjusted_hac_se_pct":
                alpha_se * 100,
            "adjusted_hac_t":
                alpha_t,
            "signal_gap_coefficient":
                gamma,
            "signal_gap_t":
                gamma_t,
            "r_squared":
                model.rsquared
        })


# ============================================================
# 3. Save
# ============================================================

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    "results/"
    "continuous_return_control_summary.csv",
    index=False
)

print(
    "\nSaved continuous return-control summary."
)