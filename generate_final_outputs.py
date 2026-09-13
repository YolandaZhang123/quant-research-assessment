from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = Path("results")
FINAL_DIR = RESULTS_DIR / "final_outputs"

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. Load locked development + holdout data
# ============================================================

development = pd.read_csv(
    RESULTS_DIR
    / "magnitude_controlled_development.csv"
)

holdout = pd.read_csv(
    RESULTS_DIR
    / "locked_holdout_daily.csv"
)

robustness = pd.read_csv(
    RESULTS_DIR
    / "development_robustness_summary.csv"
)


# ============================================================
# 2. HAC helper
# ============================================================

def hac_summary(series, horizon):

    x = (
        series
        .dropna()
        .astype(float)
    )

    X = np.ones(
        (len(x), 1)
    )

    model = sm.OLS(
        x.values,
        X
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": horizon - 1
        }
    )

    mean = model.params[0]
    se = model.bse[0]
    tstat = model.tvalues[0]

    return (
        mean,
        se,
        tstat
    )


# ============================================================
# 3. Continuous signal-return control helper
# ============================================================

def controlled_summary(df, horizon):

    y = df["spread"]

    signal_gap = (
        df["high_signal_return"]
        - df["low_signal_return"]
    )

    X = sm.add_constant(
        signal_gap
    )

    model = sm.OLS(
        y,
        X
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": horizon - 1
        }
    )

    return (
        model.params["const"],
        model.bse["const"],
        model.tvalues["const"]
    )


# ============================================================
# 4. Build final main-results table
# ============================================================

main_rows = []

for sample_name, data in [
    ("Development", development),
    ("Holdout", holdout)
]:

    for mover in [
        "positive",
        "negative"
    ]:

        for horizon in [
            1, 3, 5
        ]:

            x = data[
                (data["mover"] == mover)
                &
                (data["horizon"] == horizon)
            ].copy()

            mean, se, tstat = (
                hac_summary(
                    x["spread"],
                    horizon
                )
            )

            adjusted, adjusted_se, adjusted_t = (
                controlled_summary(
                    x,
                    horizon
                )
            )

            if mover == "positive":

                hypothesis_sign_pct = (
                    (x["spread"] > 0)
                    .mean()
                    * 100
                )

            else:

                # Positive spread = reversal,
                # which is the revised finding.
                hypothesis_sign_pct = (
                    (x["spread"] > 0)
                    .mean()
                    * 100
                )

            main_rows.append({
                "Sample":
                    sample_name,

                "Mover":
                    mover.capitalize(),

                "Horizon":
                    f"{horizon}D",

                "N":
                    len(x),

                "High-Low Spread (bps)":
                    mean * 10000,

                "Median Spread (bps)":
                    x["spread"].median()
                    * 10000,

                "HAC SE (bps)":
                    se * 10000,

                "HAC t-stat":
                    tstat,

                "Positive Spread Days (%)":
                    hypothesis_sign_pct,

                "Signal-Day Gap (bps)":
                    (
                        x[
                            "high_signal_return"
                        ]
                        -
                        x[
                            "low_signal_return"
                        ]
                    ).mean()
                    * 10000,

                "Adjusted Spread (bps)":
                    adjusted * 10000,

                "Adjusted HAC t-stat":
                    adjusted_t
            })


main_table = pd.DataFrame(
    main_rows
)

main_table.to_csv(
    FINAL_DIR
    / "table1_main_results.csv",
    index=False
)


# ============================================================
# 5. Clean robustness table
#
# Negative 3D / 5D only because these were the
# pre-holdout revised hypothesis focus.
# ============================================================

robustness_negative = robustness[
    (
        robustness["mover"]
        == "negative"
    )
    &
    (
        robustness["horizon"]
        .isin([3, 5])
    )
].copy()


spec_names = {
    "MAIN_20D_1M_TAIL20":
        "Main: AVOL20, ADV $1M, Tail 20%",

    "AVOL10":
        "AVOL lookback 10D",

    "AVOL40":
        "AVOL lookback 40D",

    "LIQ5M":
        "ADV >= $5M",

    "LIQ10M":
        "ADV >= $10M",

    "TAIL10":
        "Return tail 10%"
}

robustness_negative[
    "Specification"
] = robustness_negative[
    "spec"
].map(spec_names)


robustness_table = robustness_negative[
    [
        "Specification",
        "horizon",
        "n_days",
        "spread_pct",
        "median_pct",
        "hac_t",
        "signal_gap_pct"
    ]
].copy()


robustness_table[
    "Spread (bps)"
] = (
    robustness_table[
        "spread_pct"
    ]
    * 100
)

robustness_table[
    "Median (bps)"
] = (
    robustness_table[
        "median_pct"
    ]
    * 100
)

robustness_table[
    "Signal-Day Gap (bps)"
] = (
    robustness_table[
        "signal_gap_pct"
    ]
    * 100
)


robustness_table = robustness_table[
    [
        "Specification",
        "horizon",
        "n_days",
        "Spread (bps)",
        "Median (bps)",
        "hac_t",
        "Signal-Day Gap (bps)"
    ]
]

robustness_table.columns = [
    "Specification",
    "Horizon",
    "N",
    "Spread (bps)",
    "Median (bps)",
    "HAC t-stat",
    "Signal-Day Gap (bps)"
]


robustness_table.to_csv(
    FINAL_DIR
    / "table2_negative_robustness.csv",
    index=False
)


# ============================================================
# 6. FIGURE 1
#
# Negative movers:
# Development vs Holdout
# with HAC 95% confidence intervals
# ============================================================

fig1_rows = []

for sample_name, data in [
    ("Development", development),
    ("Holdout", holdout)
]:

    for horizon in [1, 3, 5]:

        x = data[
            (
                data["mover"]
                == "negative"
            )
            &
            (
                data["horizon"]
                == horizon
            )
        ]

        mean, se, tstat = (
            hac_summary(
                x["spread"],
                horizon
            )
        )

        fig1_rows.append({
            "Sample":
                sample_name,

            "Horizon":
                horizon,

            "Mean":
                mean * 10000,

            "SE":
                se * 10000
        })


fig1_df = pd.DataFrame(
    fig1_rows
)

horizons = np.array(
    [1, 3, 5]
)

width = 0.32

dev = fig1_df[
    fig1_df["Sample"]
    == "Development"
]

test = fig1_df[
    fig1_df["Sample"]
    == "Holdout"
]


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    horizons - width / 2,
    dev["Mean"],
    width=width,
    yerr=1.96 * dev["SE"],
    capsize=4,
    label="Development"
)

plt.bar(
    horizons + width / 2,
    test["Mean"],
    width=width,
    yerr=1.96 * test["SE"],
    capsize=4,
    label="Holdout"
)

plt.axhline(
    0,
    linewidth=1
)

plt.xticks(
    horizons,
    ["1D", "3D", "5D"]
)

plt.ylabel(
    "High AVOL − Low AVOL Future Return (bps)"
)

plt.xlabel(
    "Forward Return Horizon"
)

plt.title(
    "Negative Movers: Abnormal Volume and Subsequent Reversal"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FINAL_DIR
    / "figure1_negative_development_vs_holdout.png",
    dpi=300
)

plt.close()


# ============================================================
# 7. FIGURE 2
#
# Development asymmetry:
# Positive movers vs Negative movers
# ============================================================

fig2_rows = []

for mover in [
    "positive",
    "negative"
]:

    for horizon in [1, 3, 5]:

        x = development[
            (
                development["mover"]
                == mover
            )
            &
            (
                development["horizon"]
                == horizon
            )
        ]

        mean, se, tstat = (
            hac_summary(
                x["spread"],
                horizon
            )
        )

        fig2_rows.append({
            "Mover":
                mover.capitalize(),

            "Horizon":
                horizon,

            "Mean":
                mean * 10000,

            "SE":
                se * 10000
        })


fig2_df = pd.DataFrame(
    fig2_rows
)

positive = fig2_df[
    fig2_df["Mover"]
    == "Positive"
]

negative = fig2_df[
    fig2_df["Mover"]
    == "Negative"
]


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    horizons - width / 2,
    positive["Mean"],
    width=width,
    yerr=1.96 * positive["SE"],
    capsize=4,
    label="Positive Movers"
)

plt.bar(
    horizons + width / 2,
    negative["Mean"],
    width=width,
    yerr=1.96 * negative["SE"],
    capsize=4,
    label="Negative Movers"
)

plt.axhline(
    0,
    linewidth=1
)

plt.xticks(
    horizons,
    ["1D", "3D", "5D"]
)

plt.ylabel(
    "High AVOL − Low AVOL Future Return (bps)"
)

plt.xlabel(
    "Forward Return Horizon"
)

plt.title(
    "Development Sample: Asymmetric Abnormal-Volume Effect"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FINAL_DIR
    / "figure2_positive_negative_asymmetry.png",
    dpi=300
)

plt.close()


# ============================================================
# 8. FIGURE 3
#
# Negative 5D robustness
# ============================================================

rob5 = robustness_negative[
    robustness_negative[
        "horizon"
    ] == 5
].copy()

rob5[
    "Specification"
] = rob5["spec"].map(
    spec_names
)

rob5[
    "Spread_bps"
] = (
    rob5["spread_pct"]
    * 100
)


plt.figure(
    figsize=(9, 5)
)

plt.bar(
    rob5["Specification"],
    rob5["Spread_bps"]
)

plt.axhline(
    0,
    linewidth=1
)

plt.ylabel(
    "5D High AVOL − Low AVOL Return (bps)"
)

plt.xlabel(
    "Specification"
)

plt.title(
    "Negative Movers: 5-Day Reversal Across Robustness Specifications"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FINAL_DIR
    / "figure3_negative_5d_robustness.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. Print important final numbers
# ============================================================

print("\nDONE")
print("Final outputs saved to:", FINAL_DIR)

print(
    "\nFiles created:"
)

for file in sorted(
    FINAL_DIR.iterdir()
):
    print(" -", file.name)


print(
    "\nFINAL NEGATIVE-MOVER MAIN RESULTS"
)

print(
    main_table[
        main_table["Mover"]
        == "Negative"
    ][
        [
            "Sample",
            "Horizon",
            "High-Low Spread (bps)",
            "HAC t-stat",
            "Adjusted Spread (bps)",
            "Adjusted HAC t-stat"
        ]
    ].round(2)
)