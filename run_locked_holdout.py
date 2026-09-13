from pathlib import Path
from collections import defaultdict, deque
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm


# ============================================================
# LOCKED HOLDOUT SPECIFICATION
# ============================================================

DAILY_DIR = Path("data/raw/daily")
UNIVERSE_DIR = Path("data/raw/monthly_universe")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

PROCESS_START = pd.Timestamp("2024-12-01")
PROCESS_END = pd.Timestamp("2025-12-31")

HOLDOUT_START = pd.Timestamp("2025-02-01")

PRICE_MIN = 5.0
ADV20_MIN = 1_000_000
LOOKBACK = 20

HORIZONS = [1, 3, 5]


# ============================================================
# 1. Load monthly point-in-time universes
# ============================================================

monthly_universes = {}

for file in sorted(
    UNIVERSE_DIR.glob("universe_*.json")
):
    month = file.stem.replace(
        "universe_", ""
    )

    with open(file, "r") as f:
        data = json.load(f)

    monthly_universes[month] = set(
        data["tickers"]
    )


# ============================================================
# 2. Determine final signal date automatically
#
# Require 5 future valid trading days before Dec 31.
# ============================================================

valid_2025_dates = []

for file in sorted(
    DAILY_DIR.glob("daily_2025-*.json")
):

    date_str = file.stem.replace(
        "daily_", ""
    )

    with open(file, "r") as f:
        data = json.load(f)

    if data.get("results", []):
        valid_2025_dates.append(
            pd.Timestamp(date_str)
        )

HOLDOUT_END = valid_2025_dates[-6]

print(
    "Locked holdout signal period:",
    HOLDOUT_START.date(),
    "to",
    HOLDOUT_END.date()
)


# ============================================================
# 3. Historical state
# ============================================================

volume_history = defaultdict(
    lambda: deque(maxlen=LOOKBACK)
)

dollar_volume_history = defaultdict(
    lambda: deque(maxlen=LOOKBACK)
)

day_history = defaultdict(
    lambda: deque(maxlen=LOOKBACK)
)

prev_close = {}
prev_day_index = {}

signal_cohorts = {}
future_rows = []

day_index = -1


# ============================================================
# 4. Build locked 1%-bin groups
# ============================================================

POS_BINS = np.round(
    np.arange(0.80, 1.001, 0.01),
    2
)

NEG_BINS = np.round(
    np.arange(0.00, 0.201, 0.01),
    2
)


def build_groups(day_df, mover):

    if mover == "positive":
        group = day_df[
            day_df["return_rank"] >= 0.80
        ].copy()

        edges = POS_BINS

    else:
        group = day_df[
            day_df["return_rank"] <= 0.20
        ].copy()

        edges = NEG_BINS

    labels = [
        f"{edges[i]:.2f}-{edges[i+1]:.2f}"
        for i in range(len(edges) - 1)
    ]

    group["magnitude_bin"] = pd.cut(
        group["return_rank"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True
    )

    bin_records = {}

    low_signal_returns = []
    high_signal_returns = []

    for label in labels:

        b = group[
            group["magnitude_bin"] == label
        ].copy()

        if len(b) == 0:
            continue

        median_avol = b["avol"].median()

        low = b[
            b["avol"] < median_avol
        ]

        high = b[
            b["avol"] >= median_avol
        ]

        if (
            len(low) == 0
            or len(high) == 0
        ):
            continue

        bin_records[label] = {
            "low": list(
                zip(
                    low["ticker"],
                    low["close"]
                )
            ),

            "high": list(
                zip(
                    high["ticker"],
                    high["close"]
                )
            )
        }

        low_signal_returns.append(
            low["daily_return"].mean()
        )

        high_signal_returns.append(
            high["daily_return"].mean()
        )

    # Locked main method requires all 20 bins
    if len(bin_records) != 20:
        return None

    return {
        "bins": bin_records,

        "low_signal_return":
            np.mean(
                low_signal_returns
            ),

        "high_signal_return":
            np.mean(
                high_signal_returns
            )
    }


# ============================================================
# 5. Future-return helper
# ============================================================

def get_future_returns(
    records,
    current_close
):

    out = []

    for ticker, signal_close in records:

        future_close = current_close.get(
            ticker
        )

        if future_close is None:
            continue

        out.append(
            future_close
            / signal_close
            - 1
        )

    return out


# ============================================================
# 6. Process data
# ============================================================

for file in sorted(
    DAILY_DIR.glob("daily_*.json")
):

    date_str = file.stem.replace(
        "daily_", ""
    )

    date = pd.Timestamp(date_str)

    if date < PROCESS_START:
        continue

    if date > PROCESS_END:
        break

    with open(file, "r") as f:
        data = json.load(f)

    rows = data.get("results", [])

    if not rows:
        continue

    day_index += 1


    # --------------------------------------------------------
    # Today's closes
    # --------------------------------------------------------

    current_close = {}

    for row in rows:

        ticker = row.get("T")
        close = row.get("c")

        if (
            ticker is not None
            and close is not None
            and close > 0
        ):
            current_close[ticker] = close


    # --------------------------------------------------------
    # Evaluate previous locked signals
    # --------------------------------------------------------

    for horizon in HORIZONS:

        source_index = (
            day_index - horizon
        )

        if source_index not in signal_cohorts:
            continue

        cohort = signal_cohorts[
            source_index
        ]

        for mover in [
            "positive",
            "negative"
        ]:

            mover_data = cohort[mover]

            low_bins = []
            high_bins = []

            valid = True

            for records in (
                mover_data["bins"].values()
            ):

                low_returns = (
                    get_future_returns(
                        records["low"],
                        current_close
                    )
                )

                high_returns = (
                    get_future_returns(
                        records["high"],
                        current_close
                    )
                )

                if (
                    len(low_returns) == 0
                    or len(high_returns) == 0
                ):
                    valid = False
                    break

                low_bins.append(
                    np.mean(low_returns)
                )

                high_bins.append(
                    np.mean(high_returns)
                )

            if not valid:
                continue

            low_future = np.mean(
                low_bins
            )

            high_future = np.mean(
                high_bins
            )

            future_rows.append({
                "signal_date":
                    cohort["date"],

                "mover":
                    mover,

                "horizon":
                    horizon,

                "low_avol_return":
                    low_future,

                "high_avol_return":
                    high_future,

                "spread":
                    high_future
                    - low_future,

                "low_signal_return":
                    mover_data[
                        "low_signal_return"
                    ],

                "high_signal_return":
                    mover_data[
                        "high_signal_return"
                    ]
            })


    # --------------------------------------------------------
    # Construct today's eligible universe
    # --------------------------------------------------------

    month = date.strftime("%Y-%m")

    universe = monthly_universes.get(
        month,
        set()
    )

    eligible = []

    for row in rows:

        ticker = row.get("T")
        close = row.get("c")
        volume = row.get("v")

        if ticker not in universe:
            continue

        if (
            close is None
            or volume is None
            or close <= 0
            or volume <= 0
        ):
            continue

        if close < PRICE_MIN:
            continue

        if (
            prev_day_index.get(ticker)
            != day_index - 1
        ):
            continue

        if len(
            day_history[ticker]
        ) < LOOKBACK:
            continue

        if (
            day_history[ticker][0]
            != day_index - LOOKBACK
            or day_history[ticker][-1]
            != day_index - 1
        ):
            continue

        avg_volume = (
            sum(volume_history[ticker])
            / LOOKBACK
        )

        adv20 = (
            sum(
                dollar_volume_history[
                    ticker
                ]
            )
            / LOOKBACK
        )

        if (
            avg_volume <= 0
            or adv20 < ADV20_MIN
        ):
            continue

        eligible.append({
            "ticker":
                ticker,

            "close":
                close,

            "daily_return":
                close
                / prev_close[ticker]
                - 1,

            "avol":
                volume
                / avg_volume
        })


    # --------------------------------------------------------
    # Form LOCKED holdout signals only
    # --------------------------------------------------------

    if (
        HOLDOUT_START
        <= date
        <= HOLDOUT_END
        and len(eligible) > 0
    ):

        day_df = pd.DataFrame(
            eligible
        )

        day_df["return_rank"] = (
            day_df["daily_return"]
            .rank(
                pct=True,
                method="average"
            )
        )

        positive = build_groups(
            day_df,
            "positive"
        )

        negative = build_groups(
            day_df,
            "negative"
        )

        if (
            positive is not None
            and negative is not None
        ):

            signal_cohorts[
                day_index
            ] = {
                "date":
                    date_str,

                "positive":
                    positive,

                "negative":
                    negative
            }


    # --------------------------------------------------------
    # Update history AFTER signal
    # --------------------------------------------------------

    for row in rows:

        ticker = row.get("T")
        close = row.get("c")
        volume = row.get("v")

        if (
            ticker is None
            or close is None
            or volume is None
            or close <= 0
            or volume <= 0
        ):
            continue

        volume_history[ticker].append(
            volume
        )

        dollar_volume_history[
            ticker
        ].append(
            close * volume
        )

        day_history[ticker].append(
            day_index
        )

        prev_close[ticker] = close
        prev_day_index[ticker] = (
            day_index
        )


    old_index = (
        day_index - 6
    )

    if old_index in signal_cohorts:
        del signal_cohorts[
            old_index
        ]


# ============================================================
# 7. Save untouched holdout daily results
# ============================================================

df = pd.DataFrame(
    future_rows
)

df.to_csv(
    RESULTS_DIR
    / "locked_holdout_daily.csv",
    index=False
)


# ============================================================
# 8. HAC evaluation
# ============================================================

def hac_mean_test(
    series,
    maxlags
):

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
            "maxlags":
                maxlags
        }
    )

    return (
        model.params[0],
        model.bse[0],
        model.tvalues[0]
    )


print("\nLOCKED HOLDOUT RESULTS")
print("=" * 60)

print(
    "Signal dates:",
    df["signal_date"].nunique()
)

summary_rows = []

for mover in [
    "positive",
    "negative"
]:

    print(
        "\n",
        mover.upper(),
        "MOVERS"
    )

    for horizon in [1, 3, 5]:

        x = df[
            (df["mover"] == mover)
            &
            (df["horizon"] == horizon)
        ].copy()

        spread = x["spread"]

        mean, se, tstat = (
            hac_mean_test(
                spread,
                horizon - 1
            )
        )

        median = (
            spread.median()
        )

        sign_positive = (
            (spread > 0).mean()
            * 100
        )

        signal_gap = (
            x[
                "high_signal_return"
            ]
            -
            x[
                "low_signal_return"
            ]
        )

        # Continuous exact-return control
        X = sm.add_constant(
            signal_gap
        )

        control_model = sm.OLS(
            spread,
            X
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags":
                    horizon - 1
            }
        )

        adjusted = (
            control_model.params[
                "const"
            ]
        )

        adjusted_t = (
            control_model.tvalues[
                "const"
            ]
        )

        print(
            f"\n{horizon}D"
        )

        print(
            "N:",
            len(x)
        )

        print(
            "High - Low spread:",
            round(
                mean * 100,
                4
            ),
            "%"
        )

        print(
            "Median spread:",
            round(
                median * 100,
                4
            ),
            "%"
        )

        print(
            "HAC t-stat:",
            round(
                tstat,
                3
            )
        )

        print(
            "Positive-spread days:",
            round(
                sign_positive,
                1
            ),
            "%"
        )

        print(
            "Signal-day gap:",
            round(
                signal_gap.mean()
                * 100,
                4
            ),
            "%"
        )

        print(
            "Adjusted spread at gap = 0:",
            round(
                adjusted * 100,
                4
            ),
            "%"
        )

        print(
            "Adjusted HAC t-stat:",
            round(
                adjusted_t,
                3
            )
        )

        summary_rows.append({
            "mover":
                mover,

            "horizon":
                horizon,

            "n_days":
                len(x),

            "spread_pct":
                mean * 100,

            "median_pct":
                median * 100,

            "hac_t":
                tstat,

            "positive_days_pct":
                sign_positive,

            "signal_gap_pct":
                signal_gap.mean()
                * 100,

            "adjusted_spread_pct":
                adjusted * 100,

            "adjusted_hac_t":
                adjusted_t
        })


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    RESULTS_DIR
    / "locked_holdout_summary.csv",
    index=False
)

print(
    "\nSaved locked holdout results."
)