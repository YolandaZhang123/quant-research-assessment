from pathlib import Path
from collections import defaultdict, deque
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm


# ============================================================
# SETTINGS
# ============================================================

DAILY_DIR = Path("data/raw/daily")
UNIVERSE_DIR = Path("data/raw/monthly_universe")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEVELOPMENT_START = pd.Timestamp("2015-02-01")
DEVELOPMENT_END = pd.Timestamp("2025-01-24")
PROCESS_END = pd.Timestamp("2025-01-31")

PRICE_MIN = 5.0
HORIZONS = [1, 3, 5]

# One-at-a-time robustness variations
SPECS = {
    "MAIN_20D_1M_TAIL20": {
        "avol_lookback": 20,
        "liquidity_min": 1_000_000,
        "tail": 0.20
    },

    "AVOL10": {
        "avol_lookback": 10,
        "liquidity_min": 1_000_000,
        "tail": 0.20
    },

    "AVOL40": {
        "avol_lookback": 40,
        "liquidity_min": 1_000_000,
        "tail": 0.20
    },

    "LIQ5M": {
        "avol_lookback": 20,
        "liquidity_min": 5_000_000,
        "tail": 0.20
    },

    "LIQ10M": {
        "avol_lookback": 20,
        "liquidity_min": 10_000_000,
        "tail": 0.20
    },

    "TAIL10": {
        "avol_lookback": 20,
        "liquidity_min": 1_000_000,
        "tail": 0.10
    }
}


# ============================================================
# 1. Load monthly universes
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

print(
    "Monthly universes loaded:",
    len(monthly_universes)
)


# ============================================================
# 2. Historical state
#
# Keep up to 40 days because 40D is largest AVOL lookback.
# ============================================================

MAX_LOOKBACK = 40

volume_history = defaultdict(
    lambda: deque(maxlen=MAX_LOOKBACK)
)

dollar_volume_history = defaultdict(
    lambda: deque(maxlen=MAX_LOOKBACK)
)

day_history = defaultdict(
    lambda: deque(maxlen=MAX_LOOKBACK)
)

prev_close = {}
prev_day_index = {}

signal_cohorts = {}

future_rows = []

day_index = -1
valid_days_processed = 0


# ============================================================
# 3. Check whether ticker has N consecutive prior days
# ============================================================

def has_consecutive_history(
    ticker,
    lookback,
    current_day_index
):

    days = list(
        day_history[ticker]
    )

    if len(days) < lookback:
        return False

    recent = days[-lookback:]

    return (
        recent[0]
        == current_day_index - lookback
        and recent[-1]
        == current_day_index - 1
        and len(recent) == lookback
    )


# ============================================================
# 4. Create 1-percentile magnitude bins
# ============================================================

def build_groups(
    day_df,
    mover,
    tail,
    avol_column
):

    if mover == "positive":

        lower = 1.0 - tail
        upper = 1.0

        group = day_df[
            day_df["return_rank"]
            >= lower
        ].copy()

    else:

        lower = 0.0
        upper = tail

        group = day_df[
            day_df["return_rank"]
            <= upper
        ].copy()

    # 1-percentile bins
    edges = np.round(
        np.arange(
            lower,
            upper + 0.001,
            0.01
        ),
        2
    )

    labels = [
        f"{edges[i]:.2f}-{edges[i+1]:.2f}"
        for i in range(
            len(edges) - 1
        )
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
            group["magnitude_bin"]
            == label
        ].copy()

        if len(b) < 4:
            return None

        median_avol = b[
            avol_column
        ].median()

        low = b[
            b[avol_column]
            < median_avol
        ]

        high = b[
            b[avol_column]
            >= median_avol
        ]

        if (
            len(low) == 0
            or len(high) == 0
        ):
            return None

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
# 5. Future return helper
# ============================================================

def future_returns(
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

        if signal_close <= 0:
            continue

        out.append(
            future_close
            / signal_close
            - 1
        )

    return out


# ============================================================
# 6. Process daily files
# ============================================================

for file in sorted(
    DAILY_DIR.glob("daily_*.json")
):

    date_str = file.stem.replace(
        "daily_", ""
    )

    date = pd.Timestamp(date_str)

    if date > PROCESS_END:
        break

    with open(file, "r") as f:
        data = json.load(f)

    rows = data.get(
        "results", []
    )

    if not rows:
        continue

    day_index += 1
    valid_days_processed += 1

    if (
        valid_days_processed
        % 250 == 0
    ):
        print(
            "Processed through:",
            date_str
        )


    # ========================================================
    # Current closes for evaluating old signals
    # ========================================================

    current_close = {}

    for row in rows:

        ticker = row.get("T")
        close = row.get("c")

        if (
            ticker is not None
            and close is not None
            and close > 0
        ):
            current_close[
                ticker
            ] = close


    # ========================================================
    # 7. Evaluate previous cohorts
    # ========================================================

    for horizon in HORIZONS:

        source_index = (
            day_index - horizon
        )

        if (
            source_index
            not in signal_cohorts
        ):
            continue

        source = signal_cohorts[
            source_index
        ]

        for spec_name, cohort in (
            source.items()
        ):

            for mover in [
                "positive",
                "negative"
            ]:

                mover_data = cohort[
                    mover
                ]

                low_bin_returns = []
                high_bin_returns = []

                valid = True

                for records in (
                    mover_data["bins"]
                    .values()
                ):

                    low_returns = (
                        future_returns(
                            records["low"],
                            current_close
                        )
                    )

                    high_returns = (
                        future_returns(
                            records["high"],
                            current_close
                        )
                    )

                    if (
                        len(low_returns)
                        == 0
                        or len(high_returns)
                        == 0
                    ):
                        valid = False
                        break

                    low_bin_returns.append(
                        np.mean(
                            low_returns
                        )
                    )

                    high_bin_returns.append(
                        np.mean(
                            high_returns
                        )
                    )

                if not valid:
                    continue

                low_future = np.mean(
                    low_bin_returns
                )

                high_future = np.mean(
                    high_bin_returns
                )

                future_rows.append({
                    "spec":
                        spec_name,

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


    # ========================================================
    # 8. Build base eligible dataset
    # ========================================================

    month = date.strftime(
        "%Y-%m"
    )

    universe = monthly_universes.get(
        month,
        set()
    )

    base_rows = []

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

        # Previous trading-day close
        if (
            prev_day_index.get(
                ticker
            )
            != day_index - 1
        ):
            continue

        # Liquidity always uses prior 20D ADV
        if not has_consecutive_history(
            ticker,
            20,
            day_index
        ):
            continue

        dollar_hist = list(
            dollar_volume_history[
                ticker
            ]
        )

        adv20 = np.mean(
            dollar_hist[-20:]
        )

        daily_return = (
            close
            / prev_close[ticker]
            - 1
        )

        row_out = {
            "ticker":
                ticker,

            "close":
                close,

            "daily_return":
                daily_return,

            "adv20":
                adv20
        }

        # AVOL 10D
        if has_consecutive_history(
            ticker,
            10,
            day_index
        ):

            hist = list(
                volume_history[
                    ticker
                ]
            )

            avg10 = np.mean(
                hist[-10:]
            )

            if avg10 > 0:
                row_out[
                    "avol_10"
                ] = (
                    volume / avg10
                )

        # AVOL 20D
        hist = list(
            volume_history[
                ticker
            ]
        )

        avg20 = np.mean(
            hist[-20:]
        )

        if avg20 > 0:
            row_out[
                "avol_20"
            ] = volume / avg20

        # AVOL 40D
        if has_consecutive_history(
            ticker,
            40,
            day_index
        ):

            avg40 = np.mean(
                hist[-40:]
            )

            if avg40 > 0:
                row_out[
                    "avol_40"
                ] = (
                    volume / avg40
                )

        base_rows.append(
            row_out
        )


    # ========================================================
    # 9. Form signals for each robustness specification
    # ========================================================

    if (
        DEVELOPMENT_START
        <= date
        <= DEVELOPMENT_END
        and len(base_rows) > 0
    ):

        base_df = pd.DataFrame(
            base_rows
        )

        today_specs = {}

        for (
            spec_name,
            settings
        ) in SPECS.items():

            lookback = settings[
                "avol_lookback"
            ]

            liquidity_min = settings[
                "liquidity_min"
            ]

            tail = settings[
                "tail"
            ]

            avol_col = (
                f"avol_{lookback}"
            )

            if (
                avol_col
                not in base_df.columns
            ):
                continue

            spec_df = base_df[
                (
                    base_df["adv20"]
                    >= liquidity_min
                )
                &
                (
                    base_df[
                        avol_col
                    ].notna()
                )
            ].copy()

            if len(spec_df) == 0:
                continue

            # Rank AFTER applying this spec's
            # price/liquidity/history rules
            spec_df[
                "return_rank"
            ] = (
                spec_df[
                    "daily_return"
                ]
                .rank(
                    pct=True,
                    method="average"
                )
            )

            positive = build_groups(
                spec_df,
                "positive",
                tail,
                avol_col
            )

            negative = build_groups(
                spec_df,
                "negative",
                tail,
                avol_col
            )

            if (
                positive is None
                or negative is None
            ):
                continue

            today_specs[
                spec_name
            ] = {
                "date":
                    date_str,

                "positive":
                    positive,

                "negative":
                    negative
            }

        if today_specs:
            signal_cohorts[
                day_index
            ] = today_specs


    # ========================================================
    # 10. Update histories AFTER signal construction
    # ========================================================

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

        volume_history[
            ticker
        ].append(
            volume
        )

        dollar_volume_history[
            ticker
        ].append(
            close * volume
        )

        day_history[
            ticker
        ].append(
            day_index
        )

        prev_close[
            ticker
        ] = close

        prev_day_index[
            ticker
        ] = day_index


    old_index = (
        day_index - 6
    )

    if (
        old_index
        in signal_cohorts
    ):
        del signal_cohorts[
            old_index
        ]


# ============================================================
# 11. Save all daily robustness results
# ============================================================

results = pd.DataFrame(
    future_rows
)

results.to_csv(
    RESULTS_DIR
    / "development_robustness_daily.csv",
    index=False
)


# ============================================================
# 12. HAC summary
# ============================================================

def hac_test(
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


summary_rows = []

print("\nDONE")
print("=" * 70)

for spec_name in SPECS:

    print(
        "\n",
        spec_name
    )

    print(
        "-" * 70
    )

    for mover in [
        "positive",
        "negative"
    ]:

        for horizon in [
            1, 3, 5
        ]:

            x = results[
                (
                    results["spec"]
                    == spec_name
                )
                &
                (
                    results["mover"]
                    == mover
                )
                &
                (
                    results["horizon"]
                    == horizon
                )
            ]

            if len(x) == 0:
                continue

            mean, se, tstat = (
                hac_test(
                    x["spread"],
                    horizon - 1
                )
            )

            median = (
                x["spread"]
                .median()
            )

            signal_gap = (
                x[
                    "high_signal_return"
                ]
                -
                x[
                    "low_signal_return"
                ]
            ).mean()

            summary_rows.append({
                "spec":
                    spec_name,

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

                "hac_se_pct":
                    se * 100,

                "hac_t":
                    tstat,

                "signal_gap_pct":
                    signal_gap * 100
            })

            print(
                f"{mover:8s} "
                f"{horizon}D | "
                f"N={len(x):4d} | "
                f"Spread="
                f"{mean*100:+.4f}% | "
                f"HAC t="
                f"{tstat:+.3f} | "
                f"Signal gap="
                f"{signal_gap*100:+.4f}%"
            )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    RESULTS_DIR
    / "development_robustness_summary.csv",
    index=False
)

print(
    "\nSaved robustness summary."
)