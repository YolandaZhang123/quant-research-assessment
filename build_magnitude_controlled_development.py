from pathlib import Path
from collections import defaultdict, deque
import json
import pandas as pd
import numpy as np


# ============================================================
# SETTINGS — fixed before seeing controlled future performance
# ============================================================

DAILY_DIR = Path("data/raw/daily")
UNIVERSE_DIR = Path("data/raw/monthly_universe")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEVELOPMENT_START = pd.Timestamp("2015-02-01")
DEVELOPMENT_END = pd.Timestamp("2025-01-24")

# We only need data through Jan 31 to evaluate 5D outcomes
PROCESS_END = pd.Timestamp("2025-01-31")

PRICE_MIN = 5.0
ADV20_MIN = 1_000_000
LOOKBACK = 20

HORIZONS = [1, 3, 5]


# Return-rank bins chosen BEFORE controlled performance inspection
POS_BINS = [
    0.80, 0.81, 0.82, 0.83, 0.84,
    0.85, 0.86, 0.87, 0.88, 0.89,
    0.90, 0.91, 0.92, 0.93, 0.94,
    0.95, 0.96, 0.97, 0.98, 0.99,
    1.00
]

POS_LABELS = [
    f"{i}-{i+1}"
    for i in range(80, 100)
]

NEG_BINS = [
    0.00, 0.01, 0.02, 0.03, 0.04,
    0.05, 0.06, 0.07, 0.08, 0.09,
    0.10, 0.11, 0.12, 0.13, 0.14,
    0.15, 0.16, 0.17, 0.18, 0.19,
    0.20
]

NEG_LABELS = [
    f"{i}-{i+1}"
    for i in range(0, 20)
]


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

print(
    "Monthly universes loaded:",
    len(monthly_universes)
)


# ============================================================
# 2. Historical state
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

balance_rows = []
future_rows = []

day_index = -1


# ============================================================
# 3. Create magnitude-controlled AVOL groups
# ============================================================

def build_controlled_groups(day_df, mover):

    if mover == "positive":
        bins = POS_BINS
        labels = POS_LABELS

        group = day_df[
            day_df["return_rank"] >= 0.80
        ].copy()

    else:
        bins = NEG_BINS
        labels = NEG_LABELS

        group = day_df[
            day_df["return_rank"] <= 0.20
        ].copy()

    group["magnitude_bin"] = pd.cut(
        group["return_rank"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True
    )

    bin_records = {}

    low_signal_returns = []
    high_signal_returns = []

    low_ranks = []
    high_ranks = []

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

        # Equal-weight the four magnitude bins
        low_signal_returns.append(
            low["daily_return"].mean()
        )

        high_signal_returns.append(
            high["daily_return"].mean()
        )

        low_ranks.append(
            low["return_rank"].mean()
        )

        high_ranks.append(
            high["return_rank"].mean()
        )

    # Require all four bins so each date is comparable
    if len(bin_records) != 20:
        return None

    balance = {
        "low_signal_return":
            np.mean(low_signal_returns),

        "high_signal_return":
            np.mean(high_signal_returns),

        "low_rank":
            np.mean(low_ranks),

        "high_rank":
            np.mean(high_ranks)
    }

    return {
        "bins": bin_records,
        "balance": balance
    }


# ============================================================
# 4. Future return helper
# ============================================================

def get_future_returns(
    records,
    current_close
):

    returns = []

    for ticker, signal_close in records:

        future_close = current_close.get(
            ticker
        )

        if future_close is None:
            continue

        returns.append(
            future_close / signal_close - 1
        )

    return returns


# ============================================================
# 5. Process trading days
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

    rows = data.get("results", [])

    if not rows:
        continue

    day_index += 1


    # ========================================================
    # Current closes
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
            current_close[ticker] = close


    # ========================================================
    # 6. Evaluate previous signals at 1D / 3D / 5D
    # ========================================================

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

            bin_spreads = []
            bin_low_returns = []
            bin_high_returns = []

            for (
                bin_name,
                records
            ) in mover_data[
                "bins"
            ].items():

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
                    continue

                low_mean = np.mean(
                    low_returns
                )

                high_mean = np.mean(
                    high_returns
                )

                bin_low_returns.append(
                    low_mean
                )

                bin_high_returns.append(
                    high_mean
                )

                bin_spreads.append(
                    high_mean - low_mean
                )

            # Require all four magnitude bins
            if len(bin_spreads) != 20:
                continue

            future_rows.append({
                "signal_date":
                    cohort["date"],

                "mover":
                    mover,

                "horizon":
                    horizon,

                # Equal-weight across the 4
                # return-magnitude bins
                "low_avol_return":
                    np.mean(
                        bin_low_returns
                    ),

                "high_avol_return":
                    np.mean(
                        bin_high_returns
                    ),

                "spread":
                    np.mean(
                        bin_spreads
                    ),

                # Keep balance information
                # for transparency
                "low_signal_return":
                    mover_data[
                        "balance"
                    ][
                        "low_signal_return"
                    ],

                "high_signal_return":
                    mover_data[
                        "balance"
                    ][
                        "high_signal_return"
                    ]
            })


    # ========================================================
    # 7. Build today's eligible universe
    # ========================================================

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

        if (
            len(day_history[ticker])
            < LOOKBACK
        ):
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

        daily_return = (
            close / prev_close[ticker]
            - 1
        )

        avol = (
            volume / avg_volume
        )

        eligible.append({
            "ticker": ticker,
            "close": close,
            "daily_return":
                daily_return,
            "avol":
                avol
        })


    # ========================================================
    # 8. Form development signal
    # ========================================================

    if (
        DEVELOPMENT_START
        <= date
        <= DEVELOPMENT_END
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

        positive = (
            build_controlled_groups(
                day_df,
                "positive"
            )
        )

        negative = (
            build_controlled_groups(
                day_df,
                "negative"
            )
        )

        if (
            positive is not None
            and negative is not None
        ):

            signal_cohorts[
                day_index
            ] = {
                "date": date_str,
                "positive": positive,
                "negative": negative
            }

            for mover, result in [
                ("positive", positive),
                ("negative", negative)
            ]:

                b = result["balance"]

                balance_rows.append({
                    "date": date_str,
                    "mover": mover,

                    "low_signal_return":
                        b[
                            "low_signal_return"
                        ],

                    "high_signal_return":
                        b[
                            "high_signal_return"
                        ],

                    "signal_return_difference":
                        b[
                            "high_signal_return"
                        ]
                        -
                        b[
                            "low_signal_return"
                        ],

                    "low_rank":
                        b["low_rank"],

                    "high_rank":
                        b["high_rank"]
                })


    # ========================================================
    # 9. Update historical data AFTER signal construction
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


    # Old cohorts no longer needed
    old_index = day_index - 6

    if old_index in signal_cohorts:
        del signal_cohorts[
            old_index
        ]


# ============================================================
# 10. Save results
# ============================================================

balance_df = pd.DataFrame(
    balance_rows
)

future_df = pd.DataFrame(
    future_rows
)

balance_df.to_csv(
    RESULTS_DIR
    / "magnitude_control_balance.csv",
    index=False
)

future_df.to_csv(
    RESULTS_DIR
    / "magnitude_controlled_development.csv",
    index=False
)


# ============================================================
# 11. PRINT ONLY THE BALANCE CHECK
# ============================================================

print("\nDONE")

print(
    "Controlled signal dates:",
    balance_df["date"].nunique()
)

for mover in [
    "positive",
    "negative"
]:

    x = balance_df[
        balance_df["mover"] == mover
    ]

    print(
        "\n================================"
    )

    print(
        mover.upper(),
        "MOVERS"
    )

    print(
        "================================"
    )

    print(
        "Signal days:",
        len(x)
    )

    print(
        "Low AVOL signal-day return:",
        round(
            x[
                "low_signal_return"
            ].mean() * 100,
            4
        ),
        "%"
    )

    print(
        "High AVOL signal-day return:",
        round(
            x[
                "high_signal_return"
            ].mean() * 100,
            4
        ),
        "%"
    )

    print(
        "High - Low signal-day return:",
        round(
            x[
                "signal_return_difference"
            ].mean() * 100,
            4
        ),
        "%"
    )

    print(
        "Low AVOL mean rank:",
        round(
            x["low_rank"].mean(),
            4
        )
    )

    print(
        "High AVOL mean rank:",
        round(
            x["high_rank"].mean(),
            4
        )
    )


print(
    "\nFuture-return results were saved "
    "but intentionally not printed yet."
)