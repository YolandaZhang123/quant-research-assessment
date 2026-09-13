from pathlib import Path
from collections import defaultdict, deque
import json
import pandas as pd
import numpy as np


# ============================================================
# SETTINGS — same as formal development study
# ============================================================

DAILY_DIR = Path("data/raw/daily")
UNIVERSE_DIR = Path("data/raw/monthly_universe")
RESULTS_DIR = Path("results")

DEVELOPMENT_START = pd.Timestamp("2015-02-01")
DEVELOPMENT_END = pd.Timestamp("2025-01-24")

PRICE_MIN = 5.0
ADV20_MIN = 1_000_000
LOOKBACK = 20

TOP_CUTOFF = 0.80
BOTTOM_CUTOFF = 0.20


# ============================================================
# 1. Load monthly historical universes
# ============================================================

monthly_universes = {}

for file in sorted(
    UNIVERSE_DIR.glob("universe_*.json")
):

    year_month = file.stem.replace(
        "universe_", ""
    )

    with open(file, "r") as f:
        data = json.load(f)

    monthly_universes[
        year_month
    ] = set(data["tickers"])


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

diagnostic_rows = []

day_index = -1


# ============================================================
# 3. Walk through trading days
# ============================================================

for file in sorted(
    DAILY_DIR.glob("daily_*.json")
):

    date_str = file.stem.replace(
        "daily_", ""
    )

    date = pd.Timestamp(date_str)

    with open(file, "r") as f:
        data = json.load(f)

    rows = data.get("results", [])

    if not rows:
        continue

    day_index += 1

    year_month = date.strftime("%Y-%m")

    universe = monthly_universes.get(
        year_month,
        set()
    )

    eligible = []

    # ========================================================
    # 4. Build today's eligible universe
    # ========================================================

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

        # Need yesterday's close
        if (
            prev_day_index.get(ticker)
            != day_index - 1
        ):
            continue

        # Need 20 consecutive prior observations
        if len(day_history[ticker]) < LOOKBACK:
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
                dollar_volume_history[ticker]
            )
            / LOOKBACK
        )

        if (
            avg_volume <= 0
            or adv20 < ADV20_MIN
        ):
            continue

        daily_return = (
            close / prev_close[ticker] - 1
        )

        avol = volume / avg_volume

        eligible.append({
            "ticker": ticker,
            "daily_return": daily_return,
            "avol": avol
        })


    # ========================================================
    # 5. Analyze development sample only
    # ========================================================

    if (
        DEVELOPMENT_START
        <= date
        <= DEVELOPMENT_END
        and len(eligible) > 0
    ):

        df = pd.DataFrame(eligible)

        df["return_rank"] = (
            df["daily_return"]
            .rank(
                pct=True,
                method="average"
            )
        )

        groups = {
            "positive": df[
                df["return_rank"]
                >= TOP_CUTOFF
            ].copy(),

            "negative": df[
                df["return_rank"]
                <= BOTTOM_CUTOFF
            ].copy()
        }

        for mover, group in groups.items():

            if len(group) == 0:
                continue

            median_avol = (
                group["avol"].median()
            )

            low = group[
                group["avol"]
                < median_avol
            ]

            high = group[
                group["avol"]
                >= median_avol
            ]

            if (
                len(low) == 0
                or len(high) == 0
            ):
                continue

            diagnostic_rows.append({
                "date": date_str,
                "mover": mover,

                "low_signal_return":
                    low[
                        "daily_return"
                    ].mean(),

                "high_signal_return":
                    high[
                        "daily_return"
                    ].mean(),

                "high_minus_low_signal_return":
                    high[
                        "daily_return"
                    ].mean()
                    -
                    low[
                        "daily_return"
                    ].mean(),

                "low_abs_return":
                    low[
                        "daily_return"
                    ].abs().mean(),

                "high_abs_return":
                    high[
                        "daily_return"
                    ].abs().mean(),

                "low_mean_rank":
                    low[
                        "return_rank"
                    ].mean(),

                "high_mean_rank":
                    high[
                        "return_rank"
                    ].mean(),

                "n_low":
                    len(low),

                "n_high":
                    len(high)
            })


    # ========================================================
    # 6. Update history AFTER today's signal construction
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

        dollar_volume_history[ticker].append(
            close * volume
        )

        day_history[ticker].append(
            day_index
        )

        prev_close[ticker] = close
        prev_day_index[ticker] = day_index


# ============================================================
# 7. Save daily diagnostic
# ============================================================

diag = pd.DataFrame(
    diagnostic_rows
)

diag.to_csv(
    RESULTS_DIR
    / "move_magnitude_diagnostic.csv",
    index=False
)


# ============================================================
# 8. Summarize
# ============================================================

print("\nDONE")

for mover in [
    "positive",
    "negative"
]:

    x = diag[
        diag["mover"] == mover
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
                "high_minus_low_signal_return"
            ].mean() * 100,
            4
        ),
        "%"
    )

    print(
        "Low AVOL absolute move:",
        round(
            x[
                "low_abs_return"
            ].mean() * 100,
            4
        ),
        "%"
    )

    print(
        "High AVOL absolute move:",
        round(
            x[
                "high_abs_return"
            ].mean() * 100,
            4
        ),
        "%"
    )

    print(
        "Low AVOL mean return rank:",
        round(
            x[
                "low_mean_rank"
            ].mean(),
            4
        )
    )

    print(
        "High AVOL mean return rank:",
        round(
            x[
                "high_mean_rank"
            ].mean(),
            4
        )
    )

print(
    "\nSaved:",
    RESULTS_DIR
    / "move_magnitude_diagnostic.csv"
)