from pathlib import Path
from collections import defaultdict, deque
import json
import pandas as pd
import numpy as np

# ============================================================
# SETTINGS — locked before formal performance inspection
# ============================================================

DAILY_DIR = Path("data/raw/daily")
UNIVERSE_DIR = Path("data/raw/monthly_universe")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEVELOPMENT_START = pd.Timestamp("2015-02-01")
DEVELOPMENT_END = pd.Timestamp("2025-01-24")

PRICE_MIN = 5.0
ADV20_MIN = 1_000_000
LOOKBACK = 20

TOP_CUTOFF = 0.80
BOTTOM_CUTOFF = 0.20

HORIZONS = [1, 3, 5]


# ============================================================
# 1. Load monthly point-in-time common-stock universes
# ============================================================

monthly_universes = {}

for file in sorted(UNIVERSE_DIR.glob("universe_*.json")):

    year_month = file.stem.replace("universe_", "")

    with open(file, "r") as f:
        data = json.load(f)

    monthly_universes[year_month] = set(data["tickers"])

print("Monthly universes loaded:", len(monthly_universes))


# ============================================================
# 2. Historical state for each ticker
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

# Signal cohorts only need to stay alive for 5 trading days
signal_cohorts = {}

results = []

eligible_counts = []

day_index = -1


# ============================================================
# Helper: split mover group into Low / High AVOL
# ============================================================

def build_volume_groups(group):

    median_avol = group["avol"].median()

    low = group[
        group["avol"] < median_avol
    ]

    high = group[
        group["avol"] >= median_avol
    ]

    return {
        "low": list(
            zip(low["ticker"], low["close"])
        ),
        "high": list(
            zip(high["ticker"], high["close"])
        )
    }


# ============================================================
# Helper: calculate future returns for one group
# ============================================================

def future_group_returns(records, current_close):

    returns = []

    for ticker, signal_close in records:

        future_close = current_close.get(ticker)

        if (
            future_close is None
            or signal_close is None
            or signal_close <= 0
        ):
            continue

        returns.append(
            future_close / signal_close - 1
        )

    return returns


# ============================================================
# 3. Walk through every valid trading day
# ============================================================

for file in sorted(DAILY_DIR.glob("daily_*.json")):

    date_str = file.stem.replace("daily_", "")
    date = pd.Timestamp(date_str)

    with open(file, "r") as f:
        data = json.load(f)

    rows = data.get("results", [])

    # Ignore holidays / closures
    if not rows:
        continue

    day_index += 1

    # All available closes today.
    # Used to evaluate old signal cohorts.
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
    # 4. Evaluate signals created 1 / 3 / 5 trading days ago
    # ========================================================

    for horizon in HORIZONS:

        source_index = day_index - horizon

        if source_index not in signal_cohorts:
            continue

        cohort = signal_cohorts[source_index]

        for mover in ["positive", "negative"]:

            low_returns = future_group_returns(
                cohort[mover]["low"],
                current_close
            )

            high_returns = future_group_returns(
                cohort[mover]["high"],
                current_close
            )

            if (
                len(low_returns) == 0
                or len(high_returns) == 0
            ):
                continue

            low_avg = np.mean(low_returns)
            high_avg = np.mean(high_returns)

            results.append({
                "signal_date": cohort["date"],
                "horizon": horizon,
                "mover": mover,

                "low_avol_return": low_avg,
                "high_avol_return": high_avg,

                "spread": (
                    high_avg - low_avg
                ),

                "n_low": len(low_returns),
                "n_high": len(high_returns)
            })


    # ========================================================
    # 5. Construct today's eligible research universe
    # ========================================================

    year_month = date.strftime("%Y-%m")

    universe = monthly_universes.get(
        year_month,
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

        # Current signal-day price filter
        if close < PRICE_MIN:
            continue

        # Need immediately previous trading-day close
        if prev_day_index.get(ticker) != day_index - 1:
            continue

        # Need exactly 20 consecutive PRIOR trading days
        if len(day_history[ticker]) < LOOKBACK:
            continue

        if (
            day_history[ticker][0] != day_index - LOOKBACK
            or day_history[ticker][-1] != day_index - 1
        ):
            continue

        average_volume = (
            sum(volume_history[ticker])
            / LOOKBACK
        )

        adv20 = (
            sum(dollar_volume_history[ticker])
            / LOOKBACK
        )

        if average_volume <= 0:
            continue

        # Pre-specified liquidity filter
        if adv20 < ADV20_MIN:
            continue

        daily_return = (
            close / prev_close[ticker] - 1
        )

        avol = (
            volume / average_volume
        )

        eligible.append({
            "ticker": ticker,
            "close": close,
            "daily_return": daily_return,
            "avol": avol
        })


    # ========================================================
    # 6. Form a signal only inside DEVELOPMENT period
    # ========================================================

    if (
        DEVELOPMENT_START
        <= date
        <= DEVELOPMENT_END
        and len(eligible) > 0
    ):

        day_df = pd.DataFrame(eligible)

        day_df["return_rank"] = (
            day_df["daily_return"]
            .rank(
                pct=True,
                method="average"
            )
        )

        positive = day_df[
            day_df["return_rank"]
            >= TOP_CUTOFF
        ].copy()

        negative = day_df[
            day_df["return_rank"]
            <= BOTTOM_CUTOFF
        ].copy()

        if (
            len(positive) > 0
            and len(negative) > 0
        ):

            signal_cohorts[day_index] = {
                "date": date_str,

                "positive":
                    build_volume_groups(
                        positive
                    ),

                "negative":
                    build_volume_groups(
                        negative
                    )
            }

            eligible_counts.append(
                len(day_df)
            )


    # ========================================================
    # 7. AFTER using today's signal:
    #    update history with today's observations
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


    # Remove cohorts too old to ever be needed again
    old_index = day_index - 6

    if old_index in signal_cohorts:
        del signal_cohorts[old_index]


# ============================================================
# 8. Save formal DEVELOPMENT results
# ============================================================

result_df = pd.DataFrame(results)

output_file = (
    RESULTS_DIR
    / "formal_development_daily_spreads.csv"
)

result_df.to_csv(
    output_file,
    index=False
)


print("\nDONE")

print(
    "Signal dates:",
    len(eligible_counts)
)

if eligible_counts:

    print(
        "Average eligible stocks per signal date:",
        round(np.mean(eligible_counts), 1)
    )

    print(
        "Minimum eligible stocks:",
        min(eligible_counts)
    )

    print(
        "Maximum eligible stocks:",
        max(eligible_counts)
    )

print(
    "Daily spread observations:",
    len(result_df)
)

print(
    "Saved:",
    output_file
)