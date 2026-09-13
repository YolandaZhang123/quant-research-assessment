from pathlib import Path
from collections import defaultdict, deque
import json
import numpy as np

DAILY_DIR = Path("data/raw/daily")
UNIVERSE_DIR = Path("data/raw/monthly_universe")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------
# 1. Load monthly point-in-time universes
# ------------------------------------------------

monthly_universes = {}
snapshot_dates = {}

for file in sorted(UNIVERSE_DIR.glob("universe_*.json")):
    year_month = file.stem.replace("universe_", "")

    with open(file, "r") as f:
        data = json.load(f)

    monthly_universes[year_month] = set(data["tickers"])
    snapshot_dates[data["reference_date"]] = year_month

print("Monthly universes loaded:", len(monthly_universes))


# ------------------------------------------------
# 2. Keep trailing 20 DAILY dollar-volume observations
#    for each ticker
# ------------------------------------------------

history = defaultdict(lambda: deque(maxlen=20))

adv_values_all = []
adv_values_price5 = []

snapshot_observations = 0
snapshot_with_20d_history = 0

# ------------------------------------------------
# 3. Walk through historical market data in order
# ------------------------------------------------

for file in sorted(DAILY_DIR.glob("daily_*.json")):

    date = file.stem.replace("daily_", "")

    with open(file, "r") as f:
        data = json.load(f)

    rows = data.get("results", [])

    # Skip holidays / closures
    if not rows:
        continue

    # ------------------------------------------------
    # If today is the monthly snapshot date,
    # inspect liquidity BEFORE adding today's data.
    # ------------------------------------------------

    if date in snapshot_dates:

        year_month = snapshot_dates[date]
        universe = monthly_universes[year_month]

        for row in rows:

            ticker = row.get("T")
            close = row.get("c")

            if ticker not in universe:
                continue

            if close is None or close <= 0:
                continue

            snapshot_observations += 1

            # Require 20 PRIOR trading observations
            if len(history[ticker]) < 20:
                continue

            adv20 = sum(history[ticker]) / 20

            snapshot_with_20d_history += 1
            adv_values_all.append(adv20)

            if close >= 5:
                adv_values_price5.append(adv20)

    # ------------------------------------------------
    # AFTER inspection, add today's dollar volume
    # to history. Thus the 20D liquidity measure
    # does not include the signal day's own volume.
    # ------------------------------------------------

    for row in rows:

        ticker = row.get("T")
        close = row.get("c")
        volume = row.get("v")

        if (
            ticker is None
            or close is None
            or volume is None
            or close <= 0
            or volume < 0
        ):
            continue

        dollar_volume = close * volume
        history[ticker].append(dollar_volume)


# ------------------------------------------------
# 4. Summarize distribution
# ------------------------------------------------

def summarize(values, label):

    x = np.array(values, dtype=float)

    percentiles = [
        1, 5, 10, 20, 25,
        50, 75, 80, 90, 95, 99
    ]

    lines = []

    lines.append(f"\n{label}")
    lines.append(f"Observations: {len(x)}")

    for p in percentiles:
        value = np.percentile(x, p)

        lines.append(
            f"P{p}: ${value:,.0f}"
        )

    thresholds = [
        100_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
        25_000_000,
        50_000_000
    ]

    lines.append("\nShare above candidate thresholds:")

    for threshold in thresholds:

        pct = (
            (x >= threshold).mean() * 100
        )

        lines.append(
            f">= ${threshold:,.0f}: {pct:.1f}%"
        )

    return lines


output = []

output.append(
    f"Monthly snapshot observations: "
    f"{snapshot_observations}"
)

output.append(
    f"With 20D history: "
    f"{snapshot_with_20d_history}"
)

output += summarize(
    adv_values_all,
    "ALL COMMON STOCKS"
)

output += summarize(
    adv_values_price5,
    "COMMON STOCKS WITH PRICE >= $5"
)

# Print
for line in output:
    print(line)

# Save summary
output_file = RESULTS_DIR / "liquidity_distribution.txt"

with open(output_file, "w") as f:
    f.write("\n".join(output))

print(
    "\nSaved summary to:",
    output_file
)