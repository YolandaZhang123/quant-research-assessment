from getpass import getpass
from pathlib import Path
import json
import time
import requests


DAILY_DIR = Path("data/raw/daily")
UNIVERSE_DIR = Path("data/raw/monthly_universe")

UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------
# 1. Check Hong Kong network
# ------------------------------------------------

def check_hk_network():

    r = requests.get(
        "https://ipinfo.io/country",
        timeout=10
    )

    country = r.text.strip()

    if country != "HK":
        raise RuntimeError(
            f"Current network country is {country}, not HK. "
            "Stopping before Massive access."
        )

    print("Network check: HK")


check_hk_network()


# ------------------------------------------------
# 2. Find first REAL trading day of each month
# ------------------------------------------------

first_trading_days = {}

for file in sorted(
    DAILY_DIR.glob("daily_*.json")
):

    date = file.stem.replace(
        "daily_", ""
    )

    year_month = date[:7]

    with open(file, "r") as f:
        data = json.load(f)

    # Ignore holidays / market closures
    if not data.get("results", []):
        continue

    if year_month not in first_trading_days:
        first_trading_days[year_month] = date


print(
    "Months identified:",
    len(first_trading_days)
)


# ------------------------------------------------
# 3. Runtime API key
# ------------------------------------------------

API_KEY = getpass(
    "Enter Massive API key: "
)

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

session = requests.Session()


# ------------------------------------------------
# 4. Safe request helper
# ------------------------------------------------

def safe_get(url, params=None):

    for attempt in range(1, 6):

        try:

            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=(10, 90)
            )

        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError
        ):

            wait = min(
                2 ** attempt,
                30
            )

            print(
                f"Network issue. "
                f"Retrying in {wait}s..."
            )

            check_hk_network()

            time.sleep(wait)
            continue

        if response.ok:
            return response

        if response.status_code in {
            429, 500, 502, 503, 504
        }:

            wait = min(
                2 ** attempt,
                30
            )

            print(
                f"Status {response.status_code}; "
                f"retrying in {wait}s..."
            )

            time.sleep(wait)
            continue

        raise RuntimeError(
            f"Unexpected Massive status: "
            f"{response.status_code}"
        )

    raise RuntimeError(
        "Request failed after 5 attempts."
    )


# ------------------------------------------------
# 5. Download monthly universes
# ------------------------------------------------

downloaded = 0
cached = 0
total_api_requests = 0

for month_number, (
    year_month,
    date
) in enumerate(
    sorted(first_trading_days.items()),
    start=1
):

    cache_file = (
        UNIVERSE_DIR
        / f"universe_{year_month}.json"
    )

    if cache_file.exists():
        cached += 1
        continue

    # Recheck network periodically
    if month_number % 10 == 0:
        check_hk_network()

    url = (
        "https://api.massive.com/"
        "v3/reference/tickers"
    )

    params = {
        "market": "stocks",
        "type": "CS",
        "date": date,
        "active": "true",
        "limit": 1000,
        "sort": "ticker",
        "order": "asc"
    }

    tickers = []

    page = 1

    while url:

        if page == 1:
            response = safe_get(
                url,
                params=params
            )
        else:
            response = safe_get(url)

        total_api_requests += 1

        data = response.json()

        for row in data.get(
            "results", []
        ):
            ticker = row.get("ticker")

            if ticker is not None:
                tickers.append(ticker)

        url = data.get("next_url")

        page += 1

    tickers = sorted(set(tickers))

    # Only cache what the study needs
    output = {
        "reference_date": date,
        "tickers": tickers
    }

    with open(cache_file, "w") as f:
        json.dump(output, f)

    downloaded += 1

    print(
        f"{year_month}: "
        f"{len(tickers)} common stocks"
    )

    time.sleep(0.2)


print("\nDONE")
print(
    "Monthly universes downloaded:",
    downloaded
)
print(
    "Already cached:",
    cached
)
print(
    "API requests:",
    total_api_requests
)