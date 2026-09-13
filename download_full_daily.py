from getpass import getpass
from pathlib import Path
from datetime import datetime, timedelta
import json
import time
import requests

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

cache_dir = Path("data/raw/daily")
cache_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------
# Safety check: verify Hong Kong
# ---------------------------------

def check_hk_network():
    try:
        r = requests.get(
            "https://ipinfo.io/country",
            timeout=10
        )
        country = r.text.strip()

        if country != "HK":
            raise RuntimeError(
                f"Network country is {country}, not HK. "
                "Stopping before Massive API access."
            )

        print("Network check: HK")

    except Exception as e:
        print("Could not safely verify HK network.")
        raise e


check_hk_network()

# ---------------------------------
# Get API key only at runtime
# ---------------------------------

API_KEY = getpass("Enter Massive API key: ")

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

session = requests.Session()

# ---------------------------------
# Download function with retries
# ---------------------------------

def download_day(date_str):

    url = (
        "https://api.massive.com/v2/aggs/grouped/"
        f"locale/us/market/stocks/{date_str}"
    )

    params = {
        "adjusted": "true",
        "include_otc": "false"
    }

    for attempt in range(1, 6):

        try:
            response = session.get(
                url,
                params=params,
                headers=headers,

                # 10 sec connection timeout
                # 90 sec read timeout
                timeout=(10, 90)
            )

        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError
        ) as e:

            wait = min(2 ** attempt, 30)

            print(
                f"{date_str}: network timeout "
                f"({type(e).__name__}), "
                f"retrying in {wait}s..."
            )

            # Safety: check that we are still on HK network
            check_hk_network()

            time.sleep(wait)
            continue

        # Successful response
        if response.ok:
            return response.json()

        # Temporary server/rate-limit problems
        if response.status_code in {
            429, 500, 502, 503, 504
        }:

            wait = min(2 ** attempt, 30)

            print(
                f"{date_str}: status "
                f"{response.status_code}, "
                f"retrying in {wait}s..."
            )

            time.sleep(wait)
            continue

        # Unexpected permanent failure
        print(
            f"{date_str}: unexpected status "
            f"{response.status_code}"
        )

        raise RuntimeError(
            f"Massive request failed for {date_str}"
        )

    raise RuntimeError(
        f"Failed after 5 attempts: {date_str}"
    )


# ---------------------------------
# Main loop
# ---------------------------------

start = datetime.strptime(
    START_DATE, "%Y-%m-%d"
)

end = datetime.strptime(
    END_DATE, "%Y-%m-%d"
)

current = start

downloaded = 0
cached = 0
empty_days = 0
api_requests = 0

while current <= end:

    # Weekends never need a request
    if current.weekday() >= 5:
        current += timedelta(days=1)
        continue

    date_str = current.strftime("%Y-%m-%d")

    cache_file = (
        cache_dir / f"daily_{date_str}.json"
    )

    # Use existing cache
    if cache_file.exists():
        cached += 1
        current += timedelta(days=1)
        continue

    # Re-check network periodically
    if api_requests > 0 and api_requests % 25 == 0:
        check_hk_network()

    data = download_day(date_str)

    api_requests += 1

    with open(cache_file, "w") as f:
        json.dump(data, f)

    downloaded += 1

    if not data.get("results", []):
        empty_days += 1

    if downloaded % 25 == 0:
        print(
            f"Progress: {date_str} | "
            f"new files: {downloaded}"
        )

    # Avoid unnecessary request bursts
    time.sleep(0.2)

    current += timedelta(days=1)


print("\nDONE")
print("New files downloaded:", downloaded)
print("Already cached:", cached)
print("Empty weekday files:", empty_days)
print("Massive API requests:", api_requests)