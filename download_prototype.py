from getpass import getpass
from pathlib import Path
from datetime import datetime, timedelta
import json
import time
import requests

START_DATE = "2024-11-01"
END_DATE = "2025-01-31"

cache_dir = Path("data/raw/daily")
cache_dir.mkdir(parents=True, exist_ok=True)

API_KEY = getpass("Enter Massive API key: ")

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

start = datetime.strptime(START_DATE, "%Y-%m-%d")
end = datetime.strptime(END_DATE, "%Y-%m-%d")

current = start

downloaded = 0
cached = 0
skipped = 0

while current <= end:

    # Weekend -> skip without making API request
    if current.weekday() >= 5:
        current += timedelta(days=1)
        continue

    date = current.strftime("%Y-%m-%d")
    cache_file = cache_dir / f"daily_{date}.json"

    # Already downloaded -> do not request again
    if cache_file.exists():
        cached += 1
        current += timedelta(days=1)
        continue

    url = (
        f"https://api.massive.com/v2/aggs/grouped/"
        f"locale/us/market/stocks/{date}"
    )

    params = {
        "adjusted": "true",
        "include_otc": "false"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=30
    )

    if response.ok:

        with open(cache_file, "w") as f:
            json.dump(response.json(), f)

        downloaded += 1
        print(f"{date}: downloaded")

    else:
        # Market holiday or unavailable date
        skipped += 1
        print(f"{date}: skipped (status {response.status_code})")

    # Be gentle with repeated requests
    time.sleep(0.2)

    current += timedelta(days=1)

print("\nDONE")
print("Downloaded:", downloaded)
print("Already cached:", cached)
print("Skipped/unavailable:", skipped)