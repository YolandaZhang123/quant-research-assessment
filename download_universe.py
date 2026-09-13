from getpass import getpass
from pathlib import Path
import json
import requests

DATE = "2025-01-02"

cache_dir = Path("data/raw")
cache_dir.mkdir(parents=True, exist_ok=True)

cache_file = cache_dir / f"universe_{DATE}.json"

# If already downloaded, use local cache
if cache_file.exists():

    print("Loading universe from local cache...")

    with open(cache_file, "r") as f:
        results = json.load(f)

else:

    API_KEY = getpass("Enter Massive API key: ")

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    url = "https://api.massive.com/v3/reference/tickers"

    params = {
        "market": "stocks",
        "type": "CS",
        "date": DATE,
        "active": "true",
        "limit": 1000,
        "sort": "ticker",
        "order": "asc"
    }

    results = []
    page = 1

    while url:

        print(f"Downloading page {page}...")

        if page == 1:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=30
            )
        else:
            response = requests.get(
                url,
                headers=headers,
                timeout=30
            )

        if not response.ok:
            print("Status:", response.status_code)
            print(response.text[:500])
            raise SystemExit

        data = response.json()

        results.extend(data.get("results", []))

        url = data.get("next_url")

        page += 1

    with open(cache_file, "w") as f:
        json.dump(results, f)

    print("Saved locally.")

print("Total historical common stocks:", len(results))