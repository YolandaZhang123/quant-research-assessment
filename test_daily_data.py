from getpass import getpass
from pathlib import Path
import json
import requests

# ----------------------------
# 1. Choose one historical trading day
# ----------------------------
DATE = "2025-01-02"

# ----------------------------
# 2. Local cache path
# ----------------------------
cache_dir = Path("data/raw")
cache_dir.mkdir(parents=True, exist_ok=True)

cache_file = cache_dir / f"daily_{DATE}.json"

# ----------------------------
# 3. If already downloaded, use local cache
# ----------------------------
if cache_file.exists():
    print("Loading from local cache...")
    
    with open(cache_file, "r") as f:
        data = json.load(f)

# ----------------------------
# 4. Otherwise request from Massive
# ----------------------------
else:
    API_KEY = getpass("Enter Massive API key: ")

    url = (
        f"https://api.massive.com/v2/aggs/grouped/"
        f"locale/us/market/stocks/{DATE}"
    )

    params = {
        "adjusted": "true",
        "include_otc": "false"
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=30
    )

    print("Status:", response.status_code)

    if not response.ok:
        print(response.text[:500])
        raise SystemExit

    data = response.json()

    # Save locally so we do not request it again
    with open(cache_file, "w") as f:
        json.dump(data, f)

    print("Saved to:", cache_file)

# ----------------------------
# 5. Inspect the returned data
# ----------------------------
results = data.get("results", [])

print("Number of stocks returned:", len(results))

print("\nFirst 5 records:")
for row in results[:5]:
    print(row)