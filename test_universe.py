from getpass import getpass
import requests

DATE = "2025-01-02"

API_KEY = getpass("Enter Massive API key: ")

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
results = data.get("results", [])

print("Common stocks in first page:", len(results))
print("Has another page:", "next_url" in data)

print("\nFirst 5 ticker symbols:")
for row in results[:5]:
    print(
        row.get("ticker"),
        "| type:", row.get("type"),
        "| active:", row.get("active")
    )