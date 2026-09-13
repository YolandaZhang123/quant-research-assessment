from getpass import getpass
import requests

API_KEY = getpass("Enter Massive API key: ")

url = "https://api.massive.com/v3/reference/tickers"

params = {
    "market": "stocks",
    "active": "true",
    "limit": 1
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

if response.ok:
    print(response.json().get("results", [])[:1])
else:
    print(response.text[:500])