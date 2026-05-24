import os
import requests

FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN")
FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")

BASE_URL = f"https://{FRESHDESK_DOMAIN}.freshdesk.com/api/v2" if FRESHDESK_DOMAIN else None


def search_articles(query: str, per_page: int = 5):
    """Search Freshdesk knowledge base articles."""
    if not BASE_URL or not FRESHDESK_API_KEY:
        return {"error": "Freshdesk credentials not configured"}

    endpoint = f"{BASE_URL}/solutions/articles/search?term={query}&per_page={per_page}"
    try:
        resp = requests.get(endpoint, auth=(FRESHDESK_API_KEY, "X"))
        if resp.status_code == 200:
            return resp.json().get("results", [])
        return {"error": f"API request failed with status {resp.status_code}"}
    except requests.RequestException as e:
        return {"error": str(e)}
