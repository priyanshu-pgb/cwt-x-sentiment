"""
Fetch recent Bitcoin news headlines with Tavily (if a key is set).

The news is context for the sentiment outlook - it is NOT used as a
model feature, because the brief restricts features to sentiment +
price only. Without a TAVILY_API_KEY the step is skipped and a short
note is written instead, so the rest of the pipeline still runs.
"""

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ASSET_NAME, DATA_DIR, TAVILY_API_KEY

OUT_JSON = DATA_DIR / "news.json"
OUT_NOTE = DATA_DIR / "news_note.txt"


def fetch_news():
    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": f"{ASSET_NAME} BTC price market news",
            "topic": "news",
            "days": 7,
            "max_results": 10,
            "search_depth": "basic",
        },
        timeout=60,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "published_date": r.get("published_date"),
            "score": r.get("score"),
            "content": (r.get("content") or "")[:500],
        }
        for r in results
    ]


def main():
    if not TAVILY_API_KEY:
        msg = (
            "News ingestion skipped: TAVILY_API_KEY is not set in .env.\n"
            "The model does not use news (features are sentiment + price\n"
            "only); news headlines are only extra context for the outlook."
        )
        OUT_NOTE.write_text(msg + "\n", encoding="utf-8")
        print(msg)
        return
    news = fetch_news()
    OUT_JSON.write_text(json.dumps(news, indent=2), encoding="utf-8")
    print(f"Saved {len(news)} news items to {OUT_JSON}")


if __name__ == "__main__":
    main()
