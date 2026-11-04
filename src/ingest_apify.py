"""
Live mode: scrape recent X posts about the asset with Apify.

Runs the configured Apify actor (default `apidojo/tweet-scraper`)
through the Apify REST API, waits for the run to finish, downloads
the dataset items and normalises them to the same columns the
offline dataset mode produces:

    timestamp_utc, text, author, followers, likes, reposts, views, url

Usage: put APIFY_API_TOKEN in .env, then run this file. Without a
token it prints what is missing and exits without doing anything -
use the dataset mode (src/ingest_dataset.py) for offline runs.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import APIFY_ACTOR_ID, APIFY_API_TOKEN, CASHTAG, DATA_DIR, KEYWORDS

API_BASE = "https://api.apify.com/v2"
OUT_CSV = DATA_DIR / "posts_apify.csv"
OUT_PARQUET = DATA_DIR / "posts_apify.parquet"

# How far back to search, and a safety cap on volume per run.
DAYS_BACK = 180
MAX_POSTS = 50_000


def _first(item, *paths):
    """Tolerant getter: actor outputs vary a bit between versions."""
    for path in paths:
        cur = item
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return None


def normalise(items):
    rows = []
    for it in items:
        author = _first(it, "author", "user") or {}
        if not isinstance(author, dict):
            author = {}
        rows.append(
            {
                "timestamp_utc": _first(it, "createdAt", "created_at", "date"),
                "text": _first(it, "fullText", "text", "full_text") or "",
                "author": _first(author, "userName", "name", "screen_name") or "",
                "followers": _first(author, "followers", "followersCount"),
                "likes": _first(it, "likeCount", "favoriteCount", "likes"),
                "reposts": _first(it, "retweetCount", "reposts"),
                "views": _first(it, "viewCount", "views"),
                "url": _first(it, "url", "tweetUrl") or "",
            }
        )
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    for col in ("followers", "likes", "reposts", "views"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    return df.reset_index(drop=True)


def run_actor():
    actor_path = APIFY_ACTOR_ID.replace("/", "~")
    # Search terms: cashtag plus the main keywords, last DAYS_BACK days.
    payload = {
        "searchTerms": [CASHTAG] + KEYWORDS,
        "maxTweets": MAX_POSTS,
        "daysBack": DAYS_BACK,
    }
    print(f"Starting Apify actor {APIFY_ACTOR_ID} ...")
    resp = requests.post(
        f"{API_BASE}/acts/{actor_path}/runs",
        params={"token": APIFY_API_TOKEN},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    run = resp.json()["data"]
    run_id, dataset_id = run["id"], run["defaultDatasetId"]

    while True:
        time.sleep(15)
        status = requests.get(
            f"{API_BASE}/actor-runs/{run_id}",
            params={"token": APIFY_API_TOKEN},
            timeout=60,
        ).json()["data"]["status"]
        print(f"  run status: {status}")
        if status == "SUCCEEDED":
            break
        if status in ("FAILED", "ABORTED", "TIMED-OUT"):
            raise RuntimeError(f"Apify run ended with status {status}")

    items = requests.get(
        f"{API_BASE}/datasets/{dataset_id}/items",
        params={"token": APIFY_API_TOKEN, "format": "json"},
        timeout=300,
    ).json()
    print(f"Downloaded {len(items):,} raw items from Apify.")
    return items


def main():
    if not APIFY_API_TOKEN:
        print(
            "APIFY_API_TOKEN is not set, so live scraping is skipped.\n"
            "Add it to a .env file (see .env.example) and run again,\n"
            "or use the offline dataset mode: python src/ingest_dataset.py"
        )
        return
    df = normalise(run_actor())
    df.to_csv(OUT_CSV, index=False)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved {len(df):,} posts to {OUT_CSV} and {OUT_PARQUET}")


if __name__ == "__main__":
    main()
