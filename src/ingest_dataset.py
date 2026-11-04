"""
Download a real public Bitcoin tweets dataset for the offline run.

Source used: the Hugging Face dataset `khaihernlow/bitcoin-tweets`
(data/Bitcoin_tweets.jsonl), which mirrors the Kaggle dataset
"Bitcoin Tweets" by kaushiksuresh147. That same corpus is also one of
the sources bundled into `StephanAkkerman/crypto-stock-tweets`
(CC-BY-4.0), but the bundled CSV there only keeps text + source link -
the timestamps were stripped, so it cannot be aligned with prices.
The JSONL mirror keeps the original fields, so I use that one.

Notes on the source data:
- It has NO per-tweet like/repost counts (the original Kaggle file
  never had them). `likes`, `reposts` and `views` are therefore left
  empty for this mode; `followers` (author follower count) is present
  and is used as the attention weight instead. This is documented in
  the README and the evaluation report.
- The file is ~3.5 GB, so it is streamed line by line and only rows
  inside the configured window are kept. To keep scoring time sane I
  also cap the kept posts per day (config.POSTS_PER_DAY_CAP).
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, DATASET_END, DATASET_START, POSTS_PER_DAY_CAP

SOURCE_URL = (
    "https://huggingface.co/datasets/khaihernlow/bitcoin-tweets"
    "/resolve/main/data/Bitcoin_tweets.jsonl"
)
SOURCE_NAME = "khaihernlow/bitcoin-tweets (mirror of Kaggle kaushiksuresh147/bitcoin-tweets)"

OUT_CSV = DATA_DIR / "posts_dataset.csv"
OUT_PARQUET = DATA_DIR / "posts_dataset.parquet"


def _to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _iter_lines():
    """Yield lines from a local copy when CWT_LOCAL_JSONL is set (handy
    for re-running without another 3.5 GB download), else stream the
    file from Hugging Face."""
    local = os.environ.get("CWT_LOCAL_JSONL")
    if local:
        print(f"Reading local copy: {local}")
        with open(local, encoding="utf-8", errors="replace") as fh:
            yield from fh
        return
    print(f"Streaming {SOURCE_URL}")
    with requests.get(SOURCE_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        yield from resp.iter_lines(decode_unicode=True)


def download_posts(start=DATASET_START, end=DATASET_END, cap_per_day=POSTS_PER_DAY_CAP):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)  # inclusive of last day
    kept_per_day = Counter()
    rows = []
    scanned = 0

    print(f"Keeping posts from {start} to {end}, max {cap_per_day} per day ...")
    if True:
        for line in _iter_lines():
            if not line:
                continue
            scanned += 1
            if scanned % 500_000 == 0:
                print(f"  scanned {scanned:,} lines, kept {len(rows):,}")
            try:
                item = json.loads(line)
                ts = pd.Timestamp(item["date"])
            except Exception:
                continue
            if not (start_ts <= ts < end_ts):
                continue
            day = ts.strftime("%Y-%m-%d")
            if kept_per_day[day] >= cap_per_day:
                continue
            kept_per_day[day] += 1
            rows.append(
                {
                    "timestamp_utc": ts,
                    "text": item.get("text", ""),
                    "author": item.get("user_name", ""),
                    "followers": _to_int(item.get("user_followers")),
                    # Not present in this corpus - see module docstring.
                    "likes": None,
                    "reposts": None,
                    "views": None,
                    "url": "",
                }
            )

    df = pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)
    print(f"Done. Scanned {scanned:,} lines, kept {len(df):,} posts.")
    return df


def main():
    df = download_posts()
    df.to_csv(OUT_CSV, index=False)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved {OUT_CSV} and {OUT_PARQUET}")
    print(f"Date range: {df['timestamp_utc'].min()} .. {df['timestamp_utc'].max()}")


if __name__ == "__main__":
    main()
