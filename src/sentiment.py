"""
Score every post with VADER and aggregate to daily sentiment.

VADER is a rule-based sentiment scorer made for short social-media
text, so it is a sensible free choice here - no API key, no training.
For each post I keep the compound score (-1 .. +1). Daily aggregates:

- mean / median compound
- engagement-weighted compound. In live (Apify) mode the weight is
  1 + likes + reposts; the offline corpus has no per-post engagement,
  so there the weight is 1 + author followers (an attention proxy).
- post volume, positive/negative share, total followers reached
- distribution stats (percentiles + histogram) for the report
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, REPORTS_DIR

SCORED_PARQUET = DATA_DIR / "posts_scored.parquet"
DAILY_CSV = DATA_DIR / "daily_sentiment.csv"
DIST_JSON = REPORTS_DIR / "sentiment_distribution.json"


def load_posts():
    """Prefer the live Apify file when it exists, else the dataset file."""
    for name in ("posts_apify.parquet", "posts_dataset.parquet"):
        path = DATA_DIR / name
        if path.exists():
            print(f"Scoring posts from {path.name}")
            return pd.read_parquet(path)
    raise FileNotFoundError("No posts file found - run an ingest step first.")


def score_posts(df):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(text)["compound"] for text in df["text"].fillna("")]
    df = df.copy()
    df["compound"] = scores
    return df


def daily_aggregates(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_localize(None).dt.normalize()
    has_engagement = df["likes"].notna().any()
    if has_engagement:
        weight = 1 + df["likes"].fillna(0) + df["reposts"].fillna(0)
    else:
        weight = 1 + df["followers"].fillna(0)
    df["weight"] = weight
    df["is_pos"] = (df["compound"] > 0.05).astype(int)
    df["is_neg"] = (df["compound"] < -0.05).astype(int)

    def agg(group):
        return pd.Series(
            {
                "n_posts": len(group),
                "mean_compound": group["compound"].mean(),
                "median_compound": group["compound"].median(),
                "engagement_weighted_compound": np.average(
                    group["compound"], weights=group["weight"]
                ),
                "pos_share": group["is_pos"].mean(),
                "neg_share": group["is_neg"].mean(),
                "total_followers": group["followers"].fillna(0).sum(),
            }
        )

    daily = df.groupby("date").apply(agg).reset_index()
    return daily, df


def distribution_stats(df):
    c = df["compound"]
    hist_counts, hist_edges = np.histogram(c, bins=20, range=(-1, 1))
    return {
        "n_posts_scored": int(len(df)),
        "mean": float(c.mean()),
        "std": float(c.std()),
        "percentiles": {
            str(p): float(c.quantile(p / 100))
            for p in (1, 5, 25, 50, 75, 95, 99)
        },
        "share_positive_gt_0.05": float((c > 0.05).mean()),
        "share_negative_lt_-0.05": float((c < -0.05).mean()),
        "histogram_bins": [float(x) for x in hist_edges],
        "histogram_counts": [int(x) for x in hist_counts],
    }


def main():
    posts = load_posts()
    scored = score_posts(posts)
    scored.to_parquet(SCORED_PARQUET, index=False)
    daily, _ = daily_aggregates(scored)
    daily.to_csv(DAILY_CSV, index=False)
    stats = distribution_stats(scored)
    DIST_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Scored {len(scored):,} posts over {len(daily)} days.")
    print(f"Mean compound {stats['mean']:.3f}; "
          f"positive share {stats['share_positive_gt_0.05']:.1%}, "
          f"negative share {stats['share_negative_lt_-0.05']:.1%}")
    return daily


if __name__ == "__main__":
    main()
