"""
Load posts and prices into a small SQLite database (data/cwt.db).

The brief asks for the datasets to be joined in a database, so the
pipeline keeps three tables there:

- posts          one row per scored post
- prices         daily BTC/USD OHLCV
- daily_features the model-ready daily feature frame (written after
                 src/features.py has built it)

Everything else in the pipeline reads/writes CSV/parquet files, and
the database is the joined, queryable copy of the same data.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

DB_PATH = DATA_DIR / "cwt.db"


def _connect():
    return sqlite3.connect(DB_PATH)


def build_db():
    posts_path = DATA_DIR / "posts_scored.parquet"
    if not posts_path.exists():
        posts_path = DATA_DIR / "posts_dataset.parquet"
    posts = pd.read_parquet(posts_path)
    posts["timestamp_utc"] = pd.to_datetime(posts["timestamp_utc"], utc=True).astype(str)
    prices = pd.read_csv(DATA_DIR / "prices.csv")
    with _connect() as con:
        posts.to_sql("posts", con, if_exists="replace", index=False)
        prices.to_sql("prices", con, if_exists="replace", index=False)
        n_posts = con.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        n_prices = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    print(f"cwt.db: {n_posts:,} posts, {n_prices} price rows")
    return DB_PATH


def save_features(features: pd.DataFrame):
    with _connect() as con:
        features.to_sql("daily_features", con, if_exists="replace", index=False)
        n = con.execute("SELECT COUNT(*) FROM daily_features").fetchone()[0]
    print(f"cwt.db: daily_features now has {n} rows")


def row_counts():
    with _connect() as con:
        tables = {}
        for (name,) in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ):
            tables[name] = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    return tables


if __name__ == "__main__":
    build_db()
