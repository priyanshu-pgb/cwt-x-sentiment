"""
Run the whole project end to end (offline dataset mode).

    python run_pipeline.py            # reuse downloads that already exist
    python run_pipeline.py --force    # re-download posts and prices

For the live mode instead: put APIFY_API_TOKEN in .env, run
`python src/ingest_apify.py`, then run this script - it automatically
prefers the Apify posts file when one exists.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR
from src import (
    backtest,
    charts,
    features,
    ingest_dataset,
    ingest_news,
    ingest_price,
    model,
    outlook,
    report,
    sentiment,
    store,
)


def main():
    force = "--force" in sys.argv

    posts_file = DATA_DIR / "posts_dataset.parquet"
    if force or not posts_file.exists():
        if (DATA_DIR / "posts_apify.parquet").exists() and not force:
            print("Apify posts file exists - using live data, skipping dataset download.")
        else:
            ingest_dataset.main()
    if force or not (DATA_DIR / "prices.csv").exists():
        ingest_price.main()
    ingest_news.main()

    sentiment.main()
    store.build_db()
    feats = features.main()
    store.save_features(feats)
    model.main()
    backtest.run_backtest()
    charts.main()
    outlook.main()
    report.main()
    print("\nPipeline finished. Reports are in reports/.")


if __name__ == "__main__":
    main()
