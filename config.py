"""
Project config for the CWT X-sentiment price prediction project.

Everything that might change between runs (API keys, the asset, the
download window) lives here so the rest of the code stays clean.
Keys are read from a local .env file which is never committed.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"

for _d in (DATA_DIR, REPORTS_DIR, CHARTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _load_env():
    """Tiny .env reader so I don't need python-dotenv for three keys."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()

# --- API keys (from .env, empty string when not set) -------------------------
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN", "")
APIFY_ACTOR_ID = os.environ.get("APIFY_ACTOR_ID", "apidojo/tweet-scraper")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")  # optional, not used by default

# --- Asset -------------------------------------------------------------------
ASSET_NAME = "Bitcoin"
ASSET_TICKER = "BTC"
ASSET_PAIR = "BTC/USD"
CASHTAG = "$BTC"
KEYWORDS = ["Bitcoin", "BTC", "$BTC"]

# --- Offline dataset run ------------------------------------------------------
# Window taken from the public Bitcoin tweets corpus (see
# src/ingest_dataset.py): 2022-08-30 to 2023-01-09, 133 days - the
# most recent long stretch in that corpus with heavy daily volume
# (the corpus itself ends in early January 2023).
DATASET_START = "2022-08-30"
DATASET_END = "2023-01-09"
# The corpus has far more tweets per day than I can score in reasonable
# time, so I cap the stored/scored posts per day. The cap is documented
# in the README; daily volume features use the capped counts.
POSTS_PER_DAY_CAP = 2000

# --- Model --------------------------------------------------------------------
HORIZONS = [0, 1, 2, 3, 4, 5]  # forecast price at n+0 .. n+5
