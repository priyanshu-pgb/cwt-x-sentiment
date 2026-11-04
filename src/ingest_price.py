"""
Download daily BTC/USD OHLCV bars for the project window.

Primary source: Coinbase Exchange public API (no key needed). It
takes explicit start/end dates, which I need because the tweet
corpus is historical. Kraken's public OHLC endpoint only serves the
most recent ~720 daily candles, so it is kept as a fallback for
recent windows. Coinbase returns at most 300 candles per request,
so the range is fetched in chunks.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, DATASET_END, DATASET_START

OUT_CSV = DATA_DIR / "prices.csv"
HEADERS = {"User-Agent": "cwt-x-sentiment-student-project/1.0"}
CHUNK_DAYS = 250  # safely under Coinbase's 300-candle limit


def fetch_coinbase(start, end):
    rows = []
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=CHUNK_DAYS - 1), end_ts)
        resp = requests.get(
            "https://api.exchange.coinbase.com/products/BTC-USD/candles",
            params={
                "start": cursor.strftime("%Y-%m-%dT00:00:00Z"),
                "end": (chunk_end + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
                "granularity": 86400,
            },
            headers=HEADERS,
            timeout=60,
        )
        resp.raise_for_status()
        # Coinbase candle format: [time, low, high, open, close, volume]
        for t, low, high, open_, close, volume in resp.json():
            rows.append(
                {
                    "date": pd.Timestamp(t, unit="s", tz="UTC").normalize().tz_localize(None),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        cursor = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.4)  # be polite to a free public endpoint
    return pd.DataFrame(rows)


def fetch_kraken():
    """Fallback: only the last ~720 daily bars Kraken still serves."""
    resp = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": "XBTUSD", "interval": 1440},
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()["result"]
    candles = result[next(k for k in result if k != "last")]
    rows = [
        {
            "date": pd.Timestamp(t, unit="s", tz="UTC").normalize().tz_localize(None),
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": float(v),
        }
        for t, o, h, l, c, _vwap, v, _n in candles
    ]
    return pd.DataFrame(rows)


def download_prices(start=DATASET_START, end=DATASET_END):
    try:
        df = fetch_coinbase(start, end)
        source = "Coinbase Exchange public API"
    except Exception as exc:  # fall back to Kraken and clip to window
        print(f"Coinbase failed ({exc}); trying Kraken ...")
        df = fetch_kraken()
        source = "Kraken public API (recent bars only)"
    df = df.drop_duplicates("date").sort_values("date")
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    print(f"Got {len(df)} daily bars from {source}: "
          f"{df['date'].min().date()} .. {df['date'].max().date()}")
    return df.reset_index(drop=True)


def main():
    df = download_prices()
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
