"""
Build the daily feature frame from sentiment + price ONLY.

One row per day t. To keep the setup honest (no look-ahead):
- sentiment features use posts from day t and earlier;
- price features (past returns, volatility) are lagged by one day,
  so they only use closes up to day t-1;
- targets are future moves measured from close_t:
    horizon 0 : close_t / open_t - 1   (today's intraday move)
    horizon h : close_{t+h} / close_t - 1   for h = 1..5
  Predicted returns are converted back to prices by the model code
  (price = close_t * (1 + ret), or open_t * (1 + ret) for h = 0).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, HORIZONS

OUT_CSV = DATA_DIR / "daily_features.csv"

FEATURE_COLS = [
    "sent", "sent_ew", "sent_lag1", "sent_lag3", "sent_lag7",
    "sent_ma3", "sent_ma7", "sent_ma14", "sent_mom",
    "pos_share", "neg_share", "n_posts", "volume_z", "followers_log",
    "ret_lag1", "ret_lag3", "ret_lag7", "vol7", "dow",
]


def build_features():
    prices = pd.read_csv(DATA_DIR / "prices.csv", parse_dates=["date"])
    sent = pd.read_csv(DATA_DIR / "daily_sentiment.csv", parse_dates=["date"])

    df = prices.set_index("date").sort_index()
    s = sent.set_index("date").sort_index()

    # Align sentiment onto the price calendar. The public corpus was
    # collected in bursts (a few weeks on, a few off), so on days with
    # no collected posts the last known sentiment is carried forward
    # and n_posts stays 0 - the volume features keep the gaps visible.
    for col in ("mean_compound", "median_compound", "engagement_weighted_compound",
                "pos_share", "neg_share", "total_followers"):
        df[col] = s[col].reindex(df.index).ffill()
    df["n_posts"] = s["n_posts"].reindex(df.index).fillna(0)

    # --- sentiment features (day t and earlier) ---
    df["sent"] = df["mean_compound"]
    df["sent_ew"] = df["engagement_weighted_compound"]
    df["sent_lag1"] = df["sent"].shift(1)
    df["sent_lag3"] = df["sent"].shift(3)
    df["sent_lag7"] = df["sent"].shift(7)
    df["sent_ma3"] = df["sent"].rolling(3).mean()
    df["sent_ma7"] = df["sent"].rolling(7).mean()
    df["sent_ma14"] = df["sent"].rolling(14).mean()
    df["sent_mom"] = df["sent_ma3"] - df["sent_ma14"]
    vol_mean = df["n_posts"].rolling(30).mean()
    vol_std = df["n_posts"].rolling(30).std().replace(0, np.nan)
    df["volume_z"] = ((df["n_posts"] - vol_mean) / vol_std).fillna(0)
    df["followers_log"] = np.log1p(df["total_followers"])

    # --- price features (lagged one day: closes up to t-1 only) ---
    logret = np.log(df["close"] / df["close"].shift(1))
    df["ret_lag1"] = logret.shift(1)
    df["ret_lag3"] = logret.rolling(3).sum().shift(1)
    df["ret_lag7"] = logret.rolling(7).sum().shift(1)
    df["vol7"] = logret.rolling(7).std().shift(1)

    df["dow"] = df.index.dayofweek

    # --- targets ---
    df["target_ret_0"] = df["close"] / df["open"] - 1
    df["target_price_0"] = df["close"]
    for h in HORIZONS[1:]:
        df[f"target_ret_{h}"] = df["close"].shift(-h) / df["close"] - 1
        df[f"target_price_{h}"] = df["close"].shift(-h)

    keep = ["open", "high", "low", "close", "volume"] + FEATURE_COLS + [
        c for c in df.columns if c.startswith("target_")
    ]
    out = df[keep].dropna(subset=FEATURE_COLS).reset_index()
    return out


def main():
    features = build_features()
    features.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}: {len(features)} daily rows, "
          f"{features['date'].min().date()} .. {features['date'].max().date()}")
    return features


if __name__ == "__main__":
    main()
