"""
Charts for the report - the "visual matrix" of sentiment vs price.

(a) price and daily sentiment over time (dual axis)
(b) distribution of per-post sentiment scores
(c) walk-forward predicted vs actual price (n+1)
(d) backtest equity curve with drawdown shading
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHARTS_DIR, DATA_DIR


def chart_price_sentiment():
    prices = pd.read_csv(DATA_DIR / "prices.csv", parse_dates=["date"])
    sent = pd.read_csv(DATA_DIR / "daily_sentiment.csv", parse_dates=["date"])
    df = prices.merge(sent, on="date", how="inner")
    fig, ax1 = plt.subplots(figsize=(11, 4.2))
    ax1.plot(df["date"], df["close"], color="#1f77b4", lw=1.4, label="BTC close (USD)")
    ax1.set_ylabel("BTC/USD close")
    ax1.set_xlabel("Date")
    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["mean_compound"], color="#d62728", lw=0.7,
             alpha=0.35, label="Daily mean sentiment")
    ax2.plot(df["date"], df["mean_compound"].rolling(7).mean(), color="#d62728",
             lw=1.6, label="Sentiment 7-day mean")
    ax2.axhline(0, color="grey", lw=0.8, ls="--")
    ax2.set_ylabel("VADER compound (daily mean)")
    ax1.set_title("BTC price vs X sentiment over time")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "price_vs_sentiment.png", dpi=130)
    plt.close(fig)


def chart_sentiment_distribution():
    scored = pd.read_parquet(DATA_DIR / "posts_scored.parquet")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scored["compound"], bins=40, range=(-1, 1), color="#2ca02c", alpha=0.8)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel("VADER compound score per post")
    ax.set_ylabel("Number of posts")
    ax.set_title(f"Sentiment distribution across {len(scored):,} posts")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "sentiment_distribution.png", dpi=130)
    plt.close(fig)


def chart_predicted_vs_actual(horizon=1):
    preds = pd.read_parquet(DATA_DIR / "predictions.parquet")
    p = preds[preds["horizon"] == horizon].sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(p["date"], p["y_true_price"], color="#1f77b4", lw=1.4, label="Actual price")
    ax.plot(p["date"], p["y_pred_price"], color="#ff7f0e", lw=1.1,
            alpha=0.85, label=f"Predicted (n+{horizon}, walk-forward)")
    ax.set_ylabel("BTC/USD")
    ax.set_xlabel("Date")
    ax.set_title(f"Walk-forward forecast vs actual, horizon n+{horizon}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "predicted_vs_actual.png", dpi=130)
    plt.close(fig)


def chart_equity_curve():
    eq = pd.read_csv(DATA_DIR / "equity_curve.csv", parse_dates=["date"])
    eq["drawdown"] = eq["equity"] / eq["equity"].cummax() - 1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(eq["date"], eq["equity"], color="#1f77b4", lw=1.5, label="Strategy (long/flat)")
    ax1.plot(eq["date"], eq["benchmark_equity"], color="grey", lw=1.1,
             ls="--", label="Buy & hold")
    ax1.set_ylabel("Equity (start = 1)")
    ax1.set_title("Backtest equity curve and drawdown (out-of-sample period)")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.fill_between(eq["date"], eq["drawdown"], 0, color="#d62728", alpha=0.45)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "equity_curve.png", dpi=130)
    plt.close(fig)


def main():
    chart_price_sentiment()
    chart_sentiment_distribution()
    chart_predicted_vs_actual()
    chart_equity_curve()
    print(f"Charts saved to {CHARTS_DIR}")


if __name__ == "__main__":
    main()
