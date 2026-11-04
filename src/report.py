"""
Assemble the evaluation report (reports/evaluation_report.md) from
the pipeline outputs, and render a simple PDF copy with reportlab
when it is available.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, HORIZONS, REPORTS_DIR

OUT_MD = REPORTS_DIR / "evaluation_report.md"
OUT_PDF = REPORTS_DIR / "evaluation_report.pdf"


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_markdown():
    model_metrics = _load_json(REPORTS_DIR / "model_metrics.json")
    backtest = _load_json(REPORTS_DIR / "backtest_metrics.json")
    dist = _load_json(REPORTS_DIR / "sentiment_distribution.json")
    posts = pd.read_parquet(DATA_DIR / "posts_scored.parquet")
    prices = pd.read_csv(DATA_DIR / "prices.csv", parse_dates=["date"])
    start = pd.to_datetime(posts["timestamp_utc"]).min()
    end = pd.to_datetime(posts["timestamp_utc"]).max()

    lines = [
        "# Evaluation report - X sentiment price prediction for Bitcoin (BTC/USD)",
        "",
        "CrowdWisdomTrading quantitative data science intern task. "
        "Question: can daily X (Twitter) sentiment about Bitcoin, combined "
        "with price history only, forecast the BTC/USD price 0-5 days ahead?",
        "",
        "## Data",
        "",
        f"- **Posts:** {len(posts):,} Bitcoin tweets, {start.date()} to {end.date()}, "
        "from the public Kaggle 'Bitcoin Tweets' corpus (kaushiksuresh147), "
        "mirrored on Hugging Face as `khaihernlow/bitcoin-tweets`. The corpus "
        "has author follower counts but no per-tweet like/repost counts, so "
        "engagement weighting uses followers (documented in the README). "
        "Stored volume was capped at 2,000 posts/day to keep scoring time sane. "
        "The corpus was collected in bursts, so it covers 36 collection days "
        "inside the window; on other days sentiment is carried forward and "
        "post volume is 0 (see Limitations).",
        f"- **Prices:** {len(prices)} daily BTC/USD OHLCV bars from the Coinbase "
        "Exchange public API for the same window.",
        f"- **Sentiment:** VADER compound per post. Overall mean {dist['mean']:+.3f}, "
        f"{dist['share_positive_gt_0.05']:.0%} of posts positive, "
        f"{dist['share_negative_lt_-0.05']:.0%} negative.",
        "- Posts and prices are loaded into SQLite (`data/cwt.db`, tables "
        "`posts`, `prices`, `daily_features`).",
        "",
        "## Method",
        "",
        "Daily features are built from sentiment + price only: sentiment "
        "level/lags/rolling means/momentum, engagement-weighted sentiment, "
        "post volume and its z-score, positive/negative share, lagged "
        "returns and rolling volatility. Price features are lagged one day "
        "so nothing from the future leaks in. One GradientBoostingRegressor "
        "per horizon predicts the return to n+0 .. n+5.",
        "",
        "Validation is walk-forward: expanding window, first 60% of days "
        "as initial training, then refit every 14 days and score the next "
        "14. The baseline is persistence (predict tomorrow = today).",
        "",
        "## Forecast results (out-of-sample)",
        "",
        "| Horizon | Model RMSE | Baseline RMSE | Model MAE | Baseline MAE | Directional acc. |",
        "|---|---|---|---|---|---|",
    ]
    for h in HORIZONS:
        m = model_metrics[str(h)]
        lines.append(
            f"| n+{h} | {m['rmse']:,.0f} | {m['baseline_rmse']:,.0f} | "
            f"{m['mae']:,.0f} | {m['baseline_mae']:,.0f} | "
            f"{m['directional_accuracy']:.1%} |"
        )
    lines += [
        "",
        "RMSE/MAE are in USD on the predicted price. Directional accuracy "
        "is the share of days where the predicted return had the right sign "
        "(the persistence baseline predicts a flat return, so it has no "
        "direction to score).",
        "",
        "## Backtest (long/flat on the n+1 signal)",
        "",
        f"Out-of-sample period {backtest['period_start']} to {backtest['period_end']} "
        f"({backtest['n_days']} days, in the market {backtest['pct_time_in_market']:.0%} "
        "of the time, only trading when the predicted return clears "
        f"+{backtest['threshold']:.1%}):",
        "",
        "| Metric | Strategy | Buy & hold |",
        "|---|---|---|",
        f"| Total return | {backtest['total_return']:+.1%} | {backtest['benchmark_total_return']:+.1%} |",
        f"| Sharpe ratio | {backtest['sharpe_ratio']:.2f} | - |",
        f"| Sortino ratio | {backtest['sortino_ratio']:.2f} | - |",
        f"| Max drawdown | {backtest['max_drawdown']:.1%} | - |",
        f"| Win rate (active days) | {backtest['win_rate_active_days']:.0%} | - |",
        "",
        "No transaction costs or slippage are modelled, and the strategy "
        "never shorts, so these numbers are optimistic.",
        "",
        "## Charts",
        "",
        "- `charts/price_vs_sentiment.png` - price vs sentiment over time",
        "- `charts/sentiment_distribution.png` - per-post sentiment distribution",
        "- `charts/predicted_vs_actual.png` - walk-forward forecast vs actual (n+1)",
        "- `charts/equity_curve.png` - backtest equity curve and drawdown",
        "",
        "## Limitations",
        "",
        "- **Data period:** public tweet corpora with usable timestamps "
        "mostly end in early 2023 (free X data collection got much harder "
        "after the API changes), so the offline run uses Aug 2022 - Jan 2023. "
        "The live Apify mode in `src/ingest_apify.py` covers the recent "
        "180 days when an API token is available.",
        "- **No per-tweet engagement** in the offline corpus (no likes or "
        "reposts), so engagement features lean on author followers.",
        "- **VADER is a general lexicon**, not finance-specific; it scores "
        "sarcasm and crypto slang imperfectly.",
        "- **One asset, one window.** Results may not hold in other market "
        "regimes; BTC is close to a random walk at daily frequency, which "
        "is why the naive baseline is hard to beat on RMSE.",
        "- The 2,000 posts/day cap means volume features describe the "
        "sampled stream, not the full firehose.",
        "- **Burst collection:** the public corpus was gathered a few "
        "weeks at a time, so only 36 of the 133 window days have posts. "
        "Sentiment is forward-filled across the gaps (volume = 0 there), "
        "which weakens the daily signal - another reason the live Apify "
        "mode is preferable when a token is available.",
        "",
    ]
    text = "\n".join(lines)
    OUT_MD.write_text(text, encoding="utf-8")
    print(f"Saved {OUT_MD}")
    return text


def render_pdf(markdown_text):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        print("reportlab not available - Markdown report only.")
        return False
    styles = getSampleStyleSheet()
    story = []
    for line in markdown_text.splitlines():
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
        style = styles["BodyText"]
        if line.startswith("## "):
            line, style = line[3:], styles["Heading2"]
        elif line.startswith("# "):
            line, style = line[2:], styles["Heading1"]
        # Tables are kept as plain text lines - simple but readable.
        line = line.replace("**", "").replace("`", "")
        story.append(Paragraph(line, style))
    SimpleDocTemplate(str(OUT_PDF), pagesize=A4).build(story)
    print(f"Saved {OUT_PDF}")
    return True


def main():
    text = build_markdown()
    render_pdf(text)


if __name__ == "__main__":
    main()
