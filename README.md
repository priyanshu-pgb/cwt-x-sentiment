# X Sentiment Price Prediction - Bitcoin (BTC/USD)

Built for the CrowdWisdomTrading quantitative data scientist intern
task (TZURONI). The idea: collect what people post on X about Bitcoin,
turn it into daily sentiment features, and test honestly - with
walk-forward validation - whether those features help forecast the
BTC/USD price 0 to 5 days ahead.

## What it does

1. **Ingest** - X posts about Bitcoin plus daily BTC/USD prices for
   the same window, loaded into a small SQLite database.
2. **Sentiment** - every post is scored with VADER; scores are
   aggregated per day (mean/median, engagement-weighted, post volume,
   positive/negative share).
3. **Features** - daily features from sentiment + price *only*:
   sentiment lags and rolling means, sentiment momentum, post-volume
   z-score, lagged returns, rolling volatility, day of week.
4. **Model** - one GradientBoostingRegressor per horizon (n+0 .. n+5),
   validated walk-forward (expanding window, refit every 14 days,
   no shuffling, no look-ahead), compared against a naive persistence
   baseline.
5. **Backtest** - a simple long/flat rule on the n+1 signal, reported
   with Sharpe, Sortino and max drawdown.
6. **Reports** - charts, a one-page sentiment outlook, and a full
   evaluation report.

## Project structure

```
config.py               asset, keys, dataset window - all settings
run_pipeline.py         one command that runs everything (dataset mode)
src/
  ingest_apify.py       live mode: scrape X with Apify (needs a token)
  ingest_dataset.py     offline mode: download the public tweet corpus
  ingest_price.py       daily BTC/USD OHLCV (Coinbase public API, Kraken fallback)
  ingest_news.py        recent headlines via Tavily (optional, context only)
  sentiment.py          VADER scoring + daily aggregates
  store.py              SQLite database (posts, prices, daily_features)
  features.py           the daily feature frame (sentiment + price only)
  model.py              walk-forward validation, per-horizon metrics
  backtest.py           long/flat backtest: Sharpe / Sortino / max drawdown
  charts.py             the four report charts
  outlook.py            writes reports/sentiment_outlook.md
  report.py             writes reports/evaluation_report.md (+ PDF)
data/                   posts, prices, cwt.db, model outputs
reports/                evaluation report, outlook, metrics JSON, charts/
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py            # add --force to re-download data
```

### Mode 1 - offline dataset run (no keys needed)

`src/ingest_dataset.py` streams the public Bitcoin tweets corpus and
keeps the window **2022-08-30 to 2023-01-09** (133 days - the corpus
ends in early January 2023), capped at 2,000 posts per day so scoring
finishes in reasonable time. One quirk of that corpus: it was
collected in bursts, so only 36 of the 133 days actually have posts
(70,585 of them). On the other days the pipeline carries the last
known sentiment forward and records zero post volume, so the gaps
stay visible in the features instead of being hidden.

### Mode 2 - live Apify run

Copy `.env.example` to `.env` and set `APIFY_API_TOKEN` (and
optionally `TAVILY_API_KEY` for news context), then:

```bash
python src/ingest_apify.py        # scrapes the last 180 days from X
python run_pipeline.py            # picks up the Apify posts automatically
```

The actor is configurable (`APIFY_ACTOR_ID`, default
`apidojo/tweet-scraper`).

## Data sources

| Data | Source | Notes |
|---|---|---|
| Tweets (offline) | Kaggle "Bitcoin Tweets" by kaushiksuresh147, mirrored on Hugging Face as [`khaihernlow/bitcoin-tweets`](https://huggingface.co/datasets/khaihernlow/bitcoin-tweets) | Same corpus that feeds `StephanAkkerman/crypto-stock-tweets` (CC-BY-4.0), but that bundle strips timestamps, so the mirror with original fields is used |
| Prices | Coinbase Exchange public API (Kraken as fallback) | Daily BTC/USD OHLCV, no key needed |
| News (optional) | Tavily | Headlines for outlook context only - never a model feature |

## Results

From the offline run (70,585 posts scored; walk-forward test period
22 Nov 2022 - 8 Jan 2023):

- **Forecasting:** the model does *not* beat the naive persistence
  baseline on RMSE at any horizon (e.g. n+1: RMSE $438 vs $240 for
  the baseline) - daily BTC is close to a random walk and ~100
  training days is little data. Directional accuracy is 46-55%
  depending on horizon, i.e. around a coin flip with a slight edge
  at n+3/n+4.
- **Backtest (long/flat on the n+1 signal):** total return **+5.7%**
  vs +6.0% for buy & hold over the same out-of-sample weeks, with
  **Sharpe 1.76**, **Sortino 1.69** and **max drawdown -6.6%** -
  roughly market-like returns but with a smoother ride, helped by
  being out of the market ~40% of days. No transaction costs are
  modelled, so treat this as optimistic.

See `reports/evaluation_report.md` for the full numbers,
`reports/sentiment_outlook.md` for the one-page outlook, and
`reports/charts/` for the charts.

## Limitations (the honest part)

- Public tweet corpora with timestamps mostly stop in early 2023 -
  scraping X got much harder after the API changes - so the offline
  window is Aug 2022 to Jan 2023. The Apify mode exists for fresh data.
- The offline corpus has **no per-tweet likes/reposts**, only author
  follower counts, so engagement weighting uses followers as an
  attention proxy. Live Apify data has real engagement counts.
- VADER is a general-purpose lexicon, not finance-specific; it
  mishandles some crypto slang and sarcasm.
- BTC at daily frequency is close to a random walk, so the naive
  baseline is genuinely hard to beat on RMSE - the evaluation report
  shows that rather than hiding it.
- The backtest ignores transaction costs and never shorts.
