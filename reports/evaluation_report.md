# Evaluation report - X sentiment price prediction for Bitcoin (BTC/USD)

CrowdWisdomTrading quantitative data science intern task. Question: can daily X (Twitter) sentiment about Bitcoin, combined with price history only, forecast the BTC/USD price 0-5 days ahead?

## Data

- **Posts:** 70,585 Bitcoin tweets, 2022-08-30 to 2023-01-09, from the public Kaggle 'Bitcoin Tweets' corpus (kaushiksuresh147), mirrored on Hugging Face as `khaihernlow/bitcoin-tweets`. The corpus has author follower counts but no per-tweet like/repost counts, so engagement weighting uses followers (documented in the README). Stored volume was capped at 2,000 posts/day to keep scoring time sane. The corpus was collected in bursts, so it covers 36 collection days inside the window; on other days sentiment is carried forward and post volume is 0 (see Limitations).
- **Prices:** 133 daily BTC/USD OHLCV bars from the Coinbase Exchange public API for the same window.
- **Sentiment:** VADER compound per post. Overall mean +0.139, 42% of posts positive, 16% negative.
- Posts and prices are loaded into SQLite (`data/cwt.db`, tables `posts`, `prices`, `daily_features`).

## Method

Daily features are built from sentiment + price only: sentiment level/lags/rolling means/momentum, engagement-weighted sentiment, post volume and its z-score, positive/negative share, lagged returns and rolling volatility. Price features are lagged one day so nothing from the future leaks in. One GradientBoostingRegressor per horizon predicts the return to n+0 .. n+5.

Validation is walk-forward: expanding window, first 60% of days as initial training, then refit every 14 days and score the next 14. The baseline is persistence (predict tomorrow = today).

## Forecast results (out-of-sample)

| Horizon | Model RMSE | Baseline RMSE | Model MAE | Baseline MAE | Directional acc. |
|---|---|---|---|---|---|
| n+0 | 555 | 240 | 421 | 164 | 47.9% |
| n+1 | 438 | 240 | 352 | 164 | 45.8% |
| n+2 | 608 | 344 | 485 | 229 | 47.9% |
| n+3 | 605 | 396 | 495 | 286 | 55.3% |
| n+4 | 796 | 414 | 647 | 304 | 55.3% |
| n+5 | 1,054 | 437 | 822 | 343 | 34.8% |

RMSE/MAE are in USD on the predicted price. Directional accuracy is the share of days where the predicted return had the right sign (the persistence baseline predicts a flat return, so it has no direction to score).

## Backtest (long/flat on the n+1 signal)

Out-of-sample period 2022-11-22 to 2023-01-08 (48 days, in the market 60% of the time, only trading when the predicted return clears +0.2%):

| Metric | Strategy | Buy & hold |
|---|---|---|
| Total return | +5.7% | +6.0% |
| Sharpe ratio | 1.76 | - |
| Sortino ratio | 1.69 | - |
| Max drawdown | -6.6% | - |
| Win rate (active days) | 45% | - |

No transaction costs or slippage are modelled, and the strategy never shorts, so these numbers are optimistic.

## Charts

- `charts/price_vs_sentiment.png` - price vs sentiment over time
- `charts/sentiment_distribution.png` - per-post sentiment distribution
- `charts/predicted_vs_actual.png` - walk-forward forecast vs actual (n+1)
- `charts/equity_curve.png` - backtest equity curve and drawdown

## Limitations

- **Data period:** public tweet corpora with usable timestamps mostly end in early 2023 (free X data collection got much harder after the API changes), so the offline run uses Aug 2022 - Jan 2023. The live Apify mode in `src/ingest_apify.py` covers the recent 180 days when an API token is available.
- **No per-tweet engagement** in the offline corpus (no likes or reposts), so engagement features lean on author followers.
- **VADER is a general lexicon**, not finance-specific; it scores sarcasm and crypto slang imperfectly.
- **One asset, one window.** Results may not hold in other market regimes; BTC is close to a random walk at daily frequency, which is why the naive baseline is hard to beat on RMSE.
- The 2,000 posts/day cap means volume features describe the sampled stream, not the full firehose.
- **Burst collection:** the public corpus was gathered a few weeks at a time, so only 36 of the 133 window days have posts. Sentiment is forward-filled across the gaps (volume = 0 there), which weakens the daily signal - another reason the live Apify mode is preferable when a token is available.
