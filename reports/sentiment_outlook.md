# Bitcoin (BTC) sentiment outlook

_Data window ends 2023-01-09. Based on 70,585 X posts scored with VADER and daily BTC/USD prices._

## Where sentiment stands

- Last 7 days mean sentiment: **+0.152** vs a full-window average of +0.139. That puts the current week at about the 58% percentile of all 7-day periods in the window - a fairly typical patch of sentiment.
- Post volume, last 7 days: 12,585 posts (1,798/day vs 1,961/day overall).
- Positive share (last 7d): 40%; negative share: 15%.

## Themes in the last two weeks of posts

- Hashtags: #bitcoin (7,175), #btc (5,752), #crypto (3,699), #eth (2,974), #ethereum (2,021), #cryptocurrency (1,937), #altcoin (1,355), #blockchain (720)
- Frequent words: https (6,518), price (3,991), ethereum (2,086), cryptocurrency (2,079), altcoin (1,362), follow (1,307), last (1,130), update (982), updates (942), recent (908)

Biggest accounts posting recently (by followers):

- **Binance** (10,062,342 followers): 1.5m views â You guys really do love #Bitcoin. How about 2.5m by EOY? https://t.co/shano4momb
- **CoinDesk** (3,035,978 followers): #Bitcoin's computing power has recovered after dropping about 40% during a winter storm that pummeled through North America over the holiday
- **Bitcoin Magazine** (2,803,231 followers): #Bitcoin â¿ukele â¿ucks are HERE ð Limited number remaining - get yours while supplies last! ð¸ð»ð https://t.co/5Qr3oW9pDj https

## What the model forecasts

Refit on all data, from the last close of $17,178 (2023-01-09):

| Horizon | Predicted return | Predicted price |
|---|---|---|
| n+0 | -0.19% | $17,093 |
| n+1 | -0.01% | $17,176 |
| n+2 | -0.34% | $17,120 |
| n+3 | +0.21% | $17,215 |
| n+4 | +1.41% | $17,421 |
| n+5 | +0.39% | $17,245 |

## Honest caveat

This is a student model trained on one 6-month window of public tweets. Its walk-forward accuracy is close to a naive baseline (see the evaluation report), so the numbers above are a sentiment summary, not a trading signal. Nothing here is financial advice.
