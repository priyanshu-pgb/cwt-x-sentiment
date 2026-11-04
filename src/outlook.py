"""
Write the one-page sentiment outlook (reports/sentiment_outlook.md).

It summarises where sentiment stands at the end of the data window:
the latest aggregates vs their history, the themes in recent posts,
the biggest accounts posting, and what the model (refit on all the
data) currently forecasts for n+0 .. n+5.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ASSET_NAME, DATA_DIR, REPORTS_DIR
from src.features import FEATURE_COLS
from src.model import train_final_models

OUT_MD = REPORTS_DIR / "sentiment_outlook.md"
RECENT_DAYS = 14

STOPWORDS = set(
    "bitcoin btc crypto the and for with this that from have will just "
    "are you your our out all can now get one about into over more when "
    "what would there their been were they them then than its it's is a "
    "to of in on i it be as at by an or if so no do my we he she his her".split()
)


def _themes(texts):
    hashtags = Counter()
    words = Counter()
    for text in texts:
        hashtags.update(h.lower() for h in re.findall(r"#\w+", text))
        for w in re.findall(r"[a-z]{4,}", text.lower()):
            if w not in STOPWORDS:
                words.update([w])
    return hashtags.most_common(8), words.most_common(10)


def _latest_forecast(features):
    models = train_final_models(features)
    last = features.iloc[-1]
    lines = []
    for h, model in sorted(models.items()):
        pred_ret = float(model.predict(last[FEATURE_COLS].to_frame().T)[0])
        base = last["open"] if h == 0 else last["close"]
        lines.append((h, pred_ret, base * (1 + pred_ret)))
    return last, lines


def main():
    scored = pd.read_parquet(DATA_DIR / "posts_scored.parquet")
    scored["date"] = pd.to_datetime(scored["timestamp_utc"], utc=True).dt.tz_localize(None).dt.normalize()
    daily = pd.read_csv(DATA_DIR / "daily_sentiment.csv", parse_dates=["date"])
    features = pd.read_csv(DATA_DIR / "daily_features.csv", parse_dates=["date"])

    last_date = daily["date"].max()
    recent = scored[scored["date"] > last_date - pd.Timedelta(days=RECENT_DAYS)]
    latest7 = daily.tail(7)
    hist_mean = daily["mean_compound"].mean()
    latest_sent = latest7["mean_compound"].mean()
    pct = (daily["mean_compound"].rolling(7).mean() < latest_sent).mean()

    hashtags, words = _themes(recent["text"].astype(str))
    notable = recent.sort_values("followers", ascending=False).head(3)

    last_row, forecast = _latest_forecast(features)

    news_lines = ""
    news_path = DATA_DIR / "news.json"
    if news_path.exists():
        news = json.loads(news_path.read_text(encoding="utf-8"))[:3]
        if news:
            news_lines = "\nRecent headlines (Tavily):\n" + "\n".join(
                f"- {n['title']}" for n in news
            ) + "\n"

    lines = [
        f"# {ASSET_NAME} (BTC) sentiment outlook",
        "",
        f"_Data window ends {last_date.date()}. Based on "
        f"{len(scored):,} X posts scored with VADER and daily BTC/USD prices._",
        "",
        "## Where sentiment stands",
        "",
        f"- Last 7 days mean sentiment: **{latest_sent:+.3f}** vs a full-window "
        f"average of {hist_mean:+.3f}. That puts the current week at about the "
        f"{pct:.0%} percentile of all 7-day periods in the window - "
        + ("a relatively warm patch of sentiment."
           if pct > 0.6 else
           "a relatively cold patch of sentiment." if pct < 0.4
           else "a fairly typical patch of sentiment."),
        f"- Post volume, last 7 days: {int(latest7['n_posts'].sum()):,} posts "
        f"({latest7['n_posts'].mean():,.0f}/day vs {daily['n_posts'].mean():,.0f}/day overall).",
        f"- Positive share (last 7d): {latest7['pos_share'].mean():.0%}; "
        f"negative share: {latest7['neg_share'].mean():.0%}.",
        "",
        "## Themes in the last two weeks of posts",
        "",
        "- Hashtags: " + ", ".join(f"{tag} ({n:,})" for tag, n in hashtags),
        "- Frequent words: " + ", ".join(f"{w} ({n:,})" for w, n in words),
        "",
        "Biggest accounts posting recently (by followers):",
        "",
    ]
    for _, post in notable.iterrows():
        text = " ".join(str(post["text"]).split())[:140]
        lines.append(f"- **{post['author']}** ({int(post['followers'] or 0):,} followers): {text}")
    lines += [
        news_lines,
        "## What the model forecasts",
        "",
        f"Refit on all data, from the last close of ${last_row['close']:,.0f} "
        f"({pd.Timestamp(last_row['date']).date()}):",
        "",
        "| Horizon | Predicted return | Predicted price |",
        "|---|---|---|",
    ]
    for h, ret, price in forecast:
        lines.append(f"| n+{h} | {ret:+.2%} | ${price:,.0f} |")
    lines += [
        "",
        "## Honest caveat",
        "",
        "This is a student model trained on one 6-month window of public "
        "tweets. Its walk-forward accuracy is close to a naive baseline "
        "(see the evaluation report), so the numbers above are a sentiment "
        "summary, not a trading signal. Nothing here is financial advice.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()
