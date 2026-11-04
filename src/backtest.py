"""
A simple long/flat backtest on the walk-forward n+1 signal.

Rule: at the close of day t, if the model predicts a next-day return
larger than +0.2%, hold BTC for day t+1; otherwise stay in cash.
No shorting, no leverage, no transaction costs (a simplification I
flag in the report). Metrics required by the brief - Sharpe, Sortino
and max drawdown - are computed on the strategy's daily returns over
the out-of-sample walk-forward period only.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, REPORTS_DIR

METRICS_JSON = REPORTS_DIR / "backtest_metrics.json"
EQUITY_CSV = DATA_DIR / "equity_curve.csv"
THRESHOLD = 0.002  # only trade when |signal| clears +0.2%
TRADING_DAYS = 365  # BTC trades every day


def sharpe(returns):
    sd = returns.std()
    return float(returns.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def sortino(returns):
    downside = returns[returns < 0]
    dd = np.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    return float(returns.mean() / dd * np.sqrt(TRADING_DAYS)) if dd > 0 else 0.0


def max_drawdown(equity):
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def run_backtest():
    preds = pd.read_parquet(DATA_DIR / "predictions.parquet")
    sig = preds[preds["horizon"] == 1].sort_values("date").copy()
    sig["position"] = (sig["y_pred_ret"] > THRESHOLD).astype(int)
    sig["strategy_ret"] = sig["position"] * sig["y_true_ret"]
    sig["equity"] = (1 + sig["strategy_ret"]).cumprod()
    sig["benchmark_equity"] = (1 + sig["y_true_ret"]).cumprod()
    sig.to_csv(EQUITY_CSV, index=False)

    active = sig[sig["position"] == 1]
    metrics = {
        "period_start": str(sig["date"].min().date()),
        "period_end": str(sig["date"].max().date()),
        "n_days": int(len(sig)),
        "days_in_market": int(sig["position"].sum()),
        "pct_time_in_market": float(sig["position"].mean()),
        "total_return": float(sig["equity"].iloc[-1] - 1),
        "benchmark_total_return": float(sig["benchmark_equity"].iloc[-1] - 1),
        "sharpe_ratio": sharpe(sig["strategy_ret"]),
        "sortino_ratio": sortino(sig["strategy_ret"]),
        "max_drawdown": max_drawdown(sig["equity"]),
        "win_rate_active_days": float((active["strategy_ret"] > 0).mean()) if len(active) else 0.0,
        "threshold": THRESHOLD,
        "notes": "Long/flat, no leverage, no transaction costs, out-of-sample walk-forward period only.",
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Backtest {metrics['period_start']} .. {metrics['period_end']}: "
          f"return {metrics['total_return']:.1%} (buy&hold {metrics['benchmark_total_return']:.1%}), "
          f"Sharpe {metrics['sharpe_ratio']:.2f}, Sortino {metrics['sortino_ratio']:.2f}, "
          f"max DD {metrics['max_drawdown']:.1%}")
    return metrics


if __name__ == "__main__":
    run_backtest()
