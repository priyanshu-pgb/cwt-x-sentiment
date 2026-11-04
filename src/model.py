"""
Walk-forward validation of the price forecasts.

Setup: expanding window. The first 60% of days are the initial
training set; after that the model is refit every 14 days and scores
the next 14 days, so every prediction is made by a model that only
saw the past. One GradientBoostingRegressor is trained per horizon
(n+0 .. n+5) on the return target; predictions are converted back to
prices. The baseline is persistence - "tomorrow's price = today's
close" (predicted return 0) - which is hard to beat on RMSE for a
near-random-walk asset like BTC, and the report says so honestly.

Outputs:
- reports/model_metrics.json   per-horizon RMSE/MAE/direction vs baseline
- data/predictions.parquet     every out-of-sample prediction
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, HORIZONS, REPORTS_DIR
from src.features import FEATURE_COLS

PREDICTIONS_PARQUET = DATA_DIR / "predictions.parquet"
METRICS_JSON = REPORTS_DIR / "model_metrics.json"

INITIAL_TRAIN_FRAC = 0.60
FOLD_DAYS = 14


def _make_model():
    # Deliberately small/regularised: the training set is only ~100
    # daily rows, so a big ensemble just memorises noise and produces
    # wild price swings out of sample.
    return GradientBoostingRegressor(
        n_estimators=120,
        max_depth=2,
        learning_rate=0.08,
        subsample=0.9,
        min_samples_leaf=5,
        random_state=42,
    )


def _base_price(row, horizon):
    """Reference price the predicted return is applied to."""
    return row["open"] if horizon == 0 else row["close"]


def walk_forward(features):
    all_preds = []
    metrics = {}
    for h in HORIZONS:
        data = features.dropna(subset=[f"target_ret_{h}"]).reset_index(drop=True)
        n = len(data)
        start = int(n * INITIAL_TRAIN_FRAC)
        rows = []
        fold = 0
        while start < n:
            stop = min(start + FOLD_DAYS, n)
            train = data.iloc[:start]
            test = data.iloc[start:stop]
            model = _make_model()
            model.fit(train[FEATURE_COLS], train[f"target_ret_{h}"])
            pred_ret = model.predict(test[FEATURE_COLS])
            for i, (_, row) in enumerate(test.iterrows()):
                base = _base_price(row, h)
                rows.append(
                    {
                        "date": row["date"],
                        "horizon": h,
                        "fold": fold,
                        "y_true_price": row[f"target_price_{h}"],
                        "y_pred_price": base * (1 + pred_ret[i]),
                        "y_pred_ret": pred_ret[i],
                        "y_true_ret": row[f"target_ret_{h}"],
                        "base_price": base,
                    }
                )
            start = stop
            fold += 1
        preds = pd.DataFrame(rows)
        all_preds.append(preds)
        metrics[h] = _horizon_metrics(preds)
        m = metrics[h]
        print(f"h={h}: RMSE {m['rmse']:.0f} vs baseline {m['baseline_rmse']:.0f} | "
              f"MAE {m['mae']:.0f} vs {m['baseline_mae']:.0f} | "
              f"dir. acc {m['directional_accuracy']:.1%} (baseline predicts flat)")
    return pd.concat(all_preds, ignore_index=True), metrics


def _horizon_metrics(preds):
    err = preds["y_pred_price"] - preds["y_true_price"]
    base_err = preds["base_price"] - preds["y_true_price"]
    # Direction: did the predicted return have the same sign as the
    # realised one? Baseline predicts 0 return, so it has no direction.
    dir_acc = (np.sign(preds["y_pred_ret"]) == np.sign(preds["y_true_ret"])).mean()
    return {
        "n_predictions": int(len(preds)),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(err.abs().mean()),
        "baseline_rmse": float(np.sqrt((base_err ** 2).mean())),
        "baseline_mae": float(base_err.abs().mean()),
        "directional_accuracy": float(dir_acc),
    }


def train_final_models(features):
    """Refit on all data - used once for the outlook's live forecast."""
    models = {}
    for h in HORIZONS:
        data = features.dropna(subset=[f"target_ret_{h}"])
        model = _make_model()
        model.fit(data[FEATURE_COLS], data[f"target_ret_{h}"])
        models[h] = model
    return models


def main():
    features = pd.read_csv(DATA_DIR / "daily_features.csv", parse_dates=["date"])
    preds, metrics = walk_forward(features)
    preds.to_parquet(PREDICTIONS_PARQUET, index=False)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {PREDICTIONS_PARQUET} ({len(preds):,} predictions) and {METRICS_JSON}")
    return preds, metrics


if __name__ == "__main__":
    main()
