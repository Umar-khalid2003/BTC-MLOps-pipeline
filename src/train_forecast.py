"""Feature engineering, training and evaluation of a 7-day BTC close forecaster.

Model: XGBoost direct multi-horizon (one regressor per horizon day, 1..7),
trained on lagged returns / moving averages / volatility features.
Artifacts (model.joblib, metrics.json) are written to MODEL_DIR.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
HORIZON = 7
FEATURE_COLS = [
    "close",
    "ret_1",
    "ret_7",
    "ma_7",
    "ma_30",
    "ma_ratio",
    "vol_7",
    "vol_30",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
]


def build_features(df):
    """Clean data and compute technical indicators."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["close"] = df["close"].astype(float)
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_7"] = df["close"].pct_change(7)
    df["ma_7"] = df["close"].rolling(7).mean()
    df["ma_30"] = df["close"].rolling(30).mean()
    df["ma_ratio"] = df["ma_7"] / df["ma_30"]
    df["vol_7"] = df["ret_1"].rolling(7).std()
    df["vol_30"] = df["ret_1"].rolling(30).std()
    for lag in (1, 2, 3, 7):
        df[f"lag_{lag}"] = df["close"].shift(lag)
    return df


def add_targets(df):
    """Target h = close price h days ahead, expressed as ratio to today's close."""
    df = df.copy()
    for h in range(1, HORIZON + 1):
        df[f"target_{h}"] = df["close"].shift(-h) / df["close"]
    return df


def make_model():
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=2,
        random_state=42,
    )


def fit_models(train_df):
    models = {}
    for h in range(1, HORIZON + 1):
        m = make_model()
        m.fit(train_df[FEATURE_COLS], train_df[f"target_{h}"])
        models[h] = m
    return models


def predict_from_features(models, feature_row, last_close):
    """Return list of HORIZON predicted close prices from one feature row."""
    return [
        float(last_close * models[h].predict(feature_row[FEATURE_COLS])[0])
        for h in range(1, HORIZON + 1)
    ]


def evaluate(models, test_df):
    """MAE/RMSE/MAPE over all horizons on the held-out window."""
    actual, pred = [], []
    for h in range(1, HORIZON + 1):
        p = models[h].predict(test_df[FEATURE_COLS]) * test_df["close"].values
        a = test_df[f"target_{h}"].values * test_df["close"].values
        actual.append(a)
        pred.append(p)
    actual, pred = np.concatenate(actual), np.concatenate(pred)
    err = pred - actual
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err / actual)) * 100),
    }


def train(data_dir=DATA_DIR, model_dir=MODEL_DIR, test_days=60, mape_threshold=15.0):
    raw = pd.read_csv(os.path.join(data_dir, "btc_raw.csv"))
    feats = build_features(raw)
    # Rows still lacking future targets are kept aside for inference.
    latest_row = feats.dropna(subset=FEATURE_COLS).tail(1)
    data = add_targets(feats).dropna(
        subset=FEATURE_COLS + [f"target_{h}" for h in range(1, HORIZON + 1)]
    )

    if len(data) < test_days + 200:
        raise ValueError(f"Not enough data to train ({len(data)} rows)")

    train_df, test_df = data.iloc[:-test_days], data.iloc[-test_days:]
    models = fit_models(train_df)
    metrics = evaluate(models, test_df)

    passed = metrics["mape"] <= mape_threshold
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    metrics.update(
        {
            "model_version": version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "passed_threshold": bool(passed),
            "mape_threshold": mape_threshold,
        }
    )

    if not passed:
        print(f"Model REJECTED: MAPE {metrics['mape']:.2f}% > {mape_threshold}%")
        return metrics, False

    # Refit on all available data before saving the production model.
    final_models = fit_models(data)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(
        {
            "models": final_models,
            "latest_row": latest_row,
            "last_date": latest_row["date"].iloc[0],
            "version": version,
        },
        os.path.join(model_dir, "model.joblib"),
    )
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Model {version} registered. Metrics: {metrics}")
    return metrics, True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--mape-threshold", type=float, default=15.0)
    args = parser.parse_args(argv)
    _, ok = train(args.data_dir, args.model_dir, mape_threshold=args.mape_threshold)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
