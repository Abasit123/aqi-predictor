# src/pipelines/training_pipeline.py

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config.env_setup import setup
setup()

import joblib
import numpy as np
import pandas as pd
import hopsworks

from sklearn.linear_model import LinearRegression
from sklearn.ensemble     import RandomForestRegressor
from sklearn.metrics      import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost  import XGBRegressor
from pymongo  import MongoClient
from dotenv   import load_dotenv
from datetime import datetime

from src.features.feature_engineering import get_features, TARGET

load_dotenv()

os.makedirs("models", exist_ok=True)

TARGETS = {
    "24h": "target_24h",
    "48h": "target_48h",
    "72h": "target_72h",
}


def get_models(horizon: str) -> dict:
    if horizon == "24h":
        return {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(
                n_estimators=100, max_depth=6,
                min_samples_leaf=20, min_samples_split=40,
                max_features=0.5, random_state=42, n_jobs=-1
            ),
            "xgboost": XGBRegressor(
                n_estimators=500, learning_rate=0.05,
                max_depth=5, subsample=0.8,
                colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.5, reg_lambda=2.0,
                random_state=42, verbosity=0
            ),
        }
    elif horizon == "48h":
        return {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(
                n_estimators=100, max_depth=4,
                min_samples_leaf=30, min_samples_split=60,
                max_features=0.5, random_state=42, n_jobs=-1
            ),
            "xgboost": XGBRegressor(
                n_estimators=300, learning_rate=0.03,
                max_depth=3, subsample=0.7,
                colsample_bytree=0.6, min_child_weight=10,
                reg_alpha=1.0, reg_lambda=5.0,
                random_state=42, verbosity=0
            ),
        }
    else:  # 72h
        return {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(
                n_estimators=100, max_depth=3,
                min_samples_leaf=40, min_samples_split=80,
                max_features=0.4, random_state=42, n_jobs=-1
            ),
            "xgboost": XGBRegressor(
                n_estimators=200, learning_rate=0.03,
                max_depth=3, subsample=0.6,
                colsample_bytree=0.6, min_child_weight=15,
                reg_alpha=2.0, reg_lambda=10.0,
                random_state=42, verbosity=0
            ),
        }


# ── Step 1 — Load data ───────────────────────────────────────

def load_data() -> pd.DataFrame:
    print("Loading data from MongoDB...")
    client     = MongoClient(
        os.getenv("MONGO_URI"),
        serverSelectionTimeoutMS=30000
    )
    collection = client[os.getenv("MONGO_DB")]["aqi_features"]
    records    = list(collection.find({}, {"_id": 0}))

    if not records:
        raise ValueError("No data in MongoDB")

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    before = len(df)
    df     = df.dropna(subset=["target_24h", "target_48h", "target_72h"])
    print(f"✓ Loaded {len(df)} rows ({before - len(df)} dropped)")
    print(f"  Date range: {df['timestamp'].min().date()} → "
          f"{df['timestamp'].max().date()}")
    return df


# ── Step 2 — Sanity check ────────────────────────────────────

def sanity_check(df: pd.DataFrame) -> tuple:
    print("\nSanity check:")
    print(f"  Rows:            {len(df)}")
    #sprint(f"  AQI mean:        {df['aqi'].mean():.1f}")
    print(f"  target_24h mean: {df['target_24h'].mean():.1f}")
    print(f"  target_48h mean: {df['target_48h'].mean():.1f}")
    print(f"  target_72h mean: {df['target_72h'].mean():.1f}")

    from src.features.feature_engineering import MODEL_FEATURES
    nan_cols = [
        f for f in MODEL_FEATURES
        if f in df.columns and df[f].isna().any()
    ]

    if nan_cols:
        print(f"\n  - {len(nan_cols)} features with NaNs (excluded):")
        for col in nan_cols:
            print(f"    - {col}: {df[col].isna().sum()} NaN rows")
    else:
        print(" -- No NaN values")

    return df, nan_cols


# ── Step 3 — Split ───────────────────────────────────────────

def split(df, target_col, features):
    X   = df[features]
    y   = df[target_col]
    cut = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    print(f"  Train: {len(X_train)} rows  "
          f"({df['timestamp'].iloc[0].date()} → "
          f"{df['timestamp'].iloc[cut].date()})")
    print(f"  Test:  {len(X_test)} rows   "
          f"({df['timestamp'].iloc[cut].date()} → "
          f"{df['timestamp'].iloc[-1].date()})")
    return X_train, X_test, y_train, y_test


# ── Step 4 — Train and evaluate ──────────────────────────────

def train_and_evaluate(model, model_name, X_train, X_test,
                       y_train, y_test) -> dict:
    model.fit(X_train, y_train)

    preds       = model.predict(X_test)
    train_preds = model.predict(X_train)

    mae      = mean_absolute_error(y_test, preds)
    rmse     = np.sqrt(mean_squared_error(y_test, preds))
    r2       = r2_score(y_test, preds)
    train_r2 = r2_score(y_train, train_preds)
    train_mae= mean_absolute_error(y_train, train_preds)

    print(f"    {model_name:<22} "
          f"MAE={mae:.2f}  RMSE={rmse:.2f}  "
          f"R²={r2:.3f}  train_R²={train_r2:.3f}")

    return {
        "model_name": model_name,
        "mae":        round(mae,       2),
        "rmse":       round(rmse,      2),
        "r2":         round(r2,        3),
        "train_r2":   round(train_r2,  3),
        "train_mae":  round(train_mae, 2),
    }


# ── Step 5 — Save locally ────────────────────────────────────

def save_locally(model, model_name, horizon) -> str:
    path = f"models/{model_name}_{horizon}.pkl"
    joblib.dump(model, path)
    return path


# ── Step 6 — Save to Hopsworks ───────────────────────────────

def save_to_registry(project, model_path, model_name,
                     horizon, metrics, is_best) -> tuple:
    mr            = project.get_model_registry()
    registry_name = f"aqi_{model_name}_{horizon}"

    description = f"AQI {horizon} forecast — {model_name}"
    if is_best:
        description += " [BEST]"

    model_obj = mr.python.create_model(
        name=registry_name,
        metrics={
            "mae":      metrics["mae"],
            "rmse":     metrics["rmse"],
            "r2":       metrics["r2"],
            "train_r2": metrics["train_r2"],
            "is_best":  1.0 if is_best else 0.0,
        },
        description=description
    )
    model_obj.save(model_path)

    label = " - Saved + marked best" if is_best else " - Saved"
    print(f"    {label}: {registry_name} v{model_obj.version}")
    return registry_name, model_obj.version


# ── Step 7 — Save metrics to MongoDB ─────────────────────────

def save_metrics_to_mongo(all_metrics: list):
    client     = MongoClient(os.getenv("MONGO_URI"))
    collection = client[os.getenv("MONGO_DB")]["model_metrics"]

    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    for m in all_metrics:
        m["trained_at"] = trained_at

    collection.insert_many(all_metrics)
    print(f"\n - Appended {len(all_metrics)} metric records to MongoDB")


# ── Step 8 — Connect Hopsworks ───────────────────────────────

def connect_hopsworks():
    print("\nConnecting to Hopsworks...")
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    print(f"Connected to {project.name}")
    return project


# ── Step 9 — Summary ─────────────────────────────────────────

def print_summary(all_metrics, best_per_horizon):
    print(f"\n{'=' * 65}")
    print("FINAL SUMMARY")
    print(f"{'=' * 65}")
    summary = pd.DataFrame(all_metrics)[
        ["horizon", "model_name", "mae", "rmse", "r2", "train_r2"]
    ].sort_values(["horizon", "mae"])
    print(summary.to_string(index=False))

    print(f"\n{'=' * 65}")
    print("BEST MODEL PER HORIZON")
    print(f"{'=' * 65}")
    for horizon, info in best_per_horizon.items():
        print(f"  {horizon}: {info['model_name']} "
              f"(MAE={info['mae']}  R²={info['r2']})")


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print(f"Training Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    df               = load_data()
    df, nan_cols     = sanity_check(df)
    project          = connect_hopsworks()

    all_metrics      = []
    best_per_horizon = {}

    from src.features.feature_engineering import MODEL_FEATURES

    for horizon, target_col in TARGETS.items():
        print(f"\n{'─' * 65}")
        print(f"  Horizon: {horizon}  ({target_col})")
        print(f"{'─' * 65}")

        horizon_features = [
            f for f in get_features(horizon)
            if f not in nan_cols and f in df.columns
        ]
        models = get_models(horizon)

        print(f"  Features: {len(horizon_features)}")

        X_train, X_test, y_train, y_test = split(
            df, target_col, horizon_features
        )

        horizon_results = []
        trained_models  = {}

        for model_name, model in models.items():
            print(f"\n  Training {model_name}...")
            metrics              = train_and_evaluate(
                model, model_name,
                X_train, X_test,
                y_train, y_test
            )
            metrics["horizon"]    = horizon
            metrics["target"]     = target_col
            metrics["n_features"] = len(horizon_features)

            horizon_results.append(metrics)
            trained_models[model_name] = model

        best      = min(horizon_results, key=lambda x: x["mae"])
        best_name = best["model_name"]
        print(f"\n  - Best for {horizon}: {best_name} "
              f"(MAE={best['mae']}  R²={best['r2']})")

        for result in horizon_results:
            model_name = result["model_name"]
            is_best    = model_name == best_name
            model_path = save_locally(
                trained_models[model_name], model_name, horizon
            )
            registry_name, version = save_to_registry(
                project, model_path, model_name,
                horizon, result, is_best
           )
            result["registry_name"] = registry_name
            result["version"]       = version

        best["registry_name"]    = f"aqi_{best_name}_{horizon}"
        best_per_horizon[horizon] = best
        all_metrics.extend(horizon_results)

    save_metrics_to_mongo(all_metrics)
    print_summary(all_metrics, best_per_horizon)
    print(f"\n--- Training complete---")