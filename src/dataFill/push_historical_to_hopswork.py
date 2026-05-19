# push_historical_to_hopswork.py

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "historical_data_clean.json"

import os
from src.config import env_setup

import pandas as pd
import hopsworks
from dotenv import load_dotenv
from src.features.feature_engineering import engineer_features, MODEL_FEATURES

load_dotenv()


def load_clean_data():
    df = pd.read_json(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} rows")
    return df


def push_to_hopsworks(df: pd.DataFrame):

    for col in df.columns:
        if col == "timestamp":
            df[col] = pd.to_datetime(df[col])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print("Data types:")
    print(df.dtypes)
    print(f"\nShape: {df.shape}")
    print(f"NaNs:  {df.isnull().sum().sum()}")

    # Connecting
    print("\nConnecting to Hopsworks...")
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    fs = project.get_feature_store()
    print(f"Connected to {project.name}")

    # Create or get feature group
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="Hourly AQI features with engineering — Hyderabad Sindh",
        event_time="timestamp"
    )

    # Pushng
    print(f"\nPushing {len(df)} rows to Hopsworks...")
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"{len(df)} rows pushed to feature group 'aqi_features'")


if __name__ == "__main__":
    # Load
    df_raw = load_clean_data()

    # Engineer features
    print("\nEngineering features...")
    df_engineered = engineer_features(df_raw)
    print(f"Shape after engineering: {df_engineered.shape}")

    # Keeping only what model needs + targets + timestamp
    KEEP_COLS = (
        ["timestamp"]
        + MODEL_FEATURES
        + ["target_day1", "target_day2", "target_day3"]
    )
    df_final = df_engineered[KEEP_COLS].copy()
    print(f"Final columns: {df_final.columns.tolist()}")

    # Push to hopswork

    push_to_hopsworks(df_final)

