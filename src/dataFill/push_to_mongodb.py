# src/dataFill/push_to_mongodb.py

from pathlib import Path
ROOT_DIR  = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "historical_data_clean.json"

import os
import sys
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from dotenv import load_dotenv
from src.features.feature_engineering import engineer_features, MODEL_FEATURES
from src.dataFill.mongo_store import save_features, create_index

load_dotenv()


def load_clean_data():
    df = pd.read_json(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} rows")
    return df


if __name__ == "__main__":
    # Step 1 — Load
    df_raw = load_clean_data()
    print(type(df_raw))
    print(df_raw.shape)
    print(df_raw["aqi"].dtype)
    print(df_raw["aqi"].isna().sum())
    print(df_raw["aqi"].head(5))
    # Engineer features
    print("\nEngineering features...")
    df_engineered = engineer_features(df_raw)
    print(f"Shape after engineering: {df_engineered.shape}")

    # Keep only needed columns
    KEEP_COLS = (
        ["timestamp"]
        + MODEL_FEATURES + ["aqi"]
        + ["target_24h", "target_48h", "target_72h"]
    )
    df_final = df_engineered[KEEP_COLS].copy()
    
    # Sanity check
    print(f"\nSanity check:")
    print(f"  Rows:    {len(df_final)}")
    print(f"  Columns: {len(df_final.columns)}")
    print(f"  NaNs:    {df_final.isnull().sum().sum()}")
    print(f"  From:    {df_final['timestamp'].min()}")
    print(f"  To:      {df_final['timestamp'].max()}")

    if df_final.isnull().sum().sum() > 0:
        print("\nNaN values found — fix before pushing")
        print(df_final.isnull().sum()[df_final.isnull().sum() > 0])
    else:
        # Step 5 — Create index then push
        print("\nCreating MongoDB index...")
        create_index()

        print(f"\nPushing {len(df_final)} rows to MongoDB...")
        save_features(df_final)
        print("Backfill complete")