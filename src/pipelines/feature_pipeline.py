# src/pipelines/feature_pipeline.py

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import requests
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from dotenv import load_dotenv

from src.features.feature_engineering import engineer_features, MODEL_FEATURES
from src.dataFill.mongo_store import save_features, load_recent, update_targets

load_dotenv()

LAT      = float(os.getenv("LAT",   25.3960))
LON      = float(os.getenv("LON",   68.3578))
TIMEZONE = os.getenv("TIMEZONE",    "Asia/Karachi")


# 1. Fetch current hour from Open-Meteo

def fetch_current_hour() -> pd.DataFrame:
    print("\nFetching current hour from Open-Meteo...")
    tz = pytz.timezone(TIMEZONE)
    now = datetime.now(tz).replace(minute=0, second=0, microsecond=0)
    current_time_str = now.strftime("%Y-%m-%dT%H:%M")

    r_weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":        LAT,
            "longitude":       LON,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
                "cloud_cover",
            ],
            "wind_speed_unit": "kmh",
            "forecast_days":   1,
            "timezone":        TIMEZONE
        }
    ).json()

    r_air = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude":      LAT,
            "longitude":     LON,
            "hourly": [
                "pm2_5", "pm10",
                "nitrogen_dioxide",
                "ozone",
                "european_aqi"
            ],
            "forecast_days": 1,
            "timezone":      TIMEZONE
        }
    ).json()

    hw = r_weather["hourly"]
    ha = r_air["hourly"]
    times = hw["time"]

    try:
        idx = times.index(current_time_str)
    except ValueError:
        raise ValueError(f"Current hour {current_time_str} not found in forecast data")

    df = pd.DataFrame([{
        "timestamp":       times[idx],
        "temperature_c":   hw["temperature_2m"][idx],
        "humidity_pct":    hw["relative_humidity_2m"][idx],
        "pressure_hpa":    hw["surface_pressure"][idx],
        "wind_speed_kmh":  hw["wind_speed_10m"][idx],
        "wind_dir":        hw["wind_direction_10m"][idx],
        "cloud_cover_pct": hw["cloud_cover"][idx],
        "pm25":            ha["pm2_5"][idx],
        "pm10":            ha["pm10"][idx],
        "no2":             ha["nitrogen_dioxide"][idx],
        "o3":              float(ha["ozone"][idx]),
        "aqi":             ha["european_aqi"][idx],
    }])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f" Current hour: {df['timestamp'].values[0]}")
    print(f" AQI:          {df['aqi'].values[0]}")
    return df


# 2. Fetch recent history from MongoDB for lag context

def fetch_recent_from_mongo() -> pd.DataFrame:
    print("\nFetching last 24 hours from MongoDB...")

    df = load_recent(n=24)

    # Drop rows with missing aqi — unusable for lag context
    df = df.dropna(subset=["aqi"])

    if df.empty:
        print(" No history found in MongoDB — lag features will be NaN")
        return df

    print(f" Got {len(df)} clean rows from MongoDB")
    print(f"  From: {df['timestamp'].iloc[0]}")
    print(f"  To:   {df['timestamp'].iloc[-1]}")
    return df


# 3. Engineer features for current row

def build_current_row(df_history: pd.DataFrame, df_current: pd.DataFrame) -> pd.DataFrame:
    print("\nEngineering features...")

    df_combined = pd.concat(
        [df_history, df_current],
        ignore_index=True
    )
    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)

    # Inference mode — skips targets and dropna
    df_engineered = engineer_features(df_combined, mode="inference")

    current_row = df_engineered.tail(1).copy()

    # Targets unknown — will be filled later
    current_row["target_day1"] = np.nan
    current_row["target_day2"] = np.nan
    current_row["target_day3"] = np.nan

    print(f"Built row for {current_row['timestamp'].values[0]}")
    print(f"Targets: NaN (will be filled in 24-72 hours)")
    return current_row


# 4. Fill targets for past rows that now have known futures

def fill_past_targets():
    print("\nFilling past targets...")

    df = load_recent(n=96)

    if df.empty:
        print("  Not enough data")
        return

    today = pd.Timestamp.now().normalize()  # midnight today

    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()  # floor to midnight

    # 1. Ensure target columns exist in the dataframe to prevent KeyErrors
    for col in ["target_day1", "target_day2", "target_day3"]:
        if col not in df.columns:
            df[col] = np.nan

    # 2. BACKUP ORIGINAL TARGETS (to see what is already filled in DB)
    df["orig_target_day1"] = df["target_day1"]
    df["orig_target_day2"] = df["target_day2"]
    df["orig_target_day3"] = df["target_day3"]

    # Count how many rows exist per calendar day
    rows_per_day = df.groupby("date")["aqi"].count().reset_index()
    rows_per_day.columns = ["date", "row_count"]

    # Compute daily average only for days that are COMPLETE
    daily_avg = df.groupby("date")["aqi"].mean().reset_index()
    daily_avg.columns = ["date", "daily_avg_aqi"]
    daily_avg = daily_avg.merge(rows_per_day, on="date")

    # Only consider days that have fully passed (not today or future)
    # and have at least 12 rows (half a day) — guards against corrupt days
    complete_days = daily_avg[
        (daily_avg["date"] < today) &
        (daily_avg["row_count"] >= 12)
    ].copy().reset_index(drop=True)

    if len(complete_days) < 2:
        print("  Not enough complete days to fill any targets yet")
        return

    print(f"  Complete days found: {complete_days['date'].dt.date.tolist()}")

    # For each complete day, assign targets from subsequent complete days
    complete_days = complete_days.sort_values("date").reset_index(drop=True)

    complete_days["target_day1"] = complete_days["daily_avg_aqi"].shift(-1)
    complete_days["target_day2"] = complete_days["daily_avg_aqi"].shift(-2)
    complete_days["target_day3"] = complete_days["daily_avg_aqi"].shift(-3)

    # Drop old target columns so merging doesn't append suffixes like _x and _y
    df = df.drop(columns=["target_day1", "target_day2", "target_day3"])

    # Merge newly computed targets back onto hourly rows
    df = df.merge(
        complete_days[["date", "target_day1", "target_day2", "target_day3"]],
        on="date",
        how="left"
    )

    
    # 3. EFFICIENCY FILTER: Only track columns transitioning from NaN -> Value
  
    t1_newly_filled = df["orig_target_day1"].isna() & df["target_day1"].notna()
    t2_newly_filled = df["orig_target_day2"].isna() & df["target_day2"].notna()
    t3_newly_filled = df["orig_target_day3"].isna() & df["target_day3"].notna()
    
    has_new_target_data = t1_newly_filled | t2_newly_filled | t3_newly_filled

    # Only select rows belonging to past days that actually have new target data
    df_to_update = df[
        (df["date"] < today) & 
        has_new_target_data
    ].drop(columns=["date", "orig_target_day1", "orig_target_day2", "orig_target_day3"])

    # 4. Push updates to MongoDB only if there are rows to update
    if len(df_to_update) > 0:
        update_targets(df_to_update)
        print(f"  Filled targets for {len(df_to_update)} rows across "
              f"{df_to_update['timestamp'].dt.normalize().nunique()} days")
    else:
        print("  No targets to fill")


# Main

if __name__ == "__main__":
    print("=" * 45)
    print(f"Feature Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 45)

    # Fetch current hour from API
    df_current = fetch_current_hour()

    # Fetch recent history from MongoDB for lag context
    df_history = fetch_recent_from_mongo()

    # Engineer features
    df_row = build_current_row(df_history, df_current)

    # Sanity check
    nan_count = df_row[MODEL_FEATURES].isnull().sum().sum()
    if nan_count > 0:
        print(f"\n{nan_count} NaN values in features:")
        print(df_row[MODEL_FEATURES].isnull().sum()[
            df_row[MODEL_FEATURES].isnull().sum() > 0
        ])
    else:
        # Keep only needed columns before pushing
        KEEP_COLS = (
            ["timestamp"]
            + MODEL_FEATURES
            + ["target_day1", "target_day2", "target_day3"]
        )
        df_final = df_row[KEEP_COLS].copy()

        print("\nPushing to MongoDB...")
        save_features(df_final)
        print(" Success: Pushed current row")

        # Fill targets for rows that now have known futures
        fill_past_targets()

        print("\nPipeline completed")