# feature_pipeline.py
import os

from src.config.env_setup import setup

import requests
import pandas as pd
import hopsworks
import pytz
from datetime import datetime, timezone
from dotenv import load_dotenv
from src.features.feature_engineering import engineer_features, MODEL_FEATURES

load_dotenv()
setup()

LAT      = float(os.getenv("LAT",   25.3960))
LON      = float(os.getenv("LON",   68.3578))
TIMEZONE = os.getenv("TIMEZONE",    "Asia/Karachi")


# 1. Connect to Hopsworks function

def connect_hopsworks():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="Hourly AQI features — Hyderabad Sindh",
        event_time="timestamp"
    )
    print(f" Connected to {project.name}")
    return fg


# 2. Fetch current hour from Open-Meteo function

def fetch_current_hour() -> pd.DataFrame:
    print("\nFetching current hour from Open-Meteo...")
    tz = pytz.timezone(TIMEZONE)
    now = datetime.now(tz).replace(minute=0, second=0, microsecond=0)
    current_time_str = now.strftime("%Y-%m-%dT%H:%M")  # e.g. "2026-05-19T20:00"

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


    # idx = current hour only
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
        "o3":              ha["ozone"][idx],
        "aqi":             ha["european_aqi"][idx],
    }])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f" Current hour: {df['timestamp'].values[0]}")
    print(f" AQI:   {df['aqi'].values[0]}")
    return df


# 3. Fetch recent data from Hopsworks for engineered features

# Data Read from Hopswork Function
def read_fg(fg) -> pd.DataFrame:
    """
    Safe wrapper around fg.read() that always returns
    a clean dataframe with no index issues.
    """
    df = fg.read()
    df = df.reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(None)  # convert UTC to tz-naive
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_recent_from_hopsworks(fg) -> pd.DataFrame:
    """
    Pull last 24 rows from Hopsworks.
    These are used to compute lag features for the current hour.
    """
    print("\nFetching last 24 hours from Hopsworks...")

    df = read_fg (fg)

    # Keep only last 24 rows
    df_recent = df.tail(24).copy()

    print(f" Got {len(df_recent)} rows from Hopsworks")
    print(f"  From: {df_recent['timestamp'].iloc[0]}")
    print(f"  To:   {df_recent['timestamp'].iloc[-1]}")

    return df_recent


# 4. Engineer features for current row ───────────────

def build_current_row( df_history: pd.DataFrame, df_current: pd.DataFrame ) -> pd.DataFrame:
    """
    Combine history + current hour.
    """
    print("\nEngineering features...")

    # Combine history and current hour
    df_combined = pd.concat(
        [df_history, df_current],
        ignore_index=True
    )
    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)

    # Call the shared function — not duplicated logic
    df_engineered = engineer_features(df_combined)

    # Return only the last row = current hour
    current_row = df_engineered.tail(1).copy()

   # Targets are NaN for now — filled later by fill_past_targets()
    current_row["target_day1"] = None
    current_row["target_day2"] = None
    current_row["target_day3"] = None

    print(f"Built row for {current_row['timestamp'].values[0]}")
    print(f"Targets: NaN (will be filled in 24-72 hours)")
    return current_row

# Step 5: Fill targets for past rows ─────────────────────

def fill_past_targets(fg):
    """
    Rows pushed in previous hours now have known futures.
    Row from 24h ago →  when target_day1 is now knowable.
    Row from 48h ago → when target_day2 is now knowable.
    Row from 72h ago → when target_day3 is now knowable.
    """
    print("\nFilling past targets...")

    df = read_fg (fg)

    # Recompute all targets
    df["target_day1"] = (
        df["aqi"].shift(-24)
        .rolling(window=24, min_periods=24)
        .mean()
        .shift(-23)
    )
    df["target_day2"] = (
        df["aqi"].shift(-48)
        .rolling(window=24, min_periods=24)
        .mean()
        .shift(-23)
    )
    df["target_day3"] = (
        df["aqi"].shift(-72)
        .rolling(window=24, min_periods=24)
        .mean()
        .shift(-23)
    )

    # Only update rows where target just became available
    now  = pd.Timestamp.now()
    mask = (
        (df["timestamp"] >= now - pd.Timedelta(hours=96)) &
        (df["timestamp"] <= now - pd.Timedelta(hours=24)) &
        (df["target_day1"].notna())
    )
    df_to_update = df[mask]

    if len(df_to_update) > 0:
        fg.insert(
            df_to_update,
            write_options={"wait_for_job": True}
        )
        print(f"Filled targets for {len(df_to_update)} past rows")
    else:
        print("  No past targets to fill yet")

# Main Function

if __name__ == "__main__":
    print("=" * 45)
    print(f"Feature Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 45)

    fg = connect_hopsworks()

    df_current = fetch_current_hour()

    df_history = fetch_recent_from_hopsworks(fg)

    df_row = build_current_row(df_history, df_current)

    # Sanity check
    nan_count = df_row[MODEL_FEATURES].isnull().sum().sum()
    if nan_count > 0:
        print(f"\n{nan_count} NaN values in features:")
        print(df_row[MODEL_FEATURES].isnull().sum()[
            df_row[MODEL_FEATURES].isnull().sum() > 0
        ])
    else:
        # Push current row
        print("\nPushing to Hopsworks...")
        df_row["o3"] = df_row["o3"].astype(int)
        fg.insert(df_row, write_options={"wait_for_job": True})
        print(f" Success: Pushed current row")

        # Fill targets for rows that now have known futures
        fill_past_targets(fg)

        print("\nPipeline completed")