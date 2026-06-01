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


# ── Step 1 — Fetch current hour + forecasts from Open-Meteo ──

def fetch_current_hour() -> pd.DataFrame:
    print("\nFetching current hour from Open-Meteo...")
    tz  = pytz.timezone(TIMEZONE)
    now = datetime.now(tz).replace(minute=0, second=0, microsecond=0)
    current_time_str = now.strftime("%Y-%m-%dT%H:%M")

    # forecast_days=4 covers current hour + 72h ahead
    r_weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":   LAT,
            "longitude":  LON,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
                "cloud_cover",
            ],
            "wind_speed_unit": "kmh",
            "forecast_days":   4,
            "timezone":        TIMEZONE
        }
    ).json()

    r_air = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude":   LAT,
            "longitude":  LON,
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

    hw    = r_weather["hourly"]
    ha    = r_air["hourly"]
    times = hw["time"]

    try:
        idx = times.index(current_time_str)
    except ValueError:
        raise ValueError(
            f"Current hour {current_time_str} not found in forecast data"
        )

    def forecast_val(field: str, offset: int):
        """Safely get forecast value at current index + offset."""
        target_idx = idx + offset
        if target_idx < len(hw[field]):
            v = hw[field][target_idx]
            return float(v) if v is not None else None
        return None

    df = pd.DataFrame([{
        # ── Current observations ──────────────────────────────
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

        # ── Forecast weather at +48h ──────────────────────────
        # Mirrors shift(-48) columns computed in training mode
        "temp_forecast_48h":     forecast_val("temperature_2m",       48),
        "humidity_forecast_48h": forecast_val("relative_humidity_2m", 48),
        "wind_forecast_48h":     forecast_val("wind_speed_10m",       48),
        "cloud_forecast_48h":    forecast_val("cloud_cover",          48),

        # ── Forecast weather at +72h ──────────────────────────
        "temp_forecast_72h":     forecast_val("temperature_2m",       72),
        "humidity_forecast_72h": forecast_val("relative_humidity_2m", 72),
        "wind_forecast_72h":     forecast_val("wind_speed_10m",       72),
        "cloud_forecast_72h":    forecast_val("cloud_cover",          72),
    }])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  Current hour:        {df['timestamp'].values[0]}")
    print(f"  AQI:                 {df['aqi'].values[0]}")
    print(f"  temp_forecast_48h:   {df['temp_forecast_48h'].values[0]}")
    print(f"  wind_forecast_48h:   {df['wind_forecast_48h'].values[0]}")
    print(f"  temp_forecast_72h:   {df['temp_forecast_72h'].values[0]}")
    print(f"  wind_forecast_72h:   {df['wind_forecast_72h'].values[0]}")
    return df


# ── Step 2 — Fetch recent history from MongoDB ───────────────

def fetch_recent_from_mongo() -> pd.DataFrame:
    print("\nFetching last 72 hours from MongoDB...")

    # Need 72 rows for aqi_lag_72h
    df = load_recent(n=72)
    df = df.dropna(subset=["aqi"])

    if df.empty:
        print("  No history found — lag features will be NaN")
        return df

    print(f"  Got {len(df)} clean rows from MongoDB")
    print(f"  From: {df['timestamp'].iloc[0]}")
    print(f"  To:   {df['timestamp'].iloc[-1]}")
    return df


# ── Step 3 — Build current row ───────────────────────────────

def build_current_row(df_history: pd.DataFrame,
                      df_current: pd.DataFrame) -> pd.DataFrame:
    print("\nEngineering features...")

    # Forecast columns are already in df_current from the API call
    # Concat history + current for lag/rolling computation
    df_combined = pd.concat(
        [df_history, df_current],
        ignore_index=True
    ).sort_values("timestamp").reset_index(drop=True)

    # Inference mode:
    # - skips shift(-48)/shift(-72) forecast computation (already filled above)
    # - skips targets and dropna
    df_engineered = engineer_features(df_combined, mode="inference")

    current_row = df_engineered.tail(1).copy()

    # Targets unknown — filled later by fill_past_targets()
    current_row["target_24h"] = np.nan
    current_row["target_48h"] = np.nan
    current_row["target_72h"] = np.nan

    print(f"  Built row for {current_row['timestamp'].values[0]}")
    print(f"  Targets: NaN (will be filled in 24-72 hours)")
    return current_row


# ── Step 4 — Fill past targets ───────────────────────────────

def fill_past_targets():
    print("\nFilling past targets...")

    df = load_recent(n=96)
    if df.empty:
        print("  Not enough data")
        return

    today = pd.Timestamp.now().normalize()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()

    # Ensure target columns exist
    for col in ["target_24h", "target_48h", "target_72h"]:
        if col not in df.columns:
            df[col] = np.nan

    # Backup original targets to detect newly filled rows
    df["orig_24h"] = df["target_24h"]
    df["orig_48h"] = df["target_48h"]
    df["orig_72h"] = df["target_72h"]

    # Count rows per day — guards against incomplete days
    rows_per_day = df.groupby("date")["aqi"].count().reset_index()
    rows_per_day.columns = ["date", "row_count"]

    daily_avg = df.groupby("date")["aqi"].mean().reset_index()
    daily_avg.columns = ["date", "daily_avg_aqi"]
    daily_avg = daily_avg.merge(rows_per_day, on="date")

    # Only complete past days with at least 12 rows
    complete_days = daily_avg[
        (daily_avg["date"] < today) &
        (daily_avg["row_count"] >= 12)
    ].copy().sort_values("date").reset_index(drop=True)

    if len(complete_days) < 2:
        print("  Not enough complete days yet")
        return

    print(f"  Complete days: {complete_days['date'].dt.date.tolist()}")

    # Compute rolling targets from daily averages
    complete_days["target_24h"] = (
        complete_days["daily_avg_aqi"]
        .shift(-1).rolling(window=1, min_periods=1).mean()
    )
    complete_days["target_48h"] = (
        complete_days["daily_avg_aqi"]
        .shift(-2).rolling(window=1, min_periods=1).mean()
    )
    complete_days["target_72h"] = (
        complete_days["daily_avg_aqi"]
        .shift(-3).rolling(window=1, min_periods=1).mean()
    )

    # Drop old targets before merge to avoid _x/_y suffixes
    df = df.drop(columns=["target_24h", "target_48h", "target_72h"])

    df = df.merge(
        complete_days[["date", "target_24h", "target_48h", "target_72h"]],
        on="date",
        how="left"
    )

    # Only update rows that transitioned from NaN → value
    newly_filled = (
        (df["orig_24h"].isna() & df["target_24h"].notna()) |
        (df["orig_48h"].isna() & df["target_48h"].notna()) |
        (df["orig_72h"].isna() & df["target_72h"].notna())
    )

    df_to_update = df[
        (df["date"] < today) & newly_filled
    ].drop(columns=["date", "orig_24h", "orig_48h", "orig_72h"])

    if len(df_to_update) > 0:
        update_targets(df_to_update)
        print(f"  Filled targets for {len(df_to_update)} rows across "
              f"{df_to_update['timestamp'].dt.normalize().nunique()} days")
    else:
        print("  No new targets to fill")


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 45)
    print(f"Feature Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 45)

    df_current = fetch_current_hour()
    df_history = fetch_recent_from_mongo()
    df_row     = build_current_row(df_history, df_current)

    nan_count = df_row[MODEL_FEATURES].isnull().sum().sum()
    if nan_count > 0:
        print(f"\n{nan_count} NaN values in features:")
        print(df_row[MODEL_FEATURES].isnull().sum()[
            df_row[MODEL_FEATURES].isnull().sum() > 0
        ])
    else:
        KEEP_COLS = (
            ["timestamp"]
            + MODEL_FEATURES
            + ["target_24h", "target_48h", "target_72h"]
        )
        df_final = df_row[
            [c for c in KEEP_COLS if c in df_row.columns]
        ].copy()

        print("\nPushing to MongoDB...")
        save_features(df_final)
        print("  ✓ Pushed current row")

        fill_past_targets()
        print("\n✓ Pipeline complete")