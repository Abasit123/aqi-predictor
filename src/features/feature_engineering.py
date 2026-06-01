# src/features/feature_engineering.py

import pandas as pd
import numpy as np


# ── Step 0 — Clean column names ──────────────────────────────

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw API column names to clean snake_case.
    Called first before any feature engineering.
    """
    df = df.rename(columns={
        "temperature(°C)":          "temperature_c",
        "humidity(%)":              "humidity_pct",
        "pressure(hPa)":            "pressure_hpa",
        "wind_speed(km/h)":         "wind_speed_kmh",
        "cloud_cover(%)":           "cloud_cover_pct",
    })
    return df


# ── Step 1 — Time features ───────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_sin"]      = np.sin(2 * np.pi * df["timestamp"].dt.hour      / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * df["timestamp"].dt.hour      / 24)
    df["month_sin"]     = np.sin(2 * np.pi * df["timestamp"].dt.month     / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["timestamp"].dt.month     / 12)

    return df

# ── Step 2 — Lag features ────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    # AQI lags
    df["aqi_lag_1h"]  = df["aqi"].shift(1)
    df["aqi_lag_6h"]  = df["aqi"].shift(6)
    df["aqi_lag_12h"] = df["aqi"].shift(12)
    df["aqi_lag_24h"] = df["aqi"].shift(24)
    df["aqi_lag_48h"] = df["aqi"].shift(48)
    df["aqi_lag_72h"] = df["aqi"].shift(72)

    # Pollutant lags
    df["pm25_lag_1h"]  = df["pm25"].shift(1)
    df["pm10_lag_1h"]  = df["pm10"].shift(1)
    df["pm25_lag_24h"] = df["pm25"].shift(24)
    df["pm10_lag_24h"] = df["pm10"].shift(24)
    df["no2_lag_24h"]  = df["no2"].shift(24)

    return df


# ── Step 3 — Rolling features ────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    # AQI rolling means
    df["aqi_roll_mean_3h"]  = df["aqi"].rolling(window=3,  min_periods=1).mean()
    df["aqi_roll_mean_6h"]  = df["aqi"].rolling(window=6,  min_periods=1).mean()
    df["aqi_roll_mean_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()
    df["aqi_roll_mean_48h"] = df["aqi"].rolling(window=48, min_periods=1).mean()
    df["aqi_roll_mean_72h"] = df["aqi"].rolling(window=72, min_periods=1).mean()

    # AQI rolling std — captures volatility
    df["aqi_roll_std_24h"] = df["aqi"].rolling(window=24, min_periods=2).std()
    df["aqi_roll_std_48h"] = df["aqi"].rolling(window=48, min_periods=2).std()

    # Rate of change
    df["aqi_change_1h"] = df["aqi"].diff(1)
    df["aqi_change_6h"] = df["aqi"].diff(6)

    # Pollutant rolling means
    df["pm25_roll_mean_24h"] = df["pm25"].rolling(window=24, min_periods=1).mean()
    df["pm10_roll_mean_24h"] = df["pm10"].rolling(window=24, min_periods=1).mean()

    return df


# ── Step 4 — Forecast features (training only) ───────────────

def add_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift actual future weather back to current row.
    Used in training mode only — gives model the true
    weather-AQI relationship at +48h and +72h.

    In inference mode, these columns are pre-filled from
    Open-Meteo forecast API before engineer_features() is called.
    """
    df["temp_forecast_48h"]     = df["temperature_c"].shift(-48)
    df["humidity_forecast_48h"] = df["humidity_pct"].shift(-48)
    df["wind_forecast_48h"]     = df["wind_speed_kmh"].shift(-48)
    df["cloud_forecast_48h"]    = df["cloud_cover_pct"].shift(-48)

    df["temp_forecast_72h"]     = df["temperature_c"].shift(-72)
    df["humidity_forecast_72h"] = df["humidity_pct"].shift(-72)
    df["wind_forecast_72h"]     = df["wind_speed_kmh"].shift(-72)
    df["cloud_forecast_72h"]    = df["cloud_cover_pct"].shift(-72)

    return df


# ── Step 5 — Targets ─────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling average AQI over next 24h, 48h, 72h windows.
    """
    df["target_24h"] = (
        df["aqi"].shift(-1)
        .rolling(window=24, min_periods=24)
        .mean()
        .shift(-23)
    )
    df["target_48h"] = (
        df["aqi"].shift(-1)
        .rolling(window=48, min_periods=48)
        .mean()
        .shift(-47)
    )
    df["target_72h"] = (
        df["aqi"].shift(-1)
        .rolling(window=72, min_periods=72)
        .mean()
        .shift(-71)
    )
    return df


# ── Master function ───────────────────────────────────────────

def engineer_features(df: pd.DataFrame, mode: str = "training") -> pd.DataFrame:
    """
    mode = "training"  → cleans names, computes forecast features via shift,
                         computes targets, drops NaN rows
    mode = "inference" → cleans names, skips shift-based forecasts
                         (already filled by feature pipeline),
                         skips targets and dropna
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("  Cleaning column names...")
    df = clean_column_names(df)

    print("  Adding time features...")
    df = add_time_features(df)

    print("  Adding lag features...")
    df = add_lag_features(df)

    print("  Adding rolling features...")
    df = add_rolling_features(df)

    if mode == "training":
        print("  Adding forecast features (shift-based)...")
        df = add_forecast_features(df)

        print("  Adding targets...")
        df = add_target(df)

        before = len(df)
        df = df.dropna(subset=[
            "aqi_lag_72h",          # need full lag history
            "temp_forecast_72h",    # need full forecast window
            "target_24h",           # need full target window
            "target_48h",
            "target_72h",
        ])
        print(f"  Dropped {before - len(df)} rows with NaN")

    else:
        print("  Inference mode — skipping shift forecasts and targets")

    print(f"  Final shape: {df.shape}")
    return df


# ── Feature lists ─────────────────────────────────────────────

IRRELEVANT_48H = {"aqi_lag_1h", "aqi_lag_6h", "aqi_change_1h"}
IRRELEVANT_72H = {"aqi_lag_1h", "aqi_lag_6h", "aqi_lag_12h",
                  "aqi_change_1h", "aqi_change_6h"}


# FEATURE IMPORTANCE TELLS THESE FEATURES ARE NOT USEFUL
# Remove from MODEL_FEATURES in feature_engineering.py
# "boundary_layer_h"   — never appears in any top 15
# "precipitation_mm"   — never appears in any top 15
# "cloud_cover_pct"    — never appears in any top 15
# "wind_dir"           — never appears in any top 15
# "no2_lag_24h"        — never appears in any top 15
# "aqi_roll_std_24h"   — never appears in any top 15
# "aqi_roll_std_48h"   — never appears in any top 15
MODEL_FEATURES = [
    # Raw pollutants
    "pm25", "pm10", "no2", "o3",

    # Pollutant lags
    "pm25_lag_1h",  "pm10_lag_1h",
    "pm25_lag_24h", "pm10_lag_24h", "no2_lag_24h",

    # Pollutant rolling
    "pm25_roll_mean_24h", "pm10_roll_mean_24h",

    # Raw weather
    "temperature_c", "humidity_pct", "wind_speed_kmh",
    "cloud_cover_pct", "wind_dir", "pressure_hpa",

    # AQI
    "aqi",
    "aqi_lag_1h",  "aqi_lag_6h",
    "aqi_lag_12h", "aqi_lag_24h",
    "aqi_lag_48h", "aqi_lag_72h",

    # AQI rolling
    "aqi_roll_mean_3h",  "aqi_roll_mean_6h",
    "aqi_roll_mean_24h", "aqi_roll_mean_48h", "aqi_roll_mean_72h",
    "aqi_roll_std_24h",  "aqi_roll_std_48h",

    # AQI change
    "aqi_change_1h", "aqi_change_6h",

    # Time
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",

    # Forecast weather
    "temp_forecast_48h",     "humidity_forecast_48h",
    "wind_forecast_48h",     "cloud_forecast_48h",
    "temp_forecast_72h",     "humidity_forecast_72h",
    "wind_forecast_72h",     "cloud_forecast_72h",
]

TARGET = ["target_24h", "target_48h", "target_72h"]


def get_features(horizon: str) -> list:
    if horizon == "24h":
        return MODEL_FEATURES
    elif horizon == "48h":
        return [f for f in MODEL_FEATURES if f not in IRRELEVANT_48H]
    else:
        return [f for f in MODEL_FEATURES if f not in IRRELEVANT_72H]