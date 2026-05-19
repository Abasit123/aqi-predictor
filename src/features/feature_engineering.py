# feature_engineering.py

import pandas as pd
import numpy as np

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hopsworks doesn't accept brackets or special chars in column names.
    Rename all columns to snake_case before pushing.
    """
    df = df.rename(columns={
        "temperature(°C)":      "temperature_c",
        "humidity(%)":          "humidity_pct",
        "wind_speed(km/h)":     "wind_speed_kmh",
        "cloud_cover(%)":       "cloud_cover_pct",
        "pressure(hPa)":        "pressure_hpa",
    })
    return df

# From EDA, we understood, Hours and months had clear impact on aqi
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:

    # Cyclical encoding (To make sure that the model understands the cyclic nature of the time)

    df["hour_sin"]  = np.sin(2 * np.pi * df["timestamp"].dt.hour  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["timestamp"].dt.hour  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.month / 12)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From EDA:
    lag 1h  = 0.99 correlation
    lag 6h  = 0.89 correlation
    lag 12h = 0.78 correlation
    lag 24h = 0.71 correlation
    These are most powerful features.

    """
    df["aqi_lag_1h"]  = df["aqi"].shift(1)
    df["aqi_lag_6h"]  = df["aqi"].shift(6)
    df["aqi_lag_12h"] = df["aqi"].shift(12)
    df["aqi_lag_24h"] = df["aqi"].shift(24)

    # Also lagging the strongest pollutants
    df["pm25_lag_1h"] = df["pm25"].shift(1)
    df["pm10_lag_1h"] = df["pm10"].shift(1)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:

   # Rolling averages smooth out noise and capture trends.
    
    df["aqi_roll_mean_3h"]  = df["aqi"].rolling(window=3,  min_periods=1).mean()
    df["aqi_roll_mean_6h"]  = df["aqi"].rolling(window=6,  min_periods=1).mean()
    df["aqi_roll_mean_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()

    # Rate of change

    df["aqi_change_1h"] = df["aqi"].diff(1)
    df["aqi_change_6h"] = df["aqi"].diff(6)

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each target = average AQI over the next 24 hour window.

    target_day1 = mean AQI over hours 1 -> 24  from now
    target_day2 = mean AQI over hours 25 ->  48 from now
    target_day3 = mean AQI over hours 49 -> 72 from now
    """

    # Day 1 — average over next 24 hours
    df["target_day1"] = (
        df["aqi"]
        .shift(-24)                  # Starts 24 hours from now
        .rolling(window=24, min_periods=24) .mean() # Calculates the mean for the next 24 hours
        .shift(-23)                    # align back to current row
    )

    # Day 2 — average over hours 25–48
    df["target_day2"] = (
        df["aqi"]
        .shift(-48)
        .rolling(window=24, min_periods=24)
        .mean()
        .shift(-23)
    )

    # Day 3 — average over hours 49–72
    df["target_day3"] = (
        df["aqi"]
        .shift(-72)
        .rolling(window=24, min_periods=24)
        .mean()
        .shift(-23)
    )

    return df

# Master Fuction

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = clean_column_names(df) 

    print("Adding time features...")
    df = add_time_features(df)

    print("Adding lag features...")
    df = add_lag_features(df)

    print("Adding rolling features...")
    df = add_rolling_features(df)

    print("Adding target...")
    df = add_target(df)

    """   
     ---- WHEN PUSHING HISTORIC DATA  -----

    Drop rows where lags or target are NaN
    First 24 rows have no lag history
    Last 72 rows have no future target


    before = len(df)
    df = df.dropna(subset=[
    "aqi_lag_24h",          # need full lag history
    "target_day1",           # need full future targets
    "target_day2",
    "target_day3"
     ])
    after = len(df)
    print(f"Dropped {before - after} rows with NaN lags/target")
    print(f"Final shape: {df.shape}")
    """

    return df

# FINAL LIST OF FEATURES USED BY THE MODEL

MODEL_FEATURES = [
    "pm25", "pm10", "no2", "o3",
    "pm25_lag_1h", "pm10_lag_1h",

    "wind_speed_kmh",    # was wind_speed(km/h)
    "humidity_pct",      # was humidity(%)
    "cloud_cover_pct",   # was cloud_cover(%)
    "wind_dir",
    "temperature_c",     # was temperature(°C)
    "pressure_hpa",      # was pressure(hPa)

    "aqi","aqi_lag_1h", "aqi_lag_6h",
    "aqi_lag_12h", "aqi_lag_24h",

    "aqi_roll_mean_3h", "aqi_roll_mean_6h", "aqi_roll_mean_24h",
    "aqi_change_1h", "aqi_change_6h",

    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
]

# TARGET FEATURES THAT THE MODEL WILL USE

TARGET = ["target_day1", "target_day2", "target_day3"]