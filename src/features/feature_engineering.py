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
    Targets = average AQI for each calendar day ahead.

    For each row at timestamp T:
    target_day1 = avg AQI for the next calendar day
    target_day2 = avg AQI for 2 calendar days ahead
    target_day3 = avg AQI for 3 calendar days ahead
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    # Compute daily averages
    daily_avg = df.groupby("date")["aqi"].mean().reset_index()
    daily_avg.columns = ["date", "daily_avg_aqi"]
    daily_avg["date"] = pd.to_datetime(daily_avg["date"])

    # Merge daily averages back
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(daily_avg, on="date", how="left")

    # Shift daily averages to get next 3 days
    daily_avg["target_day1"] = daily_avg["daily_avg_aqi"].shift(-1)
    daily_avg["target_day2"] = daily_avg["daily_avg_aqi"].shift(-2)
    daily_avg["target_day3"] = daily_avg["daily_avg_aqi"].shift(-3)

    # Merge targets back to hourly dataframe
    df = df.merge(
        daily_avg[["date", "target_day1", "target_day2", "target_day3"]],
        on="date",
        how="left"
    )

    df = df.drop(columns=["date", "daily_avg_aqi"])

    return df

# Master Fuction

def engineer_features(df: pd.DataFrame, mode: str = "training") -> pd.DataFrame:
    """
    mode = "training"  → used in backfill and training pipeline
                         computes targets, drops NaN rows
                         
    mode = "inference" → used in live feature pipeline
                         skips targets, keeps all rows
                         needs current aqi to compute lag features
    """
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

    if mode == "training":
        # Full historical data — compute targets and clean
        print("Adding targets...")
        df = add_target(df)

        before = len(df)
        df = df.dropna(subset=[
            "aqi_lag_24h",
            "target_day1",
            "target_day2",
            "target_day3"
        ])
        print(f"Dropped {before - len(df)} rows with NaN")

    else:
        # Live pipeline — no future data, skip targets and dropna
        print("Inference mode — skipping targets and dropna")

    print(f"Final shape: {df.shape}")
    return df

# FINAL LIST OF FEATURES USED BY THE MODEL

MODEL_FEATURES = [
    "pm25", "pm10", "no2", "o3",
    "pm25_lag_1h", "pm10_lag_1h",

    "wind_speed_kmh",   
    "humidity_pct",      
    "cloud_cover_pct",   
    "wind_dir",
    "temperature_c",     
    "pressure_hpa",      

    "aqi","aqi_lag_1h", "aqi_lag_6h",
    "aqi_lag_12h", "aqi_lag_24h",

    "aqi_roll_mean_3h", "aqi_roll_mean_6h", "aqi_roll_mean_24h",
    "aqi_change_1h", "aqi_change_6h",

    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
]

# TARGET FEATURES THAT THE MODEL WILL USE

TARGET = ["target_day1", "target_day2", "target_day3"]