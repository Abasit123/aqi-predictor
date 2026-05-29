import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

LAT      = float(os.getenv("LAT", 25.3960))
LON      = float(os.getenv("LON", 68.3578))
TIMEZONE = os.getenv("TIMEZONE", "Asia/Karachi")

START = "2025-05-28"
END   = "2026-05-27"

def fetch_historical_weather():
    print("Fetching historical weather...")
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LAT, "longitude": LON,
            "start_date": START, "end_date": END,
            "hourly": [
                "temperature_2m", "relative_humidity_2m",
                "precipitation", "surface_pressure",
                "wind_speed_10m", "wind_direction_10m",
                "cloud_cover", "boundary_layer_height"
            ],
            "timezone": TIMEZONE
        }
    ).json()

    h = r["hourly"]
    return pd.DataFrame({
        "timestamp":      h["time"],
        "temperature(°C)":    h["temperature_2m"],
        "humidity(%)":       h["relative_humidity_2m"],
        "precipitation(mm)":  h["precipitation"],
        "pressure(hPa)":       h["surface_pressure"],
        "wind_speed(km/h)":     h["wind_speed_10m"],
        "wind_dir":       h["wind_direction_10m"],
        "cloud_cover(%)":    h["cloud_cover"],
        "boundary_layer_h(meters)": h["boundary_layer_height"],
    })


def fetch_historical_air_quality():
    print("Fetching historical air quality...")
    r = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": LAT, "longitude": LON,
            "hourly": [
                "pm2_5", "pm10",
                "nitrogen_dioxide", "ozone",
                "european_aqi"
            ],
            "start_date": START, "end_date": END,
            "timezone": TIMEZONE
        }
    ).json()

    h = r["hourly"]
    return pd.DataFrame({
        "timestamp": h["time"],
        "pm25":      h["pm2_5"],
        "pm10":      h["pm10"],
        "no2":       h["nitrogen_dioxide"],
        "o3":        h["ozone"],
        "aqi":       h["european_aqi"],
    })


if __name__ == "__main__":
    # Fetch
    df_weather = fetch_historical_weather()
    df_air     = fetch_historical_air_quality()

    # Merge on timestamp
    df = pd.merge(df_weather, df_air, on="timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

# Saving locally

df.to_json("historical_data.json", orient="records", indent=2, date_format="iso")
print(f"Saved {len(df)} rows to historical_data.json")