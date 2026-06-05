# app/utils/data_loader.py

import os
import pandas as pd
from pymongo   import MongoClient
from dotenv    import load_dotenv
from datetime  import datetime, timedelta

load_dotenv()


def get_mongo_uri():
    try:
        import streamlit as st
        return st.secrets["MONGO_URI"]
    except Exception:
        return os.getenv("MONGO_URI")


def get_mongo_db():
    try:
        import streamlit as st
        return st.secrets["MONGO_DB"]
    except Exception:
        return os.getenv("MONGO_DB")


def get_collection():
    client = MongoClient(
        get_mongo_uri(),
        serverSelectionTimeoutMS=10000
    )
    return client[get_mongo_db()]["aqi_features"]


def load_latest_row() -> dict:
    """Most recent hourly row — used for prediction."""
    collection = get_collection()
    record     = list(
        collection.find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(1)
    )
    if not record:
        raise ValueError("No data in MongoDB")

    row              = record[0]
    row["timestamp"] = pd.to_datetime(row["timestamp"])
    return row


def load_last_3_days() -> pd.DataFrame:
    """
    Last 3 days of data resampled to 3-hour intervals.
    Handles GitHub Actions delays gracefully.
    """
    collection = get_collection()
    since      = datetime.now() - timedelta(days=3)

    records = list(
        collection.find(
            {"timestamp": {"$gte": str(since)}},
            {"_id": 0, "timestamp": 1, "aqi": 1,
             "pm25": 1, "pm10": 1,
             "wind_speed_kmh": 1, "humidity_pct": 1}
        ).sort("timestamp", 1)
    )

    if not records:
        # Fall back to last 72 rows
        records = list(
            collection.find({}, {"_id": 0})
            .sort("timestamp", -1)
            .limit(72)
        )
        records = records[::-1]

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Resample to 3-hour intervals — handles gaps from GitHub delays
    df = df.set_index("timestamp")
    numeric_cols = df.select_dtypes(include="number").columns
    df = df[numeric_cols].resample("3h").mean().round(1)
    df = df.reset_index()
    df = df.dropna(subset=["aqi"])

    return df


def load_model_metrics() -> pd.DataFrame:
    """Load only best model metrics per horizon."""
    client     = MongoClient(get_mongo_uri())
    collection = client[get_mongo_db()]["model_metrics"]
    records    = list(collection.find({}, {"_id": 0}))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Keep only best model per horizon
    if "horizon" in df.columns and "mae" in df.columns:
        df = df.sort_values("mae").groupby("horizon").first().reset_index()

    return df