# src/data/mongo_store.py

import os
import pandas as pd
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

load_dotenv()


def get_collection():
    client = MongoClient(
        os.getenv("MONGO_URI"),
        serverSelectionTimeoutMS=30000,   # 30s to find server
        connectTimeoutMS=30000,           # 30s to connect
        socketTimeoutMS=60000,            # 60s for operations
        maxPoolSize=1,                    # single connection for scripts
        retryWrites=True
    )
    db = client[os.getenv("MONGO_DB", "aqi_predictor")]
    return db["aqi_features"]

def save_features(df: pd.DataFrame):
    from pymongo import UpdateOne

    collection = get_collection()
    records    = df.to_dict(orient="records")

    # Build operations
    operations = []
    for record in records:
        record["timestamp"] = str(record["timestamp"])
        operations.append(
            UpdateOne(
                {"timestamp": record["timestamp"]},
                {"$set": record},
                upsert=True
            )
        )

    # Execute in batches of 500
    batch_size = 500
    total      = 0

    for i in range(0, len(operations), batch_size):
        batch  = operations[i:i + batch_size]
        result = collection.bulk_write(batch, ordered=False)
        total += result.upserted_count + result.modified_count
        print(f"  Batch {i//batch_size + 1}: {len(batch)} rows done")

    print(f"Saved {len(records)} rows to MongoDB")

def load_features() -> pd.DataFrame:
    """Load all features sorted by timestamp."""
    collection = get_collection()
    records    = list(collection.find({}, {"_id": 0}))

    if not records:
        print("No data found in MongoDB")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} rows from MongoDB")
    return df


def load_recent(n: int = 24) -> pd.DataFrame:
    """Load last n rows — used for lag features."""
    collection = get_collection()

    records = list(
        collection.find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(n)
    )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def update_targets(df: pd.DataFrame):
    """Update target columns for past rows."""
    collection = get_collection()
    updated    = 0

    for _, row in df.iterrows():
        result = collection.update_one(
            {"timestamp": str(row["timestamp"])},
            {"$set": {
                "target_day1": row["target_day1"],
                "target_day2": row["target_day2"],
                "target_day3": row["target_day3"],
            }}
        )
        if result.modified_count > 0:
            updated += 1

    print(f"Updated targets for {updated} rows")


def create_index():
    """
    Create index on timestamp for fast queries.
    Run once during setup.
    """
    collection = get_collection()
    collection.create_index(
        [("timestamp", ASCENDING)],
        unique=True
    )
    print("Index created on timestamp")