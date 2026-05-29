import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def fetch_last_5_rows():
    print("Connecting to MongoDB...")
    
    # 1. Initialize client using your environment variables
    client = MongoClient(
        os.getenv("MONGO_URI"),
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000
    )
    
    db = client[os.getenv("MONGO_DB", "aqi_predictor")]
    collection = db["aqi_features"]

    print("Fetching last 5 rows sorted by timestamp...")
    
    # 2. Query MongoDB: Sort by timestamp descending (-1) and limit to 5
    records = list(
        collection.find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(5)
    )

    if not records:
        print("❌ No data found in MongoDB collection 'aqi_features'.")
        return pd.DataFrame()

    # 3. Convert to DataFrame
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # 4. Optional: Sort ascending so the most recent row appears at the bottom of your terminal
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    print(f"📊 Successfully retrieved {len(df)} rows.")
    return df

if __name__ == "__main__":
    print("=" * 50)
    last_5_df = fetch_last_5_rows()
    print("=" * 50)
    
    if not last_5_df.empty:
        # Display specific columns for a cleaner terminal view
        preview_cols = ["timestamp", "aqi", "target_day1", "target_day2", "target_day3"]
        available_cols = [col for col in preview_cols if col in last_5_df.columns]
        
        print("\n--- Target & AQI Preview ---")
        print(last_5_df[available_cols].to_string(index=False))
        
        print("\n--- Full Data Matrix ---")
        print(last_5_df)