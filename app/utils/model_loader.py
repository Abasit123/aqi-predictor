import os
import sys
import joblib
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hopsworks
from dotenv import load_dotenv

load_dotenv()

# Horizons and their fallback model names
# Used if MongoDB is empty or unreachable
BEST_MODELS = {
    "24h": "linear_regression",
    "48h": "linear_regression",
    "72h": "linear_regression",
}


def load_models_from_registry() -> dict:
    """
    Load best model for each horizon from Hopsworks.
    Returns dict: {"24h": model, "48h": model, "72h": model}
    """
    print("Loading models from Hopsworks...")

    # 1. Dynamically read which architectures won from MongoDB metrics
    dynamic_best_models = get_best_model_names_from_mongo()

    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    mr = project.get_model_registry()

    models = {}
    for horizon, model_name in dynamic_best_models.items():
        registry_name = f"aqi_{model_name}_{horizon}"
        try:
            # SOLUTION 2: Grab the version of this specific architecture with the lowest MAE
            model_obj = mr.get_best_model(registry_name, metric="mae", direction="min")
            model_dir = model_obj.download()
            
            # Robust file loading: Try your specific format first, fallback to any .pkl if missing
            expected_pkl_path = Path(model_dir) / f"{model_name}_{horizon}.pkl"
            if expected_pkl_path.exists():
                model = joblib.load(str(expected_pkl_path))
            else:
                # Fallback safeguard: grab the first .pkl artifact in the directory
                pkl_files = list(Path(model_dir).glob("*.pkl"))
                model = joblib.load(str(pkl_files[0]))

            models[horizon] = model
            print(f"✓ Loaded {registry_name} v{model_obj.version} (Champion Version)")
            
        except Exception as e:
            print(f"✗ Failed to load {registry_name}: {e}")
            models[horizon] = None

    return models


def get_best_model_names_from_mongo() -> dict:
    """
    Read best model names dynamically from MongoDB metrics.
    Dashboard always uses whatever trained best.
    """
    from pymongo import MongoClient
    
    try:
        # Integrated tlsCAFile flag to prevent Windows verification timeouts
        client = MongoClient(os.getenv("MONGO_URI"))
        collection = client[os.getenv("MONGO_DB")]["model_metrics"]

        best = {}
        for horizon in ["24h", "48h", "72h"]:
            records = list(collection.find(
                {"horizon": horizon},
                {"_id": 0, "model_name": 1, "mae": 1, "r2": 1}
            ))
            if records:
                # Identify the model architecture with the absolute lowest Mean Absolute Error
                best_record = min(records, key=lambda x: x["mae"])
                best[horizon] = best_record["model_name"]

        return best if best else BEST_MODELS
        
    except Exception as e:
        print(f"⚠️ MongoDB lookup failed ({e}). Falling back to default dictionary mappings.")
        return BEST_MODELS