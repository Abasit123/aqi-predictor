# test_hopsworks.py
import os
import tempfile

os.environ["TMPDIR"]   = "C:\\tmp"
os.environ["TEMP"]     = "C:\\tmp"
os.environ["TMP"]      = "C:\\tmp"
tempfile.tempdir       = "C:\\tmp"

# Create the folder if it doesn't exist
os.makedirs("C:\\tmp", exist_ok=True)

import hopsworks
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(
    project=os.getenv("HOPSWORKS_PROJECT"),
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

fs = project.get_feature_store()
print("✓ Connected to:", project.name)
print("✓ Feature store ready")