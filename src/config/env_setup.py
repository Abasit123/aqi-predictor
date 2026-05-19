# src/config/env_setup.py
import os
import tempfile
import platform

def setup():
    if platform.system() == "Windows":
        os.environ["TMPDIR"] = "C:\\tmp"
        os.environ["TEMP"]   = "C:\\tmp"
        os.environ["TMP"]    = "C:\\tmp"
        tempfile.tempdir     = "C:\\tmp"
        os.makedirs("C:\\tmp", exist_ok=True)
    # On Linux/Mac — tmp already works, nothing needed