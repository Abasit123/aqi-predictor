import os
import tempfile

os.environ["TMPDIR"] = "C:\\tmp"
os.environ["TEMP"]   = "C:\\tmp"
os.environ["TMP"]    = "C:\\tmp"

tempfile.tempdir = "C:\\tmp"

os.makedirs("C:\\tmp", exist_ok=True)