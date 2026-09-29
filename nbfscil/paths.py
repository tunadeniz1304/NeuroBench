import os

# Dataset lives on fast local disk, never inside the repo or Drive.
NB_DATA = os.environ.get("NB_DATA", "/content/data")
DATA_ROOT = NB_DATA  # neurobench's MSWC expects <root>/MSWC/...
CACHE_DIR = os.path.join(NB_DATA, "cache")
