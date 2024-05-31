import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "index")
CHUNK_SIZE = 64
CHUNK_STEP = 64