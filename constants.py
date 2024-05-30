import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_PATH = os.path.join(BASE_DIR, ".pdfs")
INDEX_PATH = os.path.join(BASE_DIR, ".index")
CHUNK_SIZE = 128
CHUNK_STEP = 128