from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "images"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_GAMMA = 1.8
DEFAULT_CLAHE_CLIP = 2.0
DEFAULT_THICKNESS = 3
