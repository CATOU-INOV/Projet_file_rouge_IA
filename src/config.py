"""Configuration centralisee du projet."""
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURE_DIR = DATA_DIR / "features"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "turbofan-rul-prediction"
MODEL_NAME = os.getenv("MODEL_NAME", "turbofan_rul_predictor")

COLUMN_NAMES = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# TODO: Identifiez les capteurs a variance quasi-nulle sur FD001 et ajoutez-les ici
DROP_SENSORS = []

RUL_CLIP = int(os.getenv("RUL_CLIP", "125"))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "50"))
RANDOM_SEED = 42
TEST_SIZE = 0.2
ROLLING_WINDOWS = [10, 20, 50]
