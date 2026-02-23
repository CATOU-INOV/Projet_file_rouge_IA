"""Utilitaires modeles tabulaires. TODO: Completez."""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import FEATURE_DIR, RANDOM_SEED, TEST_SIZE

def prepare_tabular_data(df):
    """TODO: Separer features/target, split train/val."""
    raise NotImplementedError("TODO")

def compute_metrics(y_true, y_pred):
    """TODO: Calculer mae, rmse, r2, smape."""
    raise NotImplementedError("TODO")

def load_featured_data():
    path = FEATURE_DIR / "train_FD001_featured.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} non trouve. Lancez make data.")
    return pd.read_parquet(path)
