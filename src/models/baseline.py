"""Utilitaires pour les modèles tabulaires (RF, XGBoost, LightGBM)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FEATURE_DIR, RANDOM_SEED, TEST_SIZE


def prepare_tabular_data(df: pd.DataFrame) -> tuple:
    """Séparer features/target, split train/val.
    Returns: X_train, X_val, y_train, y_val, feature_cols
    """
    exclude = {"unit_id", "cycle", "rul"}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df["rul"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    return X_train, X_val, y_train, y_val, feature_cols


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculer MAE, RMSE, R², sMAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # sMAPE : Symmetric Mean Absolute Percentage Error
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(
        np.where(denominator == 0, 0, np.abs(y_true - y_pred) / denominator)
    ) * 100

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "smape": round(float(smape), 4),
    }


def load_featured_data() -> pd.DataFrame:
    """Charger les features enrichies depuis le Parquet."""
    path = FEATURE_DIR / "train_FD001_featured.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} non trouvé. Lancez: make data"
        )
    return pd.read_parquet(path)
