"""Feature engineering pour le dataset NASA C-MAPSS."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_DIR, ROLLING_WINDOWS, SEQUENCE_LENGTH


def add_rolling_features(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    """Rolling mean et std par capteur, par unit_id, pour chaque fenêtre."""
    if windows is None:
        windows = ROLLING_WINDOWS

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    df = df.copy()

    for window in windows:
        for col in sensor_cols:
            grp = df.groupby("unit_id")[col]
            df[f"{col}_roll_mean_{window}"] = grp.transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
            df[f"{col}_roll_std_{window}"] = grp.transform(
                lambda s: s.rolling(window, min_periods=1).std().fillna(0)
            )
    return df


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variation cycle-à-cycle Δs(t) = s(t) - s(t-1) par capteur, par unit_id."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    df = df.copy()

    for col in sensor_cols:
        df[f"{col}_delta"] = df.groupby("unit_id")[col].diff().fillna(0)

    return df


def normalize_features(
    df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True
) -> tuple[pd.DataFrame, StandardScaler]:
    """StandardScaler sur les features (exclure unit_id, cycle, rul).
    fit=True pour le train, fit=False pour le test (passe le scaler en argument).
    """
    exclude = {"unit_id", "cycle", "rul"}
    feature_cols = [c for c in df.columns if c not in exclude]

    df = df.copy()

    if fit:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        if scaler is None:
            raise ValueError("scaler doit être fourni quand fit=False")
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, scaler


def create_sequences(
    data: np.ndarray, targets: np.ndarray, seq_len: int = SEQUENCE_LENGTH
) -> tuple[np.ndarray, np.ndarray]:
    """Créer des séquences (n_seq, seq_len, n_features) et targets (n_seq,)."""
    n_samples = len(data) - seq_len + 1
    if n_samples <= 0:
        return np.empty((0, seq_len, data.shape[1])), np.empty((0,))

    X = np.stack([data[i : i + seq_len] for i in range(n_samples)])
    y = targets[seq_len - 1 :]
    return X, y


def create_sequences_by_unit(
    df: pd.DataFrame, feature_cols: list, seq_len: int = SEQUENCE_LENGTH
) -> tuple[np.ndarray, np.ndarray]:
    """Créer des séquences par unité moteur pour éviter le mélange entre moteurs."""
    X_list, y_list = [], []

    for _, unit_df in df.groupby("unit_id"):
        unit_df = unit_df.sort_values("cycle")
        data = unit_df[feature_cols].values
        targets = unit_df["rul"].values

        X_unit, y_unit = create_sequences(data, targets, seq_len)
        if len(X_unit) > 0:
            X_list.append(X_unit)
            y_list.append(y_unit)

    if not X_list:
        n_features = len(feature_cols)
        return np.empty((0, seq_len, n_features)), np.empty((0,))

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def build_features(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Charger Parquet, rolling, delta, normaliser, sauvegarder."""
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    train_path = FEATURE_DIR / "train_FD001.parquet"
    test_path = FEATURE_DIR / "test_FD001.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} non trouvé. Lancez: make data")

    print("Chargement des données ingérées ...")
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    print("Rolling features ...")
    df_train = add_rolling_features(df_train)
    df_test = add_rolling_features(df_test)

    print("Delta features ...")
    df_train = add_delta_features(df_train)
    df_test = add_delta_features(df_test)

    print("Normalisation ...")
    df_train, scaler = normalize_features(df_train, fit=True)
    df_test, _ = normalize_features(df_test, scaler=scaler, fit=False)

    if save:
        out_train = FEATURE_DIR / "train_FD001_featured.parquet"
        out_test = FEATURE_DIR / "test_FD001_featured.parquet"
        df_train.to_parquet(out_train, index=False)
        df_test.to_parquet(out_test, index=False)
        print(f"Features sauvegardées: {out_train}, {out_test}")

        scaler_path = FEATURE_DIR / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"Scaler sauvegardé: {scaler_path}")

    return df_train, df_test, scaler


def main():
    build_features()


if __name__ == "__main__":
    main()
