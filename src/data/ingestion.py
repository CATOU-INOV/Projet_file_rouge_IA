"""Ingestion et validation du dataset NASA C-MAPSS."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import COLUMN_NAMES, RAW_DIR, FEATURE_DIR, RUL_CLIP, DROP_SENSORS


def compute_file_checksum(filepath: Path) -> str:
    return hashlib.md5(filepath.read_bytes()).hexdigest()


def load_raw_file(filepath: Path) -> pd.DataFrame:
    """Lire un fichier C-MAPSS (sep=espaces, pas de header), renommer colonnes, caster types."""
    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    # Le dataset a parfois une colonne vide en fin de ligne – on garde les 26 premières
    df = df.iloc[:, :26]
    df.columns = COLUMN_NAMES
    df["unit_id"] = df["unit_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def add_rul(df: pd.DataFrame, clip: int = RUL_CLIP) -> pd.DataFrame:
    """Pour chaque moteur, RUL = max_cycle - cycle. Clipper à RUL_CLIP."""
    max_cycles = df.groupby("unit_id")["cycle"].transform("max")
    df = df.copy()
    df["rul"] = (max_cycles - df["cycle"]).clip(upper=clip)
    return df


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Vérifier la qualité du dataframe : doublons, monotonie, manquants, variance."""
    n_rows = len(df)
    n_units = df["unit_id"].nunique()
    missing_pct = df.isnull().mean().mean() * 100

    duplicate_rows = df.duplicated(subset=["unit_id", "cycle"]).sum()

    # Vérifier que les cycles sont monotones croissants par moteur
    cycles_monotonic = (
        df.groupby("unit_id")["cycle"]
        .apply(lambda s: s.is_monotonic_increasing)
        .all()
    )

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    sensor_variances = df[sensor_cols].var()
    zero_variance_sensors = sensor_variances[sensor_variances < 1e-4].index.tolist()

    report = {
        "n_rows": n_rows,
        "n_units": n_units,
        "missing_pct": round(missing_pct, 4),
        "duplicate_rows": int(duplicate_rows),
        "cycles_monotonic": bool(cycles_monotonic),
        "zero_variance_sensors": zero_variance_sensors,
    }

    assert missing_pct < 0.1, f"Trop de valeurs manquantes: {missing_pct:.2f}%"
    assert duplicate_rows == 0, f"{duplicate_rows} doublons (unit_id, cycle) détectés"
    assert cycles_monotonic, "Cycles non monotones pour certains moteurs"

    return report


def drop_low_variance_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Supprimer les capteurs listés dans DROP_SENSORS."""
    cols_to_drop = [c for c in DROP_SENSORS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def _identify_low_variance_sensors(df: pd.DataFrame, threshold: float = 1e-4) -> list:
    """Identifier les capteurs à variance quasi-nulle."""
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    variances = df[sensor_cols].var()
    return variances[variances < threshold].index.tolist()


def ingest_fd001():
    """Pipeline complet : charger, ajouter RUL, valider, sauver Parquet + manifest."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    train_path = RAW_DIR / "train_FD001.txt"
    test_path = RAW_DIR / "test_FD001.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} non trouvé. Lancez: make data")

    # --- Train ---
    print("Chargement train_FD001.txt ...")
    df_train = load_raw_file(train_path)
    df_train = add_rul(df_train)

    # Identifier automatiquement les capteurs à faible variance sur le train
    low_var = _identify_low_variance_sensors(df_train)
    if low_var:
        print(f"Capteurs à variance quasi-nulle détectés: {low_var}")
        df_train = df_train.drop(columns=low_var, errors="ignore")

    # Appliquer DROP_SENSORS configurés
    df_train = drop_low_variance_sensors(df_train)

    report_train = validate_dataframe(df_train)
    print(f"Validation train: {report_train}")

    train_parquet = FEATURE_DIR / "train_FD001.parquet"
    df_train.to_parquet(train_parquet, index=False)
    print(f"Sauvegardé: {train_parquet}")

    # --- Test ---
    print("Chargement test_FD001.txt ...")
    df_test = load_raw_file(test_path)

    # Supprimer les mêmes capteurs que sur le train
    df_test = df_test.drop(columns=low_var, errors="ignore")
    df_test = drop_low_variance_sensors(df_test)

    # Ajouter la RUL vraie si disponible
    rul_path = RAW_DIR / "RUL_FD001.txt"
    if rul_path.exists():
        rul_true = pd.read_csv(rul_path, header=None, names=["rul_true"])
        # La RUL vraie correspond au dernier cycle de chaque moteur
        last_cycles = df_test.groupby("unit_id").tail(1).copy()
        last_cycles = last_cycles.reset_index(drop=True)
        last_cycles["rul"] = rul_true["rul_true"].values.clip(max=RUL_CLIP)
        test_parquet = FEATURE_DIR / "test_FD001.parquet"
        df_test.to_parquet(test_parquet, index=False)
        last_cycles.to_parquet(FEATURE_DIR / "test_last_FD001.parquet", index=False)
    else:
        test_parquet = FEATURE_DIR / "test_FD001.parquet"
        df_test.to_parquet(test_parquet, index=False)
    print(f"Sauvegardé: {test_parquet}")

    # --- Manifest JSON avec checksums ---
    manifest = {
        "train_FD001": {
            "source": str(train_path),
            "checksum_md5": compute_file_checksum(train_path),
            "n_rows": report_train["n_rows"],
            "n_units": report_train["n_units"],
            "dropped_sensors": low_var + DROP_SENSORS,
        },
        "test_FD001": {
            "source": str(test_path),
            "checksum_md5": compute_file_checksum(test_path),
        },
    }
    manifest_path = FEATURE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest sauvegardé: {manifest_path}")

    return df_train, df_test


def main():
    print("=== Ingestion NASA C-MAPSS FD001 ===")
    ingest_fd001()
    print("Ingestion terminée.")


if __name__ == "__main__":
    main()
