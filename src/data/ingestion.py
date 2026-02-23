"""Ingestion et validation du dataset NASA C-MAPSS. TODO: Completez toutes les fonctions."""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from src.config import COLUMN_NAMES, RAW_DIR, FEATURE_DIR, RUL_CLIP, DROP_SENSORS

def compute_file_checksum(filepath: Path) -> str:
    return hashlib.md5(filepath.read_bytes()).hexdigest()

def load_raw_file(filepath: Path) -> pd.DataFrame:
    """TODO: Lire le fichier (sep=espaces, pas de header), garder 26 colonnes, renommer avec COLUMN_NAMES, caster unit_id et cycle en int."""
    raise NotImplementedError("TODO: Implementer load_raw_file")

def add_rul(df: pd.DataFrame, clip: int = RUL_CLIP) -> pd.DataFrame:
    """TODO: Pour chaque moteur, RUL = max_cycle - cycle. Clipper a 125."""
    raise NotImplementedError("TODO: Implementer add_rul")

def validate_dataframe(df: pd.DataFrame) -> dict:
    """TODO: Verifier n_rows, n_units, missing_pct, duplicate_rows, cycles_monotonic, sensors zero variance."""
    raise NotImplementedError("TODO: Implementer validate_dataframe")

def drop_low_variance_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: Supprimer les colonnes listees dans DROP_SENSORS."""
    raise NotImplementedError("TODO: Implementer drop_low_variance_sensors")

def ingest_fd001():
    """TODO: Pipeline complet : charger, ajouter RUL, valider, sauver Parquet + manifest."""
    raise NotImplementedError("TODO: Implementer ingest_fd001")

def main():
    print("Ingestion NASA C-MAPSS FD001")
    ingest_fd001()

if __name__ == "__main__":
    main()
