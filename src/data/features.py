"""Feature engineering. TODO: Completez toutes les fonctions."""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import FEATURE_DIR, ROLLING_WINDOWS, SEQUENCE_LENGTH

def add_rolling_features(df, windows=None):
    """TODO: Rolling mean et std par capteur, par unit_id, pour chaque fenetre."""
    raise NotImplementedError("TODO")

def add_delta_features(df):
    """TODO: diff() par capteur, par unit_id. Remplir NaN avec 0."""
    raise NotImplementedError("TODO")

def normalize_features(df, scaler=None, fit=True):
    """TODO: StandardScaler sur les features (exclure unit_id, cycle, rul). fit=True pour train."""
    raise NotImplementedError("TODO")

def create_sequences(data, targets, seq_len=SEQUENCE_LENGTH):
    """TODO: Creer des sequences (n_seq, seq_len, n_features) et targets (n_seq,)."""
    raise NotImplementedError("TODO")

def create_sequences_by_unit(df, feature_cols, seq_len=SEQUENCE_LENGTH):
    """TODO: Creer des sequences par unite moteur pour eviter le melange."""
    raise NotImplementedError("TODO")

def build_features(save=True):
    """TODO: Charger Parquet, rolling, delta, normaliser, sauver."""
    raise NotImplementedError("TODO")

def main():
    build_features()

if __name__ == "__main__":
    main()
