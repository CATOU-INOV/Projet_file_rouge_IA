"""Tests unitaires – feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    add_rolling_features,
    add_delta_features,
    normalize_features,
    create_sequences,
    create_sequences_by_unit,
)
from src.config import ROLLING_WINDOWS


def _make_df(n_units: int = 2, cycles: int = 60) -> pd.DataFrame:
    rows = []
    for uid in range(1, n_units + 1):
        for cycle in range(1, cycles + 1):
            row = {"unit_id": uid, "cycle": cycle, "rul": float(cycles - cycle)}
            for s in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]:
                row[f"sensor_{s}"] = float(np.random.rand())
            rows.append(row)
    return pd.DataFrame(rows)


class TestRollingFeatures:
    def test_adds_rolling_columns(self):
        df = _make_df()
        df_out = add_rolling_features(df)
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        for window in ROLLING_WINDOWS:
            for col in sensor_cols:
                assert f"{col}_roll_mean_{window}" in df_out.columns
                assert f"{col}_roll_std_{window}" in df_out.columns

    def test_no_nan_in_rolling(self):
        df = _make_df()
        df_out = add_rolling_features(df)
        roll_cols = [c for c in df_out.columns if "roll_" in c]
        assert df_out[roll_cols].isnull().sum().sum() == 0


class TestDeltaFeatures:
    def test_adds_delta_columns(self):
        df = _make_df()
        df_out = add_delta_features(df)
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        for col in sensor_cols:
            assert f"{col}_delta" in df_out.columns

    def test_first_delta_is_zero(self):
        """Le premier cycle de chaque moteur doit avoir delta=0 (fillna)."""
        df = _make_df(n_units=2, cycles=10)
        df_out = add_delta_features(df)
        first_rows = df_out.groupby("unit_id").head(1)
        delta_cols = [c for c in df_out.columns if c.endswith("_delta")]
        assert (first_rows[delta_cols] == 0).all().all()


class TestNormalizeFeatures:
    def test_fit_mode_zero_mean(self):
        df = _make_df()
        df_out, scaler = normalize_features(df, fit=True)
        exclude = {"unit_id", "cycle", "rul"}
        feat_cols = [c for c in df_out.columns if c not in exclude]
        means = df_out[feat_cols].mean()
        assert (means.abs() < 0.1).all()

    def test_transform_mode_needs_scaler(self):
        df = _make_df()
        with pytest.raises(ValueError):
            normalize_features(df, scaler=None, fit=False)

    def test_scaler_reuse(self):
        df_train = _make_df(n_units=2, cycles=60)
        df_test = _make_df(n_units=1, cycles=20)
        _, scaler = normalize_features(df_train, fit=True)
        df_test_out, _ = normalize_features(df_test, scaler=scaler, fit=False)
        # Pas d'exception → succès
        assert df_test_out is not None


class TestCreateSequences:
    def test_output_shape(self):
        seq_len = 10
        n_samples = 30
        n_features = 5
        data = np.random.rand(n_samples, n_features)
        targets = np.random.rand(n_samples)
        X, y = create_sequences(data, targets, seq_len=seq_len)
        assert X.shape == (n_samples - seq_len + 1, seq_len, n_features)
        assert y.shape == (n_samples - seq_len + 1,)

    def test_sequences_by_unit_no_mixing(self):
        df = _make_df(n_units=3, cycles=60)
        feat_cols = [c for c in df.columns if c.startswith("sensor_")]
        X, y = create_sequences_by_unit(df, feat_cols, seq_len=50)
        assert X.ndim == 3
        assert X.shape[1] == 50
        assert X.shape[2] == len(feat_cols)
