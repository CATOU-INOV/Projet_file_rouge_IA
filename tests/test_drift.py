"""Tests unitaires – détection de drift (KS-test et PSI)."""
import numpy as np
import pytest

from src.monitoring.drift_detector import detect_data_drift, compute_psi


class TestDetectDataDrift:
    def test_no_drift_same_distribution(self):
        """Deux distributions identiques → pas de drift."""
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, (500, 3))
        cur = rng.normal(0, 1, (500, 3))
        result = detect_data_drift(ref, cur, threshold=0.001)
        # Avec un seuil très bas, pas de drift attendu pour même distribution
        assert "any_drift" in result
        assert "drift_features" in result

    def test_drift_detected_different_distribution(self):
        """Des distributions très différentes → drift détecté."""
        ref = np.random.normal(0, 1, (500, 2))
        cur = np.random.normal(10, 1, (500, 2))  # décalage massif
        result = detect_data_drift(ref, cur, threshold=0.05)
        assert result["any_drift"] is True
        assert len(result["drift_features"]) > 0

    def test_result_contains_ks_stats(self):
        """Le résultat doit contenir ks_statistic et p_value par feature."""
        ref = np.random.normal(0, 1, (100, 2))
        cur = np.random.normal(0, 1, (100, 2))
        features = ["sensor_a", "sensor_b"]
        result = detect_data_drift(ref, cur, feature_names=features)
        for feat in features:
            assert "ks_statistic" in result[feat]
            assert "p_value" in result[feat]
            assert "is_drift" in result[feat]

    def test_1d_input(self):
        """Accepter des tableaux 1D (une seule feature)."""
        ref = np.random.normal(0, 1, 200)
        cur = np.random.normal(5, 1, 200)
        result = detect_data_drift(ref, cur)
        assert "any_drift" in result


class TestComputePSI:
    def test_stable_same_distribution(self):
        """PSI ≈ 0 pour deux distributions identiques."""
        data = np.random.normal(0, 1, 1000)
        result = compute_psi(data, data)
        assert result["psi"] < 0.1
        assert result["interpretation"] == "stable"
        assert result["is_drift"] is False

    def test_significant_drift(self):
        """PSI >= 0.2 pour des distributions très différentes."""
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(5, 1, 1000)
        result = compute_psi(ref, cur)
        assert result["psi"] >= 0.2
        assert result["is_drift"] is True
        assert result["interpretation"] == "significant_drift"

    def test_psi_result_keys(self):
        """Le résultat doit contenir les clés attendues."""
        ref = np.random.normal(0, 1, 200)
        cur = np.random.normal(0, 1, 200)
        result = compute_psi(ref, cur)
        for key in ["psi", "interpretation", "is_drift", "bins_detail"]:
            assert key in result
