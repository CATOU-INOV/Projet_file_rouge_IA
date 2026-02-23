"""Tests unitaires – factory de modèles et métriques."""
import numpy as np
import pytest


class TestModelFactory:
    def test_get_random_forest(self):
        from src.models.factory import get_model
        model = get_model("random_forest")
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("xgboost"),
        reason="xgboost non installé"
    )
    def test_get_xgboost(self):
        from src.models.factory import get_model
        model = get_model("xgboost")
        assert model is not None

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("lightgbm"),
        reason="lightgbm non installé"
    )
    def test_get_lightgbm(self):
        from src.models.factory import get_model
        model = get_model("lightgbm")
        assert model is not None

    def test_unknown_model_raises(self):
        from src.models.factory import get_model
        with pytest.raises(ValueError, match="inconnu"):
            get_model("unknown_model_xyz")

    def test_list_models_contains_all(self):
        from src.models.factory import list_models
        # Import des DL pour les enregistrer
        from src.models import deep_learning  # noqa
        models = list_models()
        for name in ["random_forest", "xgboost", "lightgbm", "lstm", "cnn1d"]:
            assert name in models


class TestComputeMetrics:
    def test_perfect_prediction(self):
        from src.models.baseline import compute_metrics
        y = np.array([10.0, 20.0, 30.0])
        metrics = compute_metrics(y, y)
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0

    def test_metrics_keys(self):
        from src.models.baseline import compute_metrics
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.0, 31.0])
        metrics = compute_metrics(y_true, y_pred)
        for key in ["mae", "rmse", "r2", "smape"]:
            assert key in metrics

    def test_smape_range(self):
        from src.models.baseline import compute_metrics
        y_true = np.array([100.0, 50.0, 25.0])
        y_pred = np.array([80.0, 60.0, 30.0])
        metrics = compute_metrics(y_true, y_pred)
        assert 0.0 <= metrics["smape"] <= 200.0
