"""Tests unitaires – API FastAPI."""
import pytest
from fastapi.testclient import TestClient

from src.serving.app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "unit_id": 1,
    "cycle": 50,
    "setting_1": -0.0007,
    "setting_2": -0.0004,
    "setting_3": 100.0,
    "sensor_2": 641.82,
    "sensor_3": 1589.70,
    "sensor_4": 1400.60,
    "sensor_7": 14.62,
    "sensor_8": 21.61,
    "sensor_9": 554.36,
    "sensor_11": 2388.02,
    "sensor_12": 9046.19,
    "sensor_13": 1.30,
    "sensor_14": 47.47,
    "sensor_15": 521.66,
    "sensor_17": 2388.02,
    "sensor_20": 39.06,
    "sensor_21": 23.419,
}


class TestHealthEndpoint:
    def test_health_returns_200(self):
        """L'endpoint /health doit retourner HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self):
        """La réponse /health doit avoir les champs requis."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] in ("ok", "degraded")


class TestPredictEndpoint:
    def test_predict_invalid_input_returns_422(self):
        """Un payload invalide (unit_id manquant) doit retourner 422."""
        response = client.post("/predict", json={"cycle": 50})
        assert response.status_code == 422

    def test_predict_negative_unit_id_returns_422(self):
        """unit_id <= 0 doit être rejeté (validation Pydantic)."""
        payload = {**VALID_PAYLOAD, "unit_id": -1}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_negative_cycle_returns_422(self):
        """cycle <= 0 doit être rejeté."""
        payload = {**VALID_PAYLOAD, "cycle": 0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_valid_input_no_model(self):
        """Sans modèle chargé, /predict doit retourner 503."""
        response = client.post("/predict", json=VALID_PAYLOAD)
        # Sans modèle MLflow disponible → 503 Service Unavailable
        assert response.status_code == 503


class TestModelVersionEndpoint:
    def test_model_version_returns_200(self):
        """/model/version doit retourner HTTP 200."""
        response = client.get("/model/version")
        assert response.status_code == 200

    def test_model_version_schema(self):
        """La réponse /model/version doit avoir les champs requis."""
        response = client.get("/model/version")
        data = response.json()
        assert "model_name" in data
        assert "experiment" in data
        assert "tracking_uri" in data


class TestMetricsEndpoint:
    def test_metrics_returns_200(self):
        """/metrics doit retourner HTTP 200 avec du contenu Prometheus."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "prediction_count" in response.text
