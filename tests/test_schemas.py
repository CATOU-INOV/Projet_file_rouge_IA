"""Tests unitaires – validation des schemas Pydantic."""
import pytest
from pydantic import ValidationError

from src.serving.schemas import SensorInput, PredictionResponse, HealthResponse


VALID_INPUT = {
    "unit_id": 1, "cycle": 50,
    "setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0,
    "sensor_2": 641.82, "sensor_3": 1589.70, "sensor_4": 1400.60,
    "sensor_7": 14.62, "sensor_8": 21.61, "sensor_9": 554.36,
    "sensor_11": 2388.02, "sensor_12": 9046.19, "sensor_13": 1.30,
    "sensor_14": 47.47, "sensor_15": 521.66, "sensor_17": 2388.02,
    "sensor_20": 39.06, "sensor_21": 23.419,
}


class TestSensorInputSchema:
    def test_valid_input_accepted(self):
        sensor = SensorInput(**VALID_INPUT)
        assert sensor.unit_id == 1
        assert sensor.cycle == 50

    def test_unit_id_must_be_positive(self):
        with pytest.raises(ValidationError):
            SensorInput(**{**VALID_INPUT, "unit_id": 0})

    def test_cycle_must_be_positive(self):
        with pytest.raises(ValidationError):
            SensorInput(**{**VALID_INPUT, "cycle": -5})

    def test_missing_sensor_raises(self):
        incomplete = {k: v for k, v in VALID_INPUT.items() if k != "sensor_2"}
        with pytest.raises(ValidationError):
            SensorInput(**incomplete)

    def test_missing_cycle_raises(self):
        incomplete = {k: v for k, v in VALID_INPUT.items() if k != "cycle"}
        with pytest.raises(ValidationError):
            SensorInput(**incomplete)


class TestPredictionResponseSchema:
    def test_valid_response(self):
        resp = PredictionResponse(
            unit_id=1,
            predicted_rul=42.5,
            model_version="1",
            model_name="turbofan_rul_predictor",
        )
        assert resp.predicted_rul == 42.5

    def test_optional_fields(self):
        """model_version et model_name sont optionnels."""
        resp = PredictionResponse(unit_id=1, predicted_rul=10.0)
        assert resp.model_version is None
        assert resp.model_name is None


class TestHealthResponseSchema:
    def test_valid_health(self):
        health = HealthResponse(status="ok", model_loaded=True, model_version="1")
        assert health.status == "ok"

    def test_degraded_health(self):
        health = HealthResponse(status="degraded", model_loaded=False)
        assert health.model_loaded is False
        assert health.model_version is None
