"""Schemas Pydantic. TODO: Completez SensorInput et PredictionResponse."""
from pydantic import BaseModel, Field

class SensorInput(BaseModel):
    """TODO: unit_id (int>0), cycle (int>0), setting_1..3 (float), sensor_2,3,4,7,8,9,11,12,13,14,15,17,20,21 (float)"""
    pass

class PredictionResponse(BaseModel):
    """TODO: unit_id, predicted_rul, model_version"""
    pass

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
