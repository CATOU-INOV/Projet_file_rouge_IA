"""Schemas Pydantic pour l'API FastAPI."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class SensorInput(BaseModel):
    """Données capteurs d'un moteur pour la prédiction RUL."""

    unit_id: int = Field(..., gt=0, description="Identifiant du moteur (>0)")
    cycle: int = Field(..., gt=0, description="Cycle actuel (>0)")

    # Settings opérationnels
    setting_1: float = Field(..., description="Setting opérationnel 1")
    setting_2: float = Field(..., description="Setting opérationnel 2")
    setting_3: float = Field(..., description="Setting opérationnel 3")

    # Capteurs sélectionnés (non constants sur FD001 : 2,3,4,7,8,9,11,12,13,14,15,17,20,21)
    sensor_2: float = Field(..., description="Température totale à l'entrée LPC (°R)")
    sensor_3: float = Field(..., description="Température totale à l'entrée HPC (°R)")
    sensor_4: float = Field(..., description="Température totale à la sortie LPT (°R)")
    sensor_7: float = Field(..., description="Pression totale à l'entrée HPC (psia)")
    sensor_8: float = Field(..., description="Pression physique à la sortie du fan (psia)")
    sensor_9: float = Field(..., description="Pression physique à l'entrée de bypass (psia)")
    sensor_11: float = Field(..., description="Vitesse physique du fan (rpm)")
    sensor_12: float = Field(..., description="Vitesse physique du cœur (rpm)")
    sensor_13: float = Field(..., description="Vitesse physique corrigée du fan (rpm)")
    sensor_14: float = Field(..., description="Vitesse physique corrigée du cœur (rpm)")
    sensor_15: float = Field(..., description="Ratio de débit BPR")
    sensor_17: float = Field(..., description="Bleed Enthalpy")
    sensor_20: float = Field(..., description="HPT coolant bleed")
    sensor_21: float = Field(..., description="LPT coolant bleed")

    class Config:
        json_schema_extra = {
            "example": {
                "unit_id": 1, "cycle": 50,
                "setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0,
                "sensor_2": 641.82, "sensor_3": 1589.70, "sensor_4": 1400.60,
                "sensor_7": 14.62, "sensor_8": 21.61, "sensor_9": 554.36,
                "sensor_11": 2388.02, "sensor_12": 9046.19, "sensor_13": 1.30,
                "sensor_14": 47.47, "sensor_15": 521.66, "sensor_17": 2388.02,
                "sensor_20": 39.06, "sensor_21": 23.4190,
            }
        }


class PredictionResponse(BaseModel):
    """Réponse de l'API de prédiction."""

    unit_id: int
    predicted_rul: float = Field(..., description="Durée de vie restante prédite (cycles)")
    model_version: Optional[str] = Field(None, description="Version du modèle utilisé")
    model_name: Optional[str] = Field(None, description="Nom du modèle")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class ModelVersionResponse(BaseModel):
    model_name: str
    model_version: Optional[str]
    experiment: str
    tracking_uri: str
