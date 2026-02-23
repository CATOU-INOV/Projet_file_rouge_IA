"""API FastAPI pour la prédiction RUL des turbofans."""
from __future__ import annotations
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest

from src.config import MLFLOW_TRACKING_URI, MODEL_NAME, EXPERIMENT_NAME, FEATURE_DIR
from src.serving.schemas import (
    SensorInput,
    PredictionResponse,
    HealthResponse,
    ModelVersionResponse,
)

logger = logging.getLogger(__name__)

# ── Métriques Prometheus ─────────────────────────────────────────────────────
PREDICTION_COUNT = Counter("prediction_count", "Nombre total de predictions")
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latence des predictions",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
PREDICTION_ERRORS = Counter("prediction_errors", "Nombre d erreurs de prediction")

# ── État global du modèle ────────────────────────────────────────────────────
_model = None
_model_version: Optional[str] = None
_model_name: Optional[str] = None
_feature_df: Optional[pd.DataFrame] = None
_feature_cols: Optional[list] = None


def _load_model_from_mlflow():
    """Charger le meilleur modèle depuis le MLflow Model Registry."""
    global _model, _model_version, _model_name

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    stage = os.getenv("MODEL_STAGE", "Production")
    model_uri = f"models:/{MODEL_NAME}/{stage}"

    try:
        _model = mlflow.pyfunc.load_model(model_uri)
        _model_version = stage
        _model_name = MODEL_NAME
        logger.info(f"Modèle chargé depuis {model_uri}")
    except Exception as e:
        logger.warning(f"Impossible de charger depuis Registry ({e}). Tentative Latest ...")
        try:
            # Fallback : dernier run de l'expérience
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.mae ASC"],
                    max_results=1,
                )
                if runs:
                    run_id = runs[0].info.run_id
                    _model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
                    _model_version = run_id[:8]
                    _model_name = MODEL_NAME
                    logger.info(f"Modèle chargé depuis run {run_id}")
                    return
        except Exception as e2:
            logger.error(f"Fallback MLflow échoué: {e2}")

        logger.warning("Aucun modèle MLflow disponible – mode dégradé.")


def _load_feature_data():
    """Charger les features pré-calculées depuis le Parquet."""
    global _feature_df, _feature_cols
    train_path = FEATURE_DIR / "train_FD001_featured.parquet"
    test_path = FEATURE_DIR / "test_FD001_featured.parquet"
    dfs = []
    for p in [train_path, test_path]:
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if dfs:
        _feature_df = pd.concat(dfs, ignore_index=True)
        exclude = {"unit_id", "cycle", "rul"}
        _feature_cols = [c for c in _feature_df.columns if c not in exclude]
        logger.info(f"Features chargées: {len(_feature_df)} lignes, {len(_feature_cols)} features")
    else:
        logger.warning("Aucun fichier de features trouvé. Lancez: make data")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charger le modèle MLflow et les features au démarrage."""
    _load_model_from_mlflow()
    _load_feature_data()
    yield


app = FastAPI(
    title="Turbofan RUL Prediction API",
    description="API de maintenance prédictive – estimation RUL des turbofans NASA C-MAPSS",
    version="1.0.0",
    lifespan=lifespan,
)

def _input_to_array(data: SensorInput) -> np.ndarray:
    """Récupérer le vecteur de features depuis le Parquet pré-calculé."""
    if _feature_df is None or _feature_cols is None:
        raise HTTPException(
            status_code=503,
            detail="Features non chargées. Lancez: make data",
        )
    mask = (_feature_df["unit_id"] == data.unit_id) & (_feature_df["cycle"] == data.cycle)
    rows = _feature_df[mask]
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"unit_id={data.unit_id}, cycle={data.cycle} introuvable dans les données. "
                   f"Utilisez un unit_id entre 1-100 et un cycle existant.",
        )
    return rows[_feature_cols].values[:1].astype(np.float32)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: SensorInput):
    """Prédire la RUL à partir des données capteurs d'un moteur."""
    if _model is None:
        PREDICTION_ERRORS.inc()
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Vérifiez MLflow et relancez le service.",
        )

    try:
        t0 = time.perf_counter()
        X = _input_to_array(data)
        predicted_rul = float(_model.predict(X)[0])
        predicted_rul = max(0.0, predicted_rul)  # RUL ne peut pas être négative
        latency = time.perf_counter() - t0

        PREDICTION_COUNT.inc()
        PREDICTION_LATENCY.observe(latency)

        return PredictionResponse(
            unit_id=data.unit_id,
            predicted_rul=round(predicted_rul, 2),
            model_version=_model_version,
            model_name=_model_name,
        )
    except HTTPException:
        raise
    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.error(f"Erreur de prédiction: {e}")
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe – vérification de l'état du service."""
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_version=_model_version,
    )


@app.get("/metrics")
async def metrics():
    """Métriques Prometheus au format texte."""
    return PlainTextResponse(generate_latest(), media_type="text/plain")


@app.get("/model/version", response_model=ModelVersionResponse)
async def get_model_version():
    """Retourner les informations sur le modèle actif."""
    return ModelVersionResponse(
        model_name=_model_name or "N/A",
        model_version=_model_version,
        experiment=EXPERIMENT_NAME,
        tracking_uri=MLFLOW_TRACKING_URI,
    )
