"""API FastAPI. TODO: Completez les endpoints."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest

PREDICTION_COUNT = Counter("prediction_count", "Nombre total de predictions")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latence des predictions")
PREDICTION_ERRORS = Counter("prediction_errors", "Nombre d erreurs")

model = None
model_version = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """TODO: Charger le modele MLflow au demarrage."""
    yield

app = FastAPI(title="Turbofan RUL Prediction API", version="1.0.0", lifespan=lifespan)

@app.post("/predict")
async def predict():
    """TODO: Recevoir SensorInput, predire RUL, retourner PredictionResponse."""
    raise HTTPException(501, "TODO")

@app.get("/health")
async def health():
    """TODO: Retourner HealthResponse."""
    raise HTTPException(501, "TODO")

@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type="text/plain")

@app.get("/model/version")
async def get_model_version():
    """TODO: Retourner la version du modele."""
    raise HTTPException(501, "TODO")
