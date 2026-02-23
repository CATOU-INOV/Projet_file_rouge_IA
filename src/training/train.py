"""Pipeline d'entraînement avec tracking MLflow.
Usage: python -m src.training.train --model xgboost
"""
import argparse
import time

import mlflow
import mlflow.sklearn
import numpy as np

from src.config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    MODEL_NAME,
    SEQUENCE_LENGTH,
)
from src.models.factory import get_model, list_models
from src.models.baseline import load_featured_data, prepare_tabular_data, compute_metrics
from src.data.features import create_sequences_by_unit

DL_MODELS = {"lstm", "cnn1d"}


def _setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def train_tabular(model_name: str, **kw) -> dict:
    """Entraîner un modèle tabulaire (RF/XGBoost/LightGBM) avec MLflow tracking."""
    _setup_mlflow()

    df = load_featured_data()
    X_train, X_val, y_train, y_val, feature_cols = prepare_tabular_data(df)

    with mlflow.start_run(run_name=model_name) as run:
        model = get_model(model_name)

        # Log des hyperparamètres
        params = model.get_params() if hasattr(model, "get_params") else {}
        mlflow.log_params({k: str(v) for k, v in params.items() if v is not None})
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_train", len(X_train))

        # Entraînement
        t0 = time.time()
        if model_name == "xgboost":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        elif model_name == "lightgbm":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Métriques
        t_pred = time.time()
        y_pred = model.predict(X_val)
        latency_ms = (time.time() - t_pred) / max(len(X_val), 1) * 1000

        metrics = compute_metrics(y_val, y_pred)
        metrics["train_time_s"] = round(train_time, 2)
        metrics["latency_ms_per_sample"] = round(latency_ms, 4)

        mlflow.log_metrics(metrics)

        # Log du modèle dans le Model Registry
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        run_id = run.info.run_id
        print(f"[{model_name}] run_id={run_id} | {metrics}")
        return {"model_name": model_name, "run_id": run_id, **metrics}


def train_deep_learning(model_name: str, **kw) -> dict:
    """Entraîner un modèle Deep Learning (LSTM/CNN1D) avec MLflow tracking."""
    import mlflow.tensorflow
    # Import pour déclencher l'enregistrement dans la factory
    from src.models import deep_learning  # noqa: F401
    from src.models.deep_learning import get_dl_callbacks

    _setup_mlflow()

    df = load_featured_data()
    exclude = {"unit_id", "cycle", "rul"}
    feature_cols = [c for c in df.columns if c not in exclude]
    n_features = len(feature_cols)

    X_seq, y_seq = create_sequences_by_unit(df, feature_cols, seq_len=SEQUENCE_LENGTH)

    # Split temporel simple (80/20)
    split = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    with mlflow.start_run(run_name=model_name) as run:
        model = get_model(model_name, n_features=n_features, seq_len=SEQUENCE_LENGTH)

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow.log_param("n_train_sequences", len(X_train))

        callbacks = get_dl_callbacks(patience=10, max_minutes=15)

        t0 = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            verbose=1,
        )
        train_time = time.time() - t0

        # Métriques
        t_pred = time.time()
        y_pred = model.predict(X_val, verbose=0).flatten()
        latency_ms = (time.time() - t_pred) / max(len(X_val), 1) * 1000

        metrics = compute_metrics(y_val, y_pred)
        metrics["train_time_s"] = round(train_time, 2)
        metrics["latency_ms_per_sample"] = round(latency_ms, 4)
        metrics["epochs_trained"] = len(history.history["loss"])

        mlflow.log_metrics(metrics)

        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        run_id = run.info.run_id
        print(f"[{model_name}] run_id={run_id} | {metrics}")
        return {"model_name": model_name, "run_id": run_id, **metrics}


def train(model_name: str, **kw) -> dict:
    """Router vers l'entraînement tabulaire ou DL selon le modèle."""
    if model_name in DL_MODELS:
        return train_deep_learning(model_name, **kw)
    return train_tabular(model_name, **kw)


def main():
    # Import des modèles DL pour les enregistrer dans la factory
    from src.models import deep_learning  # noqa: F401

    parser = argparse.ArgumentParser(description="Entraîner un modèle RUL")
    parser.add_argument("--model", default="xgboost", choices=list_models())
    args = parser.parse_args()
    result = train(args.model)
    print(f"\nRésultat: {result}")


if __name__ == "__main__":
    main()
