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
    """Entraîner un modèle Deep Learning (LSTM/CNN1D) avec MLflow tracking (PyTorch)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    import mlflow.pytorch
    from src.models import deep_learning  # noqa: F401 – enregistrement factory

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

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True
    )

    with mlflow.start_run(run_name=model_name) as run:
        model = get_model(model_name, n_features=n_features, seq_len=SEQUENCE_LENGTH)

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow.log_param("n_train_sequences", len(X_train))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-6
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience, patience_counter = 10, 0
        best_state = None
        max_seconds = 15 * 60
        start_time = time.time()
        epochs_trained = 0

        t0 = time.time()
        for epoch in range(100):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()

            scheduler.step(val_loss)
            epochs_trained = epoch + 1
            print(f"Epoch {epoch + 1}/100 — val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping à l'epoch {epoch + 1}")
                break
            if time.time() - start_time > max_seconds:
                print(f"Temps max ({max_seconds}s) atteint")
                break

        train_time = time.time() - t0

        if best_state:
            model.load_state_dict(best_state)

        # Métriques
        model.eval()
        t_pred = time.time()
        with torch.no_grad():
            y_pred = model(X_val_t).numpy()
        latency_ms = (time.time() - t_pred) / max(len(X_val), 1) * 1000

        metrics = compute_metrics(y_val, y_pred)
        metrics["train_time_s"] = round(train_time, 2)
        metrics["latency_ms_per_sample"] = round(latency_ms, 4)
        metrics["epochs_trained"] = epochs_trained

        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(
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
