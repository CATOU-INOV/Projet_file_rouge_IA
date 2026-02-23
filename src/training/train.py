"""Pipeline entrainement MLflow. TODO: Completez. Usage: python -m src.training.train --model xgboost"""
import argparse, time, mlflow, numpy as np
from src.config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MODEL_NAME, SEQUENCE_LENGTH
from src.models.factory import get_model, list_models
from src.models.baseline import load_featured_data, prepare_tabular_data, compute_metrics
from src.data.features import create_sequences_by_unit

DL_MODELS = {"lstm", "cnn1d"}

def train_tabular(model_name, **kw):
    """TODO: Configurer MLflow, charger donnees, entrainer, log_param/metrics/model."""
    raise NotImplementedError("TODO")

def train_deep_learning(model_name, **kw):
    """TODO: Creer sequences, entrainer DL avec callbacks, log MLflow."""
    raise NotImplementedError("TODO")

def train(model_name, **kw):
    return train_deep_learning(model_name, **kw) if model_name in DL_MODELS else train_tabular(model_name, **kw)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost", choices=list_models())
    args = parser.parse_args()
    train(args.model)

if __name__ == "__main__":
    main()
