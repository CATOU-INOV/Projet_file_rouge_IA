"""Benchmark complet. TODO: Completez."""
import pandas as pd
from src.config import FEATURE_DIR
from src.training.train import train

BENCHMARK_MODELS = ["random_forest", "xgboost", "lightgbm", "lstm", "cnn1d"]

def run_benchmark():
    """TODO: Boucler sur les modeles, collecter metriques, sauver CSV."""
    raise NotImplementedError("TODO")

if __name__ == "__main__":
    run_benchmark()
