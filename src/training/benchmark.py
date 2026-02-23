"""Benchmark complet : entraîner les 5 modèles et comparer les métriques."""
import traceback
from datetime import datetime

import pandas as pd

from src.config import FEATURE_DIR
from src.training.train import train

BENCHMARK_MODELS = ["random_forest", "xgboost", "lightgbm", "lstm", "cnn1d"]


def run_benchmark(models: list = None) -> pd.DataFrame:
    """Boucler sur les modèles, collecter les métriques, sauvegarder un CSV comparatif."""
    # Import des modèles DL pour les enregistrer dans la factory
    from src.models import deep_learning  # noqa: F401

    if models is None:
        models = BENCHMARK_MODELS

    results = []
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Entraînement : {model_name}")
        print(f"{'='*50}")
        try:
            result = train(model_name)
            results.append(result)
        except Exception as e:
            print(f"[ERREUR] {model_name}: {e}")
            traceback.print_exc()
            results.append({"model_name": model_name, "error": str(e)})

    df_results = pd.DataFrame(results)

    # Tri par MAE croissant
    if "mae" in df_results.columns:
        df_results = df_results.sort_values("mae")

    # Sauvegarde CSV
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = FEATURE_DIR / f"benchmark_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nBenchmark sauvegardé: {csv_path}")
    print(df_results.to_string())

    return df_results


if __name__ == "__main__":
    run_benchmark()
