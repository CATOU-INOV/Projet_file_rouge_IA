"""Factory Pattern pour instancier les modèles par leur nom."""
from src.config import RANDOM_SEED

MODELS_REGISTRY = {}


def register_model(name):
    def decorator(func):
        MODELS_REGISTRY[name] = func
        return func
    return decorator


@register_model("random_forest")
def build_random_forest(**kwargs):
    """RandomForestRegressor avec hyperparamètres optimisés."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


@register_model("xgboost")
def build_xgboost(**kwargs):
    """XGBRegressor avec early stopping."""
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )


@register_model("lightgbm")
def build_lightgbm(**kwargs):
    """LGBMRegressor."""
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )


def get_model(name: str, **kwargs):
    if name not in MODELS_REGISTRY:
        raise ValueError(
            f"Modèle inconnu '{name}'. Disponibles : {', '.join(MODELS_REGISTRY)}"
        )
    return MODELS_REGISTRY[name](**kwargs)


def list_models() -> list:
    return list(MODELS_REGISTRY.keys())
