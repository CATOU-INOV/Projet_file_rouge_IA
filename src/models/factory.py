"""Factory Pattern. TODO: Completez les fonctions de construction."""
from src.config import RANDOM_SEED
MODELS_REGISTRY = {}

def register_model(name):
    def decorator(func):
        MODELS_REGISTRY[name] = func
        return func
    return decorator

@register_model("random_forest")
def build_random_forest(**kwargs):
    """TODO: RandomForestRegressor(n_estimators=500, max_depth=12, n_jobs=-1)"""
    raise NotImplementedError("TODO")

@register_model("xgboost")
def build_xgboost(**kwargs):
    """TODO: XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, early_stopping_rounds=50)"""
    raise NotImplementedError("TODO")

@register_model("lightgbm")
def build_lightgbm(**kwargs):
    """TODO: LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8)"""
    raise NotImplementedError("TODO")

def get_model(name, **kwargs):
    if name not in MODELS_REGISTRY:
        raise ValueError(f"Modele inconnu '{name}'. Disponibles : {', '.join(MODELS_REGISTRY)}")
    return MODELS_REGISTRY[name](**kwargs)

def list_models():
    return list(MODELS_REGISTRY.keys())
