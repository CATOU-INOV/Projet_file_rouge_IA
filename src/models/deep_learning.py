"""Modeles DL. TODO: Completez LSTM et CNN1D."""
from src.config import SEQUENCE_LENGTH, RANDOM_SEED
from src.models.factory import register_model

@register_model("lstm")
def build_lstm(n_features=14, seq_len=SEQUENCE_LENGTH, **kw):
    """TODO: LSTM(64) -> Dropout -> LSTM(32) -> Dropout -> Dense(16,relu) -> Dense(1). Compile mse."""
    raise NotImplementedError("TODO")

@register_model("cnn1d")
def build_cnn1d(n_features=14, seq_len=SEQUENCE_LENGTH, **kw):
    """TODO: Conv1D(64,3) -> BN -> Pool -> Conv1D(32,3) -> BN -> GAP -> Dense(32,relu) -> Dense(1)."""
    raise NotImplementedError("TODO")

def get_dl_callbacks(patience=5, max_minutes=15):
    """TODO: EarlyStopping, ReduceLROnPlateau, MaxTimeCallback."""
    raise NotImplementedError("TODO")
