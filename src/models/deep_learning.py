"""Modèles Deep Learning : LSTM et CNN1D pour la prédiction RUL."""
import time

from src.config import SEQUENCE_LENGTH, RANDOM_SEED
from src.models.factory import register_model


@register_model("lstm")
def build_lstm(n_features: int = 14, seq_len: int = SEQUENCE_LENGTH, **kw):
    """LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16, relu) -> Dense(1)."""
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


@register_model("cnn1d")
def build_cnn1d(n_features: int = 14, seq_len: int = SEQUENCE_LENGTH, **kw):
    """Conv1D(64,3) -> BN -> MaxPool -> Conv1D(32,3) -> BN -> GAP -> Dense(32, relu) -> Dense(1)."""
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


class _MaxTimeCallback:
    """Callback Keras maison pour limiter la durée d'entraînement."""

    def __init__(self, max_minutes: float = 15):
        self.max_seconds = max_minutes * 60
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start_time > self.max_seconds:
            self.model.stop_training = True
            print(f"\nTemps max ({self.max_seconds}s) atteint à l'epoch {epoch}.")


def get_dl_callbacks(patience: int = 10, max_minutes: float = 15) -> list:
    """EarlyStopping, ReduceLROnPlateau, MaxTimeCallback."""
    import tensorflow as tf

    class MaxTimeCallback(tf.keras.callbacks.Callback):
        def __init__(self, max_min):
            super().__init__()
            self.max_seconds = max_min * 60
            self.start_time = None

        def on_train_begin(self, logs=None):
            self.start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            if time.time() - self.start_time > self.max_seconds:
                self.model.stop_training = True
                print(f"\nTemps max ({self.max_seconds}s) atteint à l'epoch {epoch}.")

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        MaxTimeCallback(max_minutes),
    ]
