"""Modèles Deep Learning : LSTM et CNN1D pour la prédiction RUL (PyTorch)."""
import torch
import torch.nn as nn

from src.config import SEQUENCE_LENGTH, RANDOM_SEED
from src.models.factory import register_model


class _LSTMModel(nn.Module):
    """LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16, relu) -> Dense(1)."""

    def __init__(self, n_features: int):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])  # dernier timestep
        x = self.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class _CNN1DModel(nn.Module):
    """Conv1D(64,3) -> BN -> MaxPool -> Conv1D(32,3) -> BN -> GAP -> Dense(32, relu) -> Dense(1)."""

    def __init__(self, n_features: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features) → (batch, n_features, seq_len) pour Conv1d
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.mean(dim=-1)  # GlobalAveragePooling
        x = self.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


@register_model("lstm")
def build_lstm(n_features: int = 14, seq_len: int = SEQUENCE_LENGTH, **kw):
    torch.manual_seed(RANDOM_SEED)
    return _LSTMModel(n_features)


@register_model("cnn1d")
def build_cnn1d(n_features: int = 14, seq_len: int = SEQUENCE_LENGTH, **kw):
    torch.manual_seed(RANDOM_SEED)
    return _CNN1DModel(n_features)
