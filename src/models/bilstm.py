import torch
import torch.nn as nn
from src.config import SEQ_LEN, FORECAST_HORIZON, TARGET_COLS_4

class BiLSTMGRUPyTorchModel(nn.Module):

    def __init__(self, seq_len: int=SEQ_LEN, n_features: int=len(TARGET_COLS_4), output_dim: int | None=None, target_feature_indices: tuple[int, ...] | None=None, forecast_horizon: int=FORECAST_HORIZON, bilstm_units: int=64, gru_units: int=64, dropout_1: float=0.2, dropout_2: float=0.1, separate_orbit_clock_heads: bool=False):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.output_dim = output_dim if output_dim is not None else n_features
        self.target_feature_indices = tuple(range(self.output_dim)) if target_feature_indices is None else tuple(target_feature_indices)
        if len(self.target_feature_indices) != self.output_dim:
            raise ValueError('target_feature_indices must have one index per output')
        self.forecast_horizon = forecast_horizon
        self.separate_orbit_clock_heads = bool(separate_orbit_clock_heads and self.output_dim == 4)
        self.bilstm = nn.LSTM(input_size=n_features, hidden_size=bilstm_units, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_1)
        self.gru = nn.GRU(input_size=bilstm_units * 2, hidden_size=gru_units, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_2)
        self.attn_pool = nn.Sequential(nn.Linear(gru_units, 32), nn.Tanh(), nn.Linear(32, 1))
        self.layer_norm = nn.LayerNorm(gru_units * 2)

        def projection(outputs: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(gru_units * 2, 128), nn.GELU(), nn.Dropout(dropout_2), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, forecast_horizon * outputs))
        if self.separate_orbit_clock_heads:
            self.orbit_proj = projection(3)
            self.clock_proj = projection(1)
            self.dense_proj = None
        else:
            self.orbit_proj = None
            self.clock_proj = None
            self.dense_proj = projection(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout1(lstm_out)
        gru_out, _ = self.gru(lstm_out)
        gru_out = self.dropout2(gru_out)
        last_state = gru_out[:, -1, :]
        attn_weights = torch.softmax(self.attn_pool(gru_out), dim=1)
        attn_context = torch.sum(attn_weights * gru_out, dim=1)
        combined = torch.cat([last_state, attn_context], dim=-1)
        combined = self.layer_norm(combined)
        if self.separate_orbit_clock_heads:
            orbit_delta = self.orbit_proj(combined).view(-1, self.forecast_horizon, 3)
            clock_delta = self.clock_proj(combined).view(-1, self.forecast_horizon, 1)
            delta = torch.cat((orbit_delta, clock_delta), dim=-1)
        else:
            delta = self.dense_proj(combined)
            delta = delta.view(-1, self.forecast_horizon, self.output_dim)
        last_obs = x[:, -1:, list(self.target_feature_indices)]
        out = last_obs + delta
        return out

# Backward-compatible alias
GNSSBiLSTMGRU = BiLSTMGRUPyTorchModel
