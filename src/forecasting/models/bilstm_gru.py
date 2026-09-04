"""BiLSTM-GRU Recurrent Forecaster Adapter."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.forecasting.base import ForecastModel, ModelMetadata


class _SeparateHeads(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.orbit = nn.Sequential(nn.Linear(input_dim, 48), nn.SiLU(), nn.Linear(48, 3))
        self.clock = nn.Sequential(nn.Linear(input_dim, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.orbit(features), self.clock(features)), dim=-1)


class BiLSTMGRUNet(nn.Module):
    def __init__(self, history_dim: int = 7, query_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(history_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.heads = _SeparateHeads(hidden_dim + query_dim)

    def forward(self, history: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(history)
        _, hidden = self.gru(encoded)
        features = torch.cat((hidden[-1], query), dim=-1)
        return self.heads(features)


class BiLSTMGRUModel(ForecastModel):
    """Bidirectional LSTM + GRU recurrent forecaster with separate orbit and clock heads."""

    def __init__(
        self,
        name: str = "BiLSTM-GRU",
        seq_len: int = 16,
        hidden_dim: int = 32,
        epochs: int = 50,
        lr: float = 0.001,
        version: str = "1.0.0",
        **kwargs,
    ):
        super().__init__(name=name, model_type="bilstm_gru", version=version)
        self.orbit_class = kwargs.get("orbit_class", "MEO")
        self.satellite_id = kwargs.get("satellite_id", "")
        self.cadence_minutes = float(kwargs.get("cadence_minutes", 15.0))
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.target_cols = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]
        self.network: Optional[BiLSTMGRUNet] = None
        self.origin: Optional[pd.Timestamp] = None
        self.center: np.ndarray = np.zeros(4, dtype=np.float64)
        self.scale: np.ndarray = np.ones(4, dtype=np.float64)

    def _make_features(self, history_sub: pd.DataFrame, query_time: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray]:
        vals = (history_sub[self.target_cols].to_numpy(dtype=float) - self.center) / self.scale
        times = pd.to_datetime(history_sub["utc_time"])
        ages = (times - query_time).dt.total_seconds().to_numpy(dtype=float).reshape(-1, 1) / (2.0 * 86400.0)
        phase = 2.0 * np.pi * (times.dt.hour * 3600.0 + times.dt.minute * 60.0).to_numpy(dtype=float) / 86400.0
        h_tensor = np.column_stack((vals, ages, np.sin(phase), np.cos(phase))).astype(np.float32)

        if len(h_tensor) < self.seq_len:
            pad = np.repeat(h_tensor[[0]], self.seq_len - len(h_tensor), axis=0)
            h_tensor = np.vstack((pad, h_tensor))
        else:
            h_tensor = h_tensor[-self.seq_len:]

        # Query features
        q_phase = 2.0 * np.pi * (query_time.hour * 3600.0 + query_time.minute * 60.0) / 86400.0
        lead_days = (query_time - history_sub["utc_time"].iloc[-1]).total_seconds() / 86400.0
        elapsed_days = (query_time - self.origin).total_seconds() / 86400.0
        q_feats = [lead_days / 2.0, elapsed_days / 7.0]
        for h in range(1, 5):
            q_feats.extend([math.sin(h * q_phase), math.cos(h * q_phase)])
        q_tensor = np.asarray(q_feats, dtype=np.float32)
        return h_tensor, q_tensor

    def fit(self, train_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> "BiLSTMGRUModel":
        clean = train_df.dropna(subset=["utc_time", *self.target_cols]).sort_values("utc_time").reset_index(drop=True)
        self.origin = clean["utc_time"].iloc[0].floor("D")

        vals = clean[self.target_cols].to_numpy(dtype=np.float64)
        self.center = np.median(vals, axis=0)
        q25, q75 = np.percentile(vals, [25, 75], axis=0)
        self.scale = np.maximum((q75 - q25) / 1.349, 0.01)

        hists, queries, targets = [], [], []
        n_rows = len(clean)
        for i in range(4, n_rows):
            query_time = clean["utc_time"].iloc[i]
            h_t, q_t = self._make_features(clean.iloc[:i], query_time)
            target = (vals[i] - self.center) / self.scale
            hists.append(h_t)
            queries.append(q_t)
            targets.append(target)

        if not hists:
            raise ValueError("Insufficient history rows for BiLSTMGRUModel")

        h_arr = np.asarray(hists, dtype=np.float32)
        q_arr = np.asarray(queries, dtype=np.float32)
        y_arr = np.asarray(targets, dtype=np.float32)

        self.network = BiLSTMGRUNet(
            history_dim=h_arr.shape[-1],
            query_dim=q_arr.shape[-1],
            hidden_dim=self.hidden_dim,
        )

        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.lr, weight_decay=1e-4)
        huber = nn.SmoothL1Loss()

        h_tensor = torch.as_tensor(h_arr)
        q_tensor = torch.as_tensor(q_arr)
        y_tensor = torch.as_tensor(y_arr)

        self.network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.network(h_tensor, q_tensor)
            loss = huber(pred, y_tensor)
            loss.backward()
            optimizer.step()

        self.network.eval()
        self.is_fitted = True
        return self

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon_epochs: Union[int, pd.DatetimeIndex, Sequence[pd.Timestamp]],
    ) -> np.ndarray:
        if not self.is_fitted or self.network is None:
            raise ValueError("BiLSTMGRUModel must be fitted before predict()")

        clean_hist = history_df.dropna(subset=["utc_time", *self.target_cols]).sort_values("utc_time").reset_index(drop=True)
        if isinstance(horizon_epochs, int):
            last_time = pd.to_datetime(clean_hist["utc_time"].iloc[-1])
            step_interval = pd.Timedelta(minutes=15)
            forecast_times = pd.date_range(start=last_time + step_interval, periods=horizon_epochs, freq=step_interval)
        else:
            forecast_times = pd.DatetimeIndex(horizon_epochs)

        preds = []
        self.network.eval()
        with torch.no_grad():
            for q_time in forecast_times:
                h_t, q_t = self._make_features(clean_hist, q_time)
                h_tensor = torch.as_tensor(h_t[None, ...], dtype=torch.float32)
                q_tensor = torch.as_tensor(q_t[None, ...], dtype=torch.float32)
                scaled_p = self.network(h_tensor, q_tensor).cpu().numpy()[0]
                p = scaled_p * self.scale + self.center
                preds.append(p)

        return np.asarray(preds, dtype=np.float64)

    def save(self, path: Union[str, Path]) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "model_type": self.model_type,
            "version": self.version,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "origin": self.origin.isoformat() if self.origin else None,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "network_state_dict": self.network.state_dict() if self.network else None,
            "target_cols": self.target_cols,
        }
        torch.save(payload, out_path)
        return out_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BiLSTMGRUModel":
        p = Path(path)
        payload = torch.load(p, map_location="cpu", weights_only=False)
        model = cls(
            name=payload.get("name", "BiLSTM-GRU"),
            seq_len=payload.get("seq_len", 16),
            hidden_dim=payload.get("hidden_dim", 32),
            version=payload.get("version", "1.0.0"),
        )
        model.center = np.asarray(payload["center"], dtype=np.float64)
        model.scale = np.asarray(payload["scale"], dtype=np.float64)
        if payload.get("origin"):
            model.origin = pd.Timestamp(payload["origin"])

        if payload.get("network_state_dict"):
            model.network = BiLSTMGRUNet(
                history_dim=7,
                query_dim=10,
                hidden_dim=model.hidden_dim,
            )
            model.network.load_state_dict(payload["network_state_dict"])
            model.network.eval()

        model.is_fitted = True
        return model

    def get_metadata(self) -> ModelMetadata:
        n_params = sum(p.numel() for p in self.network.parameters()) if self.network else 15000
        return ModelMetadata(
            name=self.name,
            model_type=self.model_type,
            version=self.version,
            architecture="Bidirectional LSTM + Gated Recurrent Unit with Dual Orbit/Clock Heads",
            parameters={"seq_len": self.seq_len, "hidden_dim": self.hidden_dim},
            lookback_steps=self.seq_len,
            forecast_horizon=96,
            features=["scaled_error", "age_hours", "sin_phase", "cos_phase", "query_harmonics"],
            target_representation="ECEF",
            supports_uncertainty=False,
            trainable=True,
            description="Recurrent architecture learning temporal sequence dynamics across lookback windows.",
            parameter_count=n_params,
        )
