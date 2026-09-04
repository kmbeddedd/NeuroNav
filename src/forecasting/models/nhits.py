"""N-HiTS (Neural Hierarchical Interpolation for Time Series) Forecaster (P6 Architecture).

Provides multi-rate hierarchical pooling and synthesis for multi-horizon satellite forecasting,
isolating diurnal (24h) and orbital (~12h) periodicities across configurable lookback windows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.forecasting.base import ForecastModel, ModelMetadata


class NHiTSBlock(nn.Module):
    """Hierarchical block with pooling, MLP backbone, and linear basis forecast projection."""

    def __init__(self, in_features: int, out_features: int, pooling_kernel: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.pooling_kernel = pooling_kernel
        pooled_in = max(1, in_features // pooling_kernel)
        self.mlp = nn.Sequential(
            nn.Linear(pooled_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.forecast_head = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, In_Features)
        if self.pooling_kernel > 1:
            x_pool = F.avg_pool1d(x.unsqueeze(1), kernel_size=self.pooling_kernel, stride=self.pooling_kernel).squeeze(1)
        else:
            x_pool = x
        feats = self.mlp(x_pool)
        return self.forecast_head(feats)


class NHiTSNetwork(nn.Module):
    """Multi-rate hierarchical synthesis network for 4 satellite error targets."""

    def __init__(self, in_len: int = 16, out_len: int = 96, n_targets: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.n_targets = n_targets

        # 3 Hierarchical stacks: High-frequency (pool=1), Mid-frequency (pool=2), Low-frequency (pool=4)
        self.stacks = nn.ModuleList([
            nn.ModuleList([NHiTSBlock(in_len, out_len, pooling_kernel=1, hidden_dim=hidden_dim) for _ in range(n_targets)]),
            nn.ModuleList([NHiTSBlock(in_len, out_len, pooling_kernel=2, hidden_dim=hidden_dim) for _ in range(n_targets)]),
            nn.ModuleList([NHiTSBlock(in_len, out_len, pooling_kernel=4, hidden_dim=hidden_dim) for _ in range(n_targets)]),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, In_Len, N_Targets)
        B, L, D = x.shape
        forecast = torch.zeros(B, self.out_len, D, device=x.device, dtype=x.dtype)

        for stack in self.stacks:
            stack_forecasts = []
            for target_idx in range(D):
                target_in = x[:, :, target_idx]
                pred_channel = stack[target_idx](target_in)
                stack_forecasts.append(pred_channel)
            stack_f = torch.stack(stack_forecasts, dim=-1)
            forecast = forecast + stack_f

        return forecast


class NHiTSModel(ForecastModel):
    """N-HiTS multi-rate hierarchical forecaster for multi-horizon satellite predictions."""

    def __init__(
        self,
        name: str = "N-HiTS",
        lookback_steps: int = 16,
        forecast_horizon: int = 96,
        hidden_dim: int = 32,
        epochs: int = 50,
        lr: float = 0.005,
        version: str = "1.0.0",
    ):
        super().__init__(name=name, model_type="nhits", version=version)
        self.lookback_steps = lookback_steps
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.target_cols = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]
        self.network: Optional[NHiTSNetwork] = None
        self.center: np.ndarray = np.zeros(4, dtype=np.float64)
        self.scale: np.ndarray = np.ones(4, dtype=np.float64)

    def fit(self, train_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> "NHiTSModel":
        clean = train_df.dropna(subset=["utc_time", *self.target_cols]).sort_values("utc_time").reset_index(drop=True)
        vals = clean[self.target_cols].to_numpy(dtype=np.float64)

        self.center = np.median(vals, axis=0)
        q25, q75 = np.percentile(vals, [25, 75], axis=0)
        self.scale = np.maximum((q75 - q25) / 1.349, 0.01)

        norm_vals = (vals - self.center) / self.scale

        # Build sliding windows
        sub_horizon = min(len(norm_vals) - self.lookback_steps - 1, 24)
        if sub_horizon <= 0:
            raise ValueError(f"Insufficient training observations for N-HiTS (needs at least {self.lookback_steps + 4} rows)")

        x_list, y_list = [], []
        for i in range(len(norm_vals) - self.lookback_steps - sub_horizon + 1):
            x_list.append(norm_vals[i:i + self.lookback_steps])
            y_list.append(norm_vals[i + self.lookback_steps:i + self.lookback_steps + sub_horizon])

        if not x_list:
            raise ValueError("Could not construct sliding windows for N-HiTS")

        x_arr = np.asarray(x_list, dtype=np.float32)
        y_arr = np.asarray(y_list, dtype=np.float32)

        self.network = NHiTSNetwork(
            in_len=self.lookback_steps,
            out_len=sub_horizon,
            n_targets=4,
            hidden_dim=self.hidden_dim,
        )

        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.lr, weight_decay=1e-4)
        huber = nn.SmoothL1Loss()

        x_tensor = torch.as_tensor(x_arr)
        y_tensor = torch.as_tensor(y_arr)

        self.network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.network(x_tensor)
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
            raise ValueError("NHiTSModel must be fitted before predict()")

        clean_hist = history_df.dropna(subset=["utc_time", *self.target_cols]).sort_values("utc_time").reset_index(drop=True)
        vals = clean_hist[self.target_cols].to_numpy(dtype=np.float64)

        if isinstance(horizon_epochs, int):
            n_steps = horizon_epochs
        else:
            n_steps = len(horizon_epochs)

        norm_vals = (vals - self.center) / self.scale
        if len(norm_vals) < self.lookback_steps:
            pad = np.repeat(norm_vals[[0]], self.lookback_steps - len(norm_vals), axis=0)
            norm_vals = np.vstack([pad, norm_vals])

        input_hist = norm_vals[-self.lookback_steps:]
        in_tensor = torch.as_tensor(input_hist[None, ...], dtype=torch.float32)

        self.network.eval()
        with torch.no_grad():
            sub_pred = self.network(in_tensor).cpu().numpy()[0]  # (sub_horizon, 4)

        # Extrapolate / tile to match required horizon steps
        if len(sub_pred) >= n_steps:
            scaled_pred = sub_pred[:n_steps] * self.scale + self.center
        else:
            repeats = int(np.ceil(n_steps / len(sub_pred)))
            tiled = np.tile(sub_pred, (repeats, 1))[:n_steps]
            scaled_pred = tiled * self.scale + self.center

        return scaled_pred

    def save(self, path: Union[str, Path]) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "model_type": self.model_type,
            "version": self.version,
            "lookback_steps": self.lookback_steps,
            "forecast_horizon": self.forecast_horizon,
            "hidden_dim": self.hidden_dim,
            "network_state_dict": self.network.state_dict() if self.network else None,
            "sub_horizon": self.network.out_len if self.network else 24,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "target_cols": self.target_cols,
        }
        torch.save(payload, out_path)
        return out_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NHiTSModel":
        p = Path(path)
        payload = torch.load(p, map_location="cpu", weights_only=False)
        model = cls(
            name=payload.get("name", "N-HiTS"),
            lookback_steps=payload.get("lookback_steps", 16),
            forecast_horizon=payload.get("forecast_horizon", 96),
            hidden_dim=payload.get("hidden_dim", 32),
            version=payload.get("version", "1.0.0"),
        )
        model.center = np.asarray(payload["center"], dtype=np.float64)
        model.scale = np.asarray(payload["scale"], dtype=np.float64)
        sub_h = payload.get("sub_horizon", 24)

        if payload.get("network_state_dict"):
            model.network = NHiTSNetwork(
                in_len=model.lookback_steps,
                out_len=sub_h,
                n_targets=4,
                hidden_dim=model.hidden_dim,
            )
            model.network.load_state_dict(payload["network_state_dict"])
            model.network.eval()

        model.is_fitted = True
        return model

    def get_metadata(self) -> ModelMetadata:
        n_params = sum(p.numel() for p in self.network.parameters()) if self.network else 5000
        return ModelMetadata(
            name=self.name,
            model_type=self.model_type,
            version=self.version,
            architecture="N-HiTS Multi-Rate Hierarchical Interpolation Network",
            parameters={"lookback_steps": self.lookback_steps, "hidden_dim": self.hidden_dim},
            lookback_steps=self.lookback_steps,
            forecast_horizon=self.forecast_horizon,
            features=["hierarchical_target_windows"],
            target_representation="ECEF",
            supports_uncertainty=False,
            trainable=True,
            description="Multi-rate hierarchical pooling and synthesis isolating multi-frequency orbital harmonics.",
            parameter_count=n_params,
        )
