"""Decoupled Clock Forecasting Model (P4 Architecture).

Separates atomic clock drift modeling from orbital dynamics:
1. Stage 1: Fits analytical quadratic polynomial (phase a0, frequency a1, drift a2) on clock history.
2. Stage 2: 1D Temporal Convolutional Network (TCN) models high-frequency non-linear residuals.
3. Spatial orbit coordinates (X, Y, Z) are predicted via an independent spatial regressor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.forecasting.base import ForecastModel, ModelMetadata
from src.forecasting.models.harmonic_ridge import extract_harmonic_time_features, HarmonicRidgeModel


class ClockTCNResidualNet(nn.Module):
    """Lightweight 1D Dilated ConvNet for atomic clock residual drift."""

    def __init__(self, in_channels: int = 1, hidden_channels: int = 16, num_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_c = in_channels if i == 0 else hidden_channels
            layers.extend([
                nn.Conv1d(in_c, hidden_channels, kernel_size=3, padding=dilation, dilation=dilation),
                nn.SiLU(),
                nn.Dropout(0.1),
            ])
        self.conv_stack = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channels, Seq)
        feats = self.conv_stack(x)
        # Pool last timestep
        out = self.head(feats[:, :, -1])
        return out.squeeze(-1)


class DecoupledClockModel(ForecastModel):
    """Two-stage decoupled forecaster separating polynomial clock drift from spatial orbit dynamics."""

    def __init__(
        self,
        name: str = "Decoupled Clock",
        poly_degree: int = 2,
        tcn_epochs: int = 40,
        version: str = "1.0.0",
    ):
        super().__init__(name=name, model_type="decoupled_clock", version=version)
        self.poly_degree = poly_degree
        self.tcn_epochs = tcn_epochs
        self.orbit_model: Optional[HarmonicRidgeModel] = None
        self.clock_poly_coeffs: Optional[np.ndarray] = None
        self.tcn_net: Optional[ClockTCNResidualNet] = None
        self.origin: Optional[pd.Timestamp] = None
        self.target_cols = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]

    def _clock_poly_features(self, times: Union[pd.Series, pd.DatetimeIndex]) -> np.ndarray:
        index = pd.DatetimeIndex(times)
        t_days = (index - self.origin).total_seconds().to_numpy(dtype=float) / 86400.0
        cols = [np.ones_like(t_days)]
        for d in range(1, self.poly_degree + 1):
            cols.append(t_days ** d)
        return np.column_stack(cols)

    def fit(self, train_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> "DecoupledClockModel":
        clean = train_df.dropna(subset=["utc_time", *self.target_cols]).sort_values("utc_time").reset_index(drop=True)
        self.origin = clean["utc_time"].iloc[0].floor("D")

        # 1. Fit Orbit model (X, Y, Z)
        self.orbit_model = HarmonicRidgeModel(name="Orbit Spatial Baseline")
        self.orbit_model.fit(clean)

        # 2. Fit Quadratic Polynomial Clock Baseline: a0 + a1*t + a2*t^2
        clock_vals = clean["clock_error_m"].to_numpy(dtype=np.float64)
        A_poly = self._clock_poly_features(clean["utc_time"])
        # Least squares solve: (A^T A) c = A^T y
        self.clock_poly_coeffs, _, _, _ = np.linalg.lstsq(A_poly, clock_vals, rcond=None)

        # 3. Compute clock residuals
        poly_pred = A_poly @ self.clock_poly_coeffs
        clock_residuals = clock_vals - poly_pred

        # 4. Train 1D TCN on clock residuals
        seq_len = 8
        if len(clock_residuals) > seq_len + 2:
            x_seqs, y_steps = [], []
            for i in range(len(clock_residuals) - seq_len):
                x_seqs.append(clock_residuals[i:i + seq_len])
                y_steps.append(clock_residuals[i + seq_len])

            x_tensor = torch.as_tensor(np.asarray(x_seqs)[:, None, :], dtype=torch.float32)
            y_tensor = torch.as_tensor(np.asarray(y_steps), dtype=torch.float32)

            self.tcn_net = ClockTCNResidualNet(in_channels=1, hidden_channels=16)
            optimizer = torch.optim.AdamW(self.tcn_net.parameters(), lr=0.005, weight_decay=1e-4)
            huber = nn.SmoothL1Loss()

            self.tcn_net.train()
            for _ in range(self.tcn_epochs):
                optimizer.zero_grad()
                pred = self.tcn_net(x_tensor)
                loss = huber(pred, y_tensor)
                loss.backward()
                optimizer.step()
            self.tcn_net.eval()

        self.is_fitted = True
        return self

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon_epochs: Union[int, pd.DatetimeIndex, Sequence[pd.Timestamp]],
    ) -> np.ndarray:
        if not self.is_fitted or self.orbit_model is None or self.clock_poly_coeffs is None:
            raise ValueError("DecoupledClockModel must be fitted before predict()")

        clean_hist = history_df.dropna(subset=["utc_time", *self.target_cols]).sort_values("utc_time").reset_index(drop=True)
        if isinstance(horizon_epochs, int):
            last_time = pd.to_datetime(clean_hist["utc_time"].iloc[-1])
            step_interval = pd.Timedelta(minutes=15)
            forecast_times = pd.date_range(start=last_time + step_interval, periods=horizon_epochs, freq=step_interval)
        else:
            forecast_times = pd.DatetimeIndex(horizon_epochs)

        # 1. Predict Orbit components (X, Y, Z)
        orbit_preds = self.orbit_model.predict(clean_hist, forecast_times)

        # 2. Predict Clock Baseline
        A_future = self._clock_poly_features(forecast_times)
        poly_clock = A_future @ self.clock_poly_coeffs

        # 3. Predict Clock TCN Residuals if available
        clock_final = poly_clock.copy()
        if self.tcn_net is not None and len(clean_hist) >= 8:
            hist_clock = clean_hist["clock_error_m"].to_numpy(dtype=float)
            A_hist = self._clock_poly_features(clean_hist["utc_time"])
            hist_res = hist_clock - (A_hist @ self.clock_poly_coeffs)

            res_window = list(hist_res[-8:])
            pred_residuals = []
            self.tcn_net.eval()
            with torch.no_grad():
                for _ in range(len(forecast_times)):
                    in_t = torch.as_tensor(np.asarray(res_window[-8:])[None, None, :], dtype=torch.float32)
                    r_pred = float(self.tcn_net(in_t).item())
                    pred_residuals.append(r_pred)
                    res_window.append(r_pred)

            clock_final = clock_final + np.asarray(pred_residuals)

        # Combine into [X, Y, Z, Clock]
        total_pred = np.empty((len(forecast_times), 4), dtype=np.float64)
        total_pred[:, :3] = orbit_preds[:, :3]
        total_pred[:, 3] = clock_final
        return total_pred

    def save(self, path: Union[str, Path]) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "model_type": self.model_type,
            "version": self.version,
            "poly_degree": self.poly_degree,
            "origin": self.origin.isoformat() if self.origin else None,
            "clock_poly_coeffs": self.clock_poly_coeffs,
            "orbit_model": self.orbit_model,
            "tcn_state_dict": self.tcn_net.state_dict() if self.tcn_net else None,
            "target_cols": self.target_cols,
        }
        joblib.dump(payload, out_path)
        return out_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DecoupledClockModel":
        p = Path(path)
        payload = joblib.load(p)
        model = cls(
            name=payload.get("name", "Decoupled Clock"),
            poly_degree=payload.get("poly_degree", 2),
            version=payload.get("version", "1.0.0"),
        )
        model.orbit_model = payload["orbit_model"]
        model.clock_poly_coeffs = payload["clock_poly_coeffs"]
        if payload.get("origin"):
            model.origin = pd.Timestamp(payload["origin"])
        if payload.get("tcn_state_dict"):
            model.tcn_net = ClockTCNResidualNet(in_channels=1, hidden_channels=16)
            model.tcn_net.load_state_dict(payload["tcn_state_dict"])
            model.tcn_net.eval()
        model.is_fitted = True
        return model

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self.name,
            model_type=self.model_type,
            version=self.version,
            architecture="Two-Stage Decoupled: Harmonic Orbit + Quadratic Polynomial + 1D Dilated TCN Clock",
            parameters={"poly_degree": self.poly_degree, "tcn_epochs": self.tcn_epochs},
            lookback_steps=8,
            forecast_horizon=96,
            features=["elapsed_days_poly", "clock_residual_window", "orbit_harmonics"],
            target_representation="ECEF",
            supports_uncertainty=False,
            trainable=True,
            description="Isolates atomic clock physics (quadratic drift + colored noise) from spatial orbital forces.",
            parameter_count=1500,
        )
