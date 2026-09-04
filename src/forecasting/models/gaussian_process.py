"""Gaussian Process Regressor with periodic and RBF kernel."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, ExpSineSquared, RBF, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor

from src.forecasting.base import ForecastModel, ModelMetadata


class GaussianProcessModel(ForecastModel):
    """Gaussian Process with compound periodic and radial basis function kernel."""

    def __init__(
        self,
        name: str = "Gaussian Process",
        random_state: int = 42,
        version: str = "1.0.0",
        **kwargs,
    ):
        super().__init__(name=name, model_type="gaussian_process", version=version)
        self.orbit_class = kwargs.get("orbit_class", "MEO")
        self.random_state = random_state
        self.origin: Optional[pd.Timestamp] = None
        self.model: Optional[MultiOutputRegressor] = None
        self.target_cols = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]

    def fit(self, train_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> "GaussianProcessModel":
        if "utc_time" not in train_df.columns:
            raise ValueError("train_df missing 'utc_time' column")
        available_cols = [c for c in self.target_cols if c in train_df.columns]
        if len(available_cols) != 4:
            raise ValueError(f"train_df must contain target columns: {self.target_cols}")

        clean = train_df.dropna(subset=["utc_time", *self.target_cols]).sort_values("utc_time")
        self.origin = clean["utc_time"].iloc[0].floor("D")

        train_days = (clean["utc_time"] - self.origin).dt.total_seconds().to_numpy(dtype=float).reshape(-1, 1) / 86400.0
        y_train = clean[self.target_cols].to_numpy(dtype=np.float64)

        kernel = (
            ConstantKernel(1.0, (0.01, 100.0))
            * (
                RBF(length_scale=1.0, length_scale_bounds=(0.05, 20.0))
                + ExpSineSquared(
                    length_scale=1.0,
                    periodicity=1.0,
                    length_scale_bounds=(0.05, 20.0),
                    periodicity_bounds=(0.8, 1.2),
                )
            )
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 100.0))
        )

        base_gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=self.random_state,
        )
        self.model = MultiOutputRegressor(base_gpr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(train_days, y_train)

        self.is_fitted = True
        return self

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon_epochs: Union[int, pd.DatetimeIndex, Sequence[pd.Timestamp]],
    ) -> np.ndarray:
        if not self.is_fitted or self.model is None or self.origin is None:
            raise ValueError("GaussianProcessModel must be fitted before predict()")

        if isinstance(horizon_epochs, int):
            if history_df.empty or "utc_time" not in history_df.columns:
                raise ValueError("history_df with 'utc_time' is required when horizon_epochs is an integer")
            last_time = pd.to_datetime(history_df["utc_time"].iloc[-1])
            step_interval = pd.Timedelta(minutes=15)
            forecast_times = pd.date_range(start=last_time + step_interval, periods=horizon_epochs, freq=step_interval)
        else:
            forecast_times = pd.DatetimeIndex(horizon_epochs)

        test_days = (forecast_times - self.origin).total_seconds().to_numpy(dtype=float).reshape(-1, 1) / 86400.0
        return self.model.predict(test_days)

    def save(self, path: Union[str, Path]) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "model_type": self.model_type,
            "version": self.version,
            "random_state": self.random_state,
            "origin": self.origin.isoformat() if self.origin else None,
            "model": self.model,
            "target_cols": self.target_cols,
        }
        joblib.dump(payload, out_path)
        return out_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GaussianProcessModel":
        p = Path(path)
        payload = joblib.load(p)
        model = cls(
            name=payload.get("name", "Gaussian Process"),
            random_state=payload.get("random_state", 42),
            version=payload.get("version", "1.0.0"),
        )
        model.model = payload["model"]
        if payload.get("origin"):
            model.origin = pd.Timestamp(payload["origin"])
        model.is_fitted = True
        return model

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self.name,
            model_type=self.model_type,
            version=self.version,
            architecture="Gaussian Process Regressor with Compound RBF + ExpSineSquared Kernel",
            parameters={"kernel": "Constant * (RBF + ExpSineSquared) + WhiteKernel"},
            lookback_steps=8,
            forecast_horizon=96,
            features=["elapsed_days"],
            target_representation="ECEF",
            supports_uncertainty=True,
            trainable=True,
            description="Non-parametric Bayesian model with periodic kernel capturing orbital cycles and uncertainty.",
            parameter_count=12,
        )
