"""Harmonic Ridge regularized baseline forecaster."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.forecasting.base import ForecastModel, ModelMetadata


def extract_harmonic_time_features(
    times: Union[pd.Series, pd.DatetimeIndex],
    origin: pd.Timestamp,
    harmonics: int = 6,
) -> np.ndarray:
    """Extracts secular and diurnal harmonic phase features from timestamps."""
    index = pd.DatetimeIndex(times)
    elapsed_days = (index - origin).total_seconds().to_numpy(dtype=np.float64) / 86400.0
    phase = 2.0 * np.pi * (
        index.hour.to_numpy(dtype=np.float64) * 3600.0
        + index.minute.to_numpy(dtype=np.float64) * 60.0
        + index.second.to_numpy(dtype=np.float64)
    ) / 86400.0

    columns = [elapsed_days, np.square(elapsed_days) / 49.0]
    for h in range(1, harmonics + 1):
        columns.extend([np.sin(h * phase), np.cos(h * phase)])

    return np.column_stack(columns).astype(np.float64)


class HarmonicRidgeModel(ForecastModel):
    """Fits analytical polynomial drift + diurnal harmonics via regularized Ridge regression."""

    def __init__(
        self,
        name: str = "Harmonic Ridge",
        alpha: float = 1.0,
        harmonics: int = 6,
        version: str = "1.0.0",
    ):
        super().__init__(name=name, model_type="harmonic_ridge", version=version)
        self.alpha = alpha
        self.harmonics = harmonics
        self.origin: Optional[pd.Timestamp] = None
        self.pipeline: Optional[Any] = None
        self.target_cols = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]

    def fit(self, train_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> "HarmonicRidgeModel":
        if "utc_time" not in train_df.columns:
            raise ValueError("train_df missing 'utc_time' column")
        available_cols = [c for c in self.target_cols if c in train_df.columns]
        if len(available_cols) != 4:
            raise ValueError(f"train_df must contain target columns: {self.target_cols}")

        clean = train_df.dropna(subset=["utc_time", *self.target_cols]).sort_values("utc_time")
        self.origin = clean["utc_time"].iloc[0].floor("D")

        x_train = extract_harmonic_time_features(clean["utc_time"], self.origin, self.harmonics)
        y_train = clean[self.target_cols].to_numpy(dtype=np.float64)

        self.pipeline = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.pipeline.fit(x_train, y_train)
        self.is_fitted = True
        return self

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon_epochs: Union[int, pd.DatetimeIndex, Sequence[pd.Timestamp]],
    ) -> np.ndarray:
        if not self.is_fitted or self.pipeline is None or self.origin is None:
            raise ValueError("HarmonicRidgeModel must be fitted before predict()")

        if isinstance(horizon_epochs, int):
            # Extrapolate forward from history_df end time
            if history_df.empty or "utc_time" not in history_df.columns:
                raise ValueError("history_df with 'utc_time' is required when horizon_epochs is an integer")
            last_time = pd.to_datetime(history_df["utc_time"].iloc[-1])
            step_interval = pd.Timedelta(minutes=15)
            forecast_times = pd.date_range(start=last_time + step_interval, periods=horizon_epochs, freq=step_interval)
        else:
            forecast_times = pd.DatetimeIndex(horizon_epochs)

        x_test = extract_harmonic_time_features(forecast_times, self.origin, self.harmonics)
        return self.pipeline.predict(x_test)

    def save(self, path: Union[str, Path]) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "model_type": self.model_type,
            "version": self.version,
            "alpha": self.alpha,
            "harmonics": self.harmonics,
            "origin": self.origin.isoformat() if self.origin else None,
            "pipeline": self.pipeline,
            "target_cols": self.target_cols,
        }
        joblib.dump(payload, out_path)
        return out_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HarmonicRidgeModel":
        p = Path(path)
        payload = joblib.load(p)
        model = cls(
            name=payload.get("name", "Harmonic Ridge"),
            alpha=payload.get("alpha", 1.0),
            harmonics=payload.get("harmonics", 6),
            version=payload.get("version", "1.0.0"),
        )
        model.pipeline = payload["pipeline"]
        if payload.get("origin"):
            model.origin = pd.Timestamp(payload["origin"])
        model.is_fitted = True
        return model

    def get_metadata(self) -> ModelMetadata:
        n_params = 0
        if self.pipeline and hasattr(self.pipeline.named_steps["ridge"], "coef_"):
            n_params = self.pipeline.named_steps["ridge"].coef_.size + self.pipeline.named_steps["ridge"].intercept_.size

        return ModelMetadata(
            name=self.name,
            model_type=self.model_type,
            version=self.version,
            architecture="Ridge-Regularized Polynomial + Multi-Harmonic Extrapolator",
            parameters={"alpha": self.alpha, "harmonics": self.harmonics},
            lookback_steps=12,
            forecast_horizon=96,
            features=["elapsed_days", "elapsed_days_squared", "sin_k_phase", "cos_k_phase"],
            target_representation="ECEF",
            supports_uncertainty=False,
            trainable=True,
            description="L2 regularized multi-frequency harmonic model capturing diurnal and orbital orbital resonance.",
            parameter_count=n_params,
        )
