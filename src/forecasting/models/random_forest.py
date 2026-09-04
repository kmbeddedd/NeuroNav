"""Random Forest ensemble forecaster."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.forecasting.base import ForecastModel, ModelMetadata
from src.forecasting.models.harmonic_ridge import extract_harmonic_time_features


class RandomForestModel(ForecastModel):
    """Multi-output Random Forest ensemble predicting orbit and clock errors from time dynamics."""

    def __init__(
        self,
        name: str = "Random Forest",
        n_estimators: int = 300,
        min_samples_leaf: int = 2,
        max_features: float = 0.8,
        random_state: int = 42,
        version: str = "1.0.0",
    ):
        super().__init__(name=name, model_type="random_forest", version=version)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.origin: Optional[pd.Timestamp] = None
        self.regressor: Optional[RandomForestRegressor] = None
        self.target_cols = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]

    def fit(self, train_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> "RandomForestModel":
        if "utc_time" not in train_df.columns:
            raise ValueError("train_df missing 'utc_time' column")
        available_cols = [c for c in self.target_cols if c in train_df.columns]
        if len(available_cols) != 4:
            raise ValueError(f"train_df must contain target columns: {self.target_cols}")

        clean = train_df.dropna(subset=["utc_time", *self.target_cols]).sort_values("utc_time")
        self.origin = clean["utc_time"].iloc[0].floor("D")

        x_train = extract_harmonic_time_features(clean["utc_time"], self.origin, harmonics=6)
        y_train = clean[self.target_cols].to_numpy(dtype=np.float64)

        self.regressor = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.regressor.fit(x_train, y_train)
        self.is_fitted = True
        return self

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon_epochs: Union[int, pd.DatetimeIndex, Sequence[pd.Timestamp]],
    ) -> np.ndarray:
        if not self.is_fitted or self.regressor is None or self.origin is None:
            raise ValueError("RandomForestModel must be fitted before predict()")

        if isinstance(horizon_epochs, int):
            if history_df.empty or "utc_time" not in history_df.columns:
                raise ValueError("history_df with 'utc_time' is required when horizon_epochs is an integer")
            last_time = pd.to_datetime(history_df["utc_time"].iloc[-1])
            step_interval = pd.Timedelta(minutes=15)
            forecast_times = pd.date_range(start=last_time + step_interval, periods=horizon_epochs, freq=step_interval)
        else:
            forecast_times = pd.DatetimeIndex(horizon_epochs)

        x_test = extract_harmonic_time_features(forecast_times, self.origin, harmonics=6)
        return self.regressor.predict(x_test)

    def save(self, path: Union[str, Path]) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "model_type": self.model_type,
            "version": self.version,
            "n_estimators": self.n_estimators,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "origin": self.origin.isoformat() if self.origin else None,
            "regressor": self.regressor,
            "target_cols": self.target_cols,
        }
        joblib.dump(payload, out_path)
        return out_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RandomForestModel":
        p = Path(path)
        payload = joblib.load(p)
        model = cls(
            name=payload.get("name", "Random Forest"),
            n_estimators=payload.get("n_estimators", 300),
            min_samples_leaf=payload.get("min_samples_leaf", 2),
            max_features=payload.get("max_features", 0.8),
            random_state=payload.get("random_state", 42),
            version=payload.get("version", "1.0.0"),
        )
        model.regressor = payload["regressor"]
        if payload.get("origin"):
            model.origin = pd.Timestamp(payload["origin"])
        model.is_fitted = True
        return model

    def get_metadata(self) -> ModelMetadata:
        n_trees = len(self.regressor.estimators_) if self.regressor and hasattr(self.regressor, "estimators_") else 0
        return ModelMetadata(
            name=self.name,
            model_type=self.model_type,
            version=self.version,
            architecture="Random Forest Multi-Output Ensemble",
            parameters={
                "n_estimators": self.n_estimators,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
            },
            lookback_steps=12,
            forecast_horizon=96,
            features=["elapsed_days", "elapsed_days_sq", "harmonics_1_to_6"],
            target_representation="ECEF",
            supports_uncertainty=False,
            trainable=True,
            description="Non-linear decision tree ensemble mapping time and solar harmonics to multi-axis orbit/clock deviations.",
            parameter_count=n_trees * 50,
        )
