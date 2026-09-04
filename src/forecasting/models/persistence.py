"""Persistence baseline forecasting model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.forecasting.base import ForecastModel, ModelMetadata


class PersistenceModel(ForecastModel):
    """Extrapolates the last known valid observation forward across the forecast horizon."""

    def __init__(self, name: str = "Persistence", version: str = "1.0.0", **kwargs):
        super().__init__(name=name, model_type="persistence", version=version)
        self.last_state: Optional[np.ndarray] = None
        self.target_cols = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]

    def fit(self, train_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> "PersistenceModel":
        available_cols = [c for c in self.target_cols if c in train_df.columns]
        if not available_cols:
            raise ValueError(f"train_df missing target columns: {self.target_cols}")
        
        last_row = train_df[self.target_cols].dropna().iloc[-1]
        self.last_state = last_row.to_numpy(dtype=np.float64)
        self.is_fitted = True
        return self

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon_epochs: Union[int, pd.DatetimeIndex, Sequence[pd.Timestamp]],
    ) -> np.ndarray:
        if isinstance(horizon_epochs, int):
            n_steps = horizon_epochs
        else:
            n_steps = len(horizon_epochs)

        if not history_df.empty and set(self.target_cols).issubset(history_df.columns):
            valid = history_df[self.target_cols].dropna()
            if not valid.empty:
                anchor = valid.iloc[-1].to_numpy(dtype=np.float64)
            elif self.last_state is not None:
                anchor = self.last_state
            else:
                raise ValueError("No valid history or fitted state available for persistence prediction")
        elif self.last_state is not None:
            anchor = self.last_state
        else:
            raise ValueError("PersistenceModel must be fitted or given valid history")

        return np.repeat(anchor[None, :], n_steps, axis=0)

    def save(self, path: Union[str, Path]) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "model_type": self.model_type,
            "version": self.version,
            "last_state": self.last_state.tolist() if self.last_state is not None else None,
            "target_cols": self.target_cols,
        }
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PersistenceModel":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        model = cls(name=data.get("name", "Persistence"), version=data.get("version", "1.0.0"))
        if data.get("last_state") is not None:
            model.last_state = np.asarray(data["last_state"], dtype=np.float64)
            model.is_fitted = True
        return model

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name=self.name,
            model_type=self.model_type,
            version=self.version,
            architecture="Zero-order hold persistence",
            lookback_steps=1,
            forecast_horizon=96,
            features=["last_observed_state"],
            target_representation="ECEF",
            supports_uncertainty=False,
            trainable=False,
            description="Extrapolates the final valid observation continuously through the forecast horizon.",
            parameter_count=4,
        )
