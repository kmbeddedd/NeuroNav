"""Base interface and common definitions for satellite-specific forecasting models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class ModelMetadata:
    """Metadata container describing a forecasting model."""
    name: str
    model_type: str
    version: str
    architecture: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    lookback_steps: int = 16
    forecast_horizon: int = 96
    features: List[str] = field(default_factory=list)
    requires_physics_features: bool = False
    physics_features: List[str] = field(default_factory=list)
    target_representation: str = "ECEF"  # "ECEF" or "RIC"
    supports_uncertainty: bool = False
    trainable: bool = True
    description: str = ""
    parameter_count: int = 0
    artifact_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "version": self.version,
            "architecture": self.architecture,
            "parameters": self.parameters,
            "lookback_steps": self.lookback_steps,
            "forecast_horizon": self.forecast_horizon,
            "features": self.features,
            "requires_physics_features": self.requires_physics_features,
            "physics_features": self.physics_features,
            "target_representation": self.target_representation,
            "supports_uncertainty": self.supports_uncertainty,
            "trainable": self.trainable,
            "description": self.description,
            "parameter_count": self.parameter_count,
            "artifact_path": self.artifact_path,
        }


class ForecastModel(ABC):
    """Abstract Base Class for all satellite-specific forecasting model adapters."""

    supports_feature_manifest: bool = False
    supports_nominal_physics: bool = False
    supports_provided_state: bool = False
    supports_irregular_timestamps: bool = True
    requires_regular_cadence: bool = False

    def __init__(self, name: str, model_type: str, version: str = "1.0.0"):
        self.name = name
        self.model_type = model_type
        self.version = version
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> "ForecastModel":
        """Fits the forecasting model on historical training data.
        
        Args:
            train_df: DataFrame containing at least utc_time/Timestamp and target error columns.
            config: Optional training and hyperparameter configuration dictionary.
        """
        pass

    @abstractmethod
    def predict(
        self,
        history_df: pd.DataFrame,
        horizon_epochs: Union[int, pd.DatetimeIndex, Sequence[pd.Timestamp]],
    ) -> np.ndarray:
        """Generates future predictions given recent historical telemetry.
        
        Args:
            history_df: DataFrame containing lookback history.
            horizon_epochs: Integer forecast horizon step count, or specific timestamps to forecast.
            
        Returns:
            np.ndarray of shape (horizon_steps, 4) with columns [x_err, y_err, z_err, clock_err].
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> Path:
        """Persists fitted model weights and metadata to file.
        
        Returns:
            Path to saved artifact.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> "ForecastModel":
        """Loads a persisted model from artifact file."""
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Returns structured metadata describing model configuration, capacity, and parameters."""
        pass
