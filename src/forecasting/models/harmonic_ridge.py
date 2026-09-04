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
from src.forecasting.features.core import (
    FeatureManifest,
    build_inference_features,
    build_training_features,
    extract_harmonic_time_features,
)
from src.forecasting.physics import OrbitalStateProvider


class HarmonicRidgeModel(ForecastModel):
    """Fits analytical polynomial drift + diurnal harmonics via regularized Ridge regression."""

    supports_feature_manifest: bool = True
    supports_nominal_physics: bool = True
    supports_provided_state: bool = True
    supports_irregular_timestamps: bool = True
    requires_regular_cadence: bool = False

    def __init__(
        self,
        name: str = "Harmonic Ridge",
        alpha: float = 1.0,
        harmonics: int = 6,
        use_srp: bool = False,
        use_ric: bool = False,
        physics_mode: str = "nominal",
        cadence_minutes: float = 15.0,
        orbital_state_provider: Optional[OrbitalStateProvider] = None,
        orbit_class: str = "MEO",
        satellite_id: str = "",
        version: str = "1.0.0",
    ):
        super().__init__(name=name, model_type="harmonic_ridge", version=version)
        self.alpha = alpha
        self.harmonics = harmonics
        self.physics_mode = physics_mode
        self.use_srp = bool(use_srp)
        self.use_ric = bool(use_ric)
        self.cadence_minutes = float(cadence_minutes)
        self.orbital_state_provider = orbital_state_provider
        self.orbit_class = orbit_class
        self.satellite_id = satellite_id
        self.origin: Optional[pd.Timestamp] = None
        self.pipeline: Optional[Any] = None
        self.target_cols = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]
        self.feature_manifest = FeatureManifest(
            physics_mode=self.physics_mode,
            use_ric=self.use_ric,
            use_srp=self.use_srp,
            cadence_minutes=self.cadence_minutes,
            harmonics_count=self.harmonics,
        )

    def fit(self, train_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> "HarmonicRidgeModel":
        if "utc_time" not in train_df.columns:
            raise ValueError("train_df missing 'utc_time' column")
        available_cols = [c for c in self.target_cols if c in train_df.columns]
        if len(available_cols) != 4:
            raise ValueError(f"train_df must contain target columns: {self.target_cols}")

        clean = train_df.dropna(subset=["utc_time", *self.target_cols]).sort_values("utc_time")
        self.feature_manifest.use_ric = self.use_ric
        self.feature_manifest.use_srp = self.use_srp
        self.feature_manifest.cadence_minutes = self.cadence_minutes

        if self.use_ric or self.use_srp:
            x_train, _, self.origin = build_training_features(
                clean,
                manifest=self.feature_manifest,
                origin=self.origin,
                provider=self.orbital_state_provider,
                satellite_id=self.satellite_id,
                orbit_class=self.orbit_class,
            )
        else:
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
            step_interval = pd.Timedelta(minutes=self.cadence_minutes)
            forecast_times = pd.date_range(start=last_time + step_interval, periods=horizon_epochs, freq=step_interval)
        else:
            forecast_times = pd.DatetimeIndex(horizon_epochs)

        if self.use_ric or self.use_srp:
            x_test, _ = build_inference_features(
                forecast_times,
                origin=self.origin,
                manifest=self.feature_manifest,
                provider=self.orbital_state_provider,
                satellite_id=self.satellite_id,
                orbit_class=self.orbit_class,
                history_df=history_df,
            )
        else:
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
            "use_srp": self.use_srp,
            "use_ric": self.use_ric,
            "cadence_minutes": self.cadence_minutes,
            "feature_manifest": self.feature_manifest.to_dict(),
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
            use_srp=payload.get("use_srp", False),
            use_ric=payload.get("use_ric", False),
            cadence_minutes=payload.get("cadence_minutes", 15.0),
            version=payload.get("version", "1.0.0"),
        )
        model.pipeline = payload["pipeline"]
        if payload.get("feature_manifest"):
            model.feature_manifest = FeatureManifest.from_dict(payload["feature_manifest"])
        if payload.get("origin"):
            model.origin = pd.Timestamp(payload["origin"])
        model.is_fitted = True
        return model

    def get_metadata(self) -> ModelMetadata:
        n_params = 0
        if self.pipeline and hasattr(self.pipeline.named_steps["ridge"], "coef_"):
            n_params = self.pipeline.named_steps["ridge"].coef_.size + self.pipeline.named_steps["ridge"].intercept_.size

        phys_suffix = []
        if self.use_srp:
            phys_suffix.append("SRP Physics")
        if self.use_ric:
            phys_suffix.append("RIC Coordinates")
        arch_suffix = f" + {' + '.join(phys_suffix)}" if phys_suffix else ""

        return ModelMetadata(
            name=self.name,
            model_type=self.model_type,
            version=self.version,
            architecture=f"Ridge-Regularized Polynomial + Multi-Harmonic Extrapolator{arch_suffix}",
            parameters={
                "alpha": self.alpha,
                "harmonics": self.harmonics,
                "use_srp": self.use_srp,
                "use_ric": self.use_ric,
            },
            lookback_steps=12,
            forecast_horizon=96,
            features=self.feature_manifest.features if (self.use_ric or self.use_srp) and self.feature_manifest.features else ["elapsed_days", "elapsed_days_squared", "sin_k_phase", "cos_k_phase"],
            target_representation="ECEF",
            supports_uncertainty=False,
            trainable=True,
            description="L2 regularized multi-frequency harmonic model capturing diurnal and orbital resonance" + arch_suffix + ".",
            parameter_count=n_params,
        )
