"""Prediction-Time Routing Engine for Satellite-Specific Inference.

Inspects incoming multi-satellite telemetry, queries the SatelliteModelRegistry,
dynamically loads the assigned model for each satellite, and generates standardized
predictions containing full provenance metadata.

CRITICAL INVARIANT: Zero silent fallbacks to BiLSTM. Unknown or unselected satellites
must raise explicit errors or return structured failure statuses.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.forecasting.base import ForecastModel
from src.forecasting.models import MODEL_REGISTRY, create_model
from src.forecasting.registry.store import SatelliteModelRegistry, SatelliteSelection
from src.forecasting.data.validation import load_telemetry_source, normalize_dataframe_columns
from src.forecasting.physics import (
    NominalStateProvider,
    OrbitalStateProvider,
    ProvidedStateProvider,
    ecef_error_to_ric,
    nominal_satellite_orbit,
)

logger = logging.getLogger(__name__)


class RoutingError(Exception):
    """Base exception for router failures."""
    pass


class NoModelSelectionError(RoutingError):
    """Raised when inference is requested for a satellite with no active model assignment."""
    def __init__(self, satellite_id: str):
        super().__init__(
            f"No model selection found in registry for satellite '{satellite_id}'. "
            f"Run calibration or call set_model_for_satellite() before requesting predictions."
        )


class ModelArtifactError(RoutingError):
    """Raised when a model artifact cannot be found or loaded from disk."""
    def __init__(self, satellite_id: str, model_name: str, reason: str = ""):
        super().__init__(
            f"Failed to load model artifact for satellite '{satellite_id}' (model: '{model_name}'): {reason}"
        )


class PredictionRouter:
    """Routes inference requests to winning satellite-specific models with strict fail-closed safety."""

    def __init__(
        self,
        registry: Optional[SatelliteModelRegistry] = None,
        artifacts_dir: Union[str, Path] = "models/registry/artifacts",
    ):
        self.registry = registry if registry is not None else SatelliteModelRegistry()
        self.artifacts_dir = Path(artifacts_dir)
        self._loaded_models: Dict[str, ForecastModel] = {}

    def _resolve_state_provider(
        self, satellite_id: str, selection: SatelliteSelection
    ) -> Optional[OrbitalStateProvider]:
        """Resolves the authoritative OrbitalStateProvider for a satellite based on registered physics mode."""
        phys_mode = getattr(selection, "physics_mode", "nominal")
        source = getattr(selection, "orbit_state_source", "nominal_approximation")

        if phys_mode == "none" or source == "none":
            return None

        if phys_mode == "provided" or source == "provided":
            state_candidates: List[Path] = []
            if getattr(selection, "state_artifact", None):
                state_candidates.append(Path(selection.state_artifact))
            state_candidates.extend([
                self.artifacts_dir / "satellites" / satellite_id / "orbital_state.csv",
                self.artifacts_dir / f"{satellite_id}_orbital_state.csv",
            ])
            for p in state_candidates:
                if p.exists():
                    state_df = pd.read_csv(p)
                    return ProvidedStateProvider(state_df=state_df)
            raise ModelArtifactError(
                satellite_id=satellite_id,
                model_name=selection.selected_model,
                reason=f"Provided-state physics artifact not found. Checked: {[str(p) for p in state_candidates]}",
            )

        return NominalStateProvider()

    def get_assigned_model(self, satellite_id: str) -> Tuple[ForecastModel, SatelliteSelection]:
        """Loads and returns the assigned model instance and metadata for a satellite."""
        selection = self.registry.get_selection(satellite_id)
        if selection is None:
            raise NoModelSelectionError(satellite_id)

        model_name = selection.selected_model
        cache_key = f"{satellite_id}:{model_name}:{selection.model_version}"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key], selection

        # Locate artifact path: if explicitly registered, verify that path strictly.
        target_path = None
        if selection.model_artifact:
            p = Path(selection.model_artifact)
            if p.exists():
                target_path = p
            elif not p.is_absolute() and (self.artifacts_dir / p).exists():
                target_path = self.artifacts_dir / p
            else:
                raise ModelArtifactError(
                    satellite_id=satellite_id,
                    model_name=model_name,
                    reason=f"Registered model artifact not found on disk: {p}",
                )
        else:
            candidate_paths = [
                self.artifacts_dir / "satellites" / satellite_id / "model.pt",
                self.artifacts_dir / "satellites" / satellite_id / "model.joblib",
                self.artifacts_dir / "satellites" / satellite_id / f"{satellite_id}_{model_name}.pt",
                self.artifacts_dir / "satellites" / satellite_id / f"{satellite_id}_{model_name}.joblib",
                self.artifacts_dir / f"{satellite_id}_{model_name}.pt",
                self.artifacts_dir / f"{satellite_id}_{model_name}.joblib",
                self.artifacts_dir / f"{model_name}.pt",
                self.artifacts_dir / f"{model_name}.joblib",
            ]
            for p in candidate_paths:
                if p.exists():
                    target_path = p
                    break

            if target_path is None:
                raise ModelArtifactError(
                    satellite_id=satellite_id,
                    model_name=model_name,
                    reason=f"Artifact not found. Checked: {[str(p) for p in candidate_paths]}",
                )

        if model_name not in MODEL_REGISTRY:
            raise ModelArtifactError(
                satellite_id=satellite_id,
                model_name=model_name,
                reason=f"Model '{model_name}' is not in the canonical MODEL_REGISTRY",
            )

        model_cls = MODEL_REGISTRY[model_name]
        try:
            model = model_cls.load(target_path)
        except Exception as exc:
            raise ModelArtifactError(
                satellite_id=satellite_id,
                model_name=model_name,
                reason=f"Failed to load artifact at {target_path}: {exc}",
            ) from exc

        # Propagate resolved provider into model if supported
        provider = self._resolve_state_provider(satellite_id, selection)
        if provider is not None and hasattr(model, "orbital_state_provider"):
            model.orbital_state_provider = provider

        self._loaded_models[cache_key] = model
        return model, selection

    def predict(
        self,
        data: Union[str, Path, pd.DataFrame, Dict[str, pd.DataFrame]],
        horizon_steps: int = 96,
        step_interval_minutes: int = 15,
        compute_ric: bool = True,
    ) -> pd.DataFrame:
        """Executes independent satellite-specific inference across all satellites present in data.
        
        Returns:
            Standardized DataFrame with predictions and provenance metadata.
        """
        series_by_sat = load_telemetry_source(data)
        if not series_by_sat:
            raise RoutingError("No telemetry data found to forecast")

        records: List[pd.DataFrame] = []

        for sat_id, sat_df in series_by_sat.items():
            model, selection = self.get_assigned_model(sat_id)

            clean_hist = sat_df.dropna(subset=["utc_time"]).sort_values("utc_time")
            if clean_hist.empty:
                raise RoutingError(f"Satellite '{sat_id}' telemetry has no valid timestamps")

            cadence = getattr(selection, "cadence_minutes", None)
            eff_cadence = cadence if (cadence is not None and step_interval_minutes == 15) else step_interval_minutes
            last_time = pd.to_datetime(clean_hist["utc_time"].iloc[-1])
            step_delta = pd.Timedelta(minutes=eff_cadence)
            forecast_times = pd.date_range(start=last_time + step_delta, periods=horizon_steps, freq=step_delta)

            pred_arr = model.predict(sat_df, forecast_times)

            pred_x = pred_arr[:, 0]
            pred_y = pred_arr[:, 1]
            pred_z = pred_arr[:, 2]
            pred_clk = pred_arr[:, 3]
            pred_3d_norm = np.sqrt(pred_x ** 2 + pred_y ** 2 + pred_z ** 2)

            sat_record: Dict[str, Any] = {
                "forecast_step": np.arange(1, horizon_steps + 1),
                "timestamp": forecast_times,
                "satellite_id": sat_id,
                "predicted_X": pred_x,
                "predicted_Y": pred_y,
                "predicted_Z": pred_z,
                "predicted_Clock": pred_clk,
                "pred_3D_Orbit_Error": pred_3d_norm,
                "model_used": selection.selected_model,
                "model_version": selection.model_version,
                "selection_mode": selection.selection_mode,
            }

            if compute_ric:
                provider = self._resolve_state_provider(sat_id, selection)
                if provider is not None:
                    orbit_type = getattr(selection, "orbit_type", sat_id)
                    pos, vel = provider.get_state(forecast_times, orbit_class=orbit_type, satellite_id=sat_id)
                    error_ecef = np.column_stack([pred_x, pred_y, pred_z])
                    ric_err = ecef_error_to_ric(error_ecef, pos, vel)
                    sat_record["predicted_R"] = ric_err[:, 0]
                    sat_record["predicted_I"] = ric_err[:, 1]
                    sat_record["predicted_C"] = ric_err[:, 2]

            df_sat = pd.DataFrame(sat_record)
            records.append(df_sat)

        return pd.concat(records, ignore_index=True)

    def predict_single_satellite(
        self,
        satellite_id: str,
        history_df: pd.DataFrame,
        horizon_steps: int = 96,
        step_interval_minutes: Optional[int] = None,
        compute_ric: bool = True,
    ) -> pd.DataFrame:
        """Executes forecast inference for a single specified satellite."""
        model, selection = self.get_assigned_model(satellite_id)
        clean_hist = history_df.dropna(subset=["utc_time"]).sort_values("utc_time")
        if clean_hist.empty:
            raise RoutingError(f"Satellite '{satellite_id}' telemetry has no valid timestamps")

        cadence = step_interval_minutes or int(getattr(selection, "cadence_minutes", 15) or 15)
        last_time = pd.to_datetime(clean_hist["utc_time"].iloc[-1])
        step_delta = pd.Timedelta(minutes=cadence)
        forecast_times = pd.date_range(start=last_time + step_delta, periods=horizon_steps, freq=step_delta)

        pred_arr = model.predict(clean_hist, forecast_times)

        pred_x = pred_arr[:, 0]
        pred_y = pred_arr[:, 1]
        pred_z = pred_arr[:, 2]
        pred_clk = pred_arr[:, 3]
        pred_3d_norm = np.sqrt(pred_x ** 2 + pred_y ** 2 + pred_z ** 2)

        sat_record: Dict[str, Any] = {
            "forecast_step": np.arange(1, horizon_steps + 1),
            "timestamp": forecast_times,
            "satellite_id": satellite_id,
            "predicted_X": pred_x,
            "predicted_Y": pred_y,
            "predicted_Z": pred_z,
            "predicted_Clock": pred_clk,
            "pred_3D_Orbit_Error": pred_3d_norm,
            "model_used": selection.selected_model,
            "model_version": selection.model_version,
            "selection_mode": selection.selection_mode,
        }

        if compute_ric:
            provider = self._resolve_state_provider(satellite_id, selection)
            if provider is not None:
                orbit_type = getattr(selection, "orbit_type", satellite_id)
                pos, vel = provider.get_state(forecast_times, orbit_class=orbit_type, satellite_id=satellite_id)
                error_ecef = np.column_stack([pred_x, pred_y, pred_z])
                ric_err = ecef_error_to_ric(error_ecef, pos, vel)
                sat_record["predicted_R"] = ric_err[:, 0]
                sat_record["predicted_I"] = ric_err[:, 1]
                sat_record["predicted_C"] = ric_err[:, 2]

        return pd.DataFrame(sat_record)
