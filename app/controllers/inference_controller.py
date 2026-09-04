"""Controller for NeuroNav Desktop Application, decoupling UI from inference and calibration engines."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.calibration_engine import SatelliteCalibrationEngine
from src.inference import (
    NeuroNavModel,
    SatelliteAutoForecaster,
    predict_satellite_heterogeneous,
)
from src.models.adapters import MODEL_ADAPTER_CLASSES, get_available_model_adapters
from src.satellite_registry import SatelliteModelRegistry
from src.visualization.forecast import plot_3d_orbit_error, plot_forecast_components


class InferenceController:
    """Orchestrates model calibration, satellite model registry, and heterogeneous forecasting."""

    def __init__(
        self,
        models_dir: str | Path = 'models/deploy',
        registry_path: Optional[str | Path] = None,
    ):
        self.models_dir = Path(models_dir)
        self.registry = SatelliteModelRegistry(registry_path=registry_path)
        self.calibration_engine = SatelliteCalibrationEngine(registry=self.registry)

        # Default model mode is now Auto (Satellite-Specific Best Model)
        self.current_model_mode: str = 'auto'
        self.current_model: Optional[Any] = SatelliteAutoForecaster(registry=self.registry)

        self.current_data: Optional[pd.DataFrame] = None
        self.current_dataset_path: Optional[str] = None
        self.current_forecast: Optional[pd.DataFrame] = None
        self.last_calibration_result: Optional[Dict[str, Any]] = None

    def get_available_models(self) -> List[Dict[str, str]]:
        """Return list of all discoverable models including the automatic satellite selector."""
        models = [
            {
                'id': 'auto',
                'name': 'Auto — Best Model for Satellite',
                'path': 'registry',
                'description': 'Per-satellite persistent model memory routing each satellite to its calibrated winner',
            }
        ]

        # Add all registered adapters
        adapters = get_available_model_adapters()
        for m_id, adapter in adapters.items():
            models.append({
                'id': m_id,
                'name': adapter.model_name,
                'path': f'adapter:{m_id}',
                'description': adapter.description,
            })

        # Add deployable neural models from disk if present
        if self.models_dir.exists():
            for d in self.models_dir.iterdir():
                if d.is_dir() and (d / 'model.pt').exists():
                    manifest_f = d / 'manifest.json'
                    name = d.name
                    desc = f"{d.name} model"
                    if manifest_f.exists():
                        try:
                            meta = json.loads(manifest_f.read_text(encoding='utf-8'))
                            name = meta.get('model_name', d.name)
                            desc = meta.get('description', desc)
                        except Exception:
                            pass
                    # Avoid duplicate id if already in adapters
                    if not any(m['id'] == d.name for m in models):
                        models.append({
                            'id': d.name,
                            'name': name,
                            'path': str(d),
                            'description': desc,
                        })

        return models

    def load_model(self, model_identifier: str) -> Any:
        """Set model selection mode (auto or explicit single-model override)."""
        norm = model_identifier.lower().strip()
        if norm in ('auto', 'satellite-auto', 'satellite_auto'):
            self.current_model_mode = 'auto'
            self.current_model = SatelliteAutoForecaster(registry=self.registry)
        elif norm in MODEL_ADAPTER_CLASSES:
            self.current_model_mode = norm
            self.current_model = SatelliteAutoForecaster(registry=self.registry, explicit_model=norm)
        else:
            self.current_model_mode = norm
            self.current_model = NeuroNavModel.load(norm)
        return self.current_model

    def load_dataset(self, file_path: str | Path) -> Tuple[int, List[str]]:
        """Load and validate telemetry data file, returning row count and satellite list."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        df = pd.read_csv(path)
        self.current_dataset_path = str(path)
        self.current_data = df

        sat_col = None
        for cand in ('Satellite_ID', 'satellite_id', 'sat_id', 'PRN', 'prn', 'Satellite'):
            if cand in df.columns:
                sat_col = cand
                break

        if sat_col:
            sats = sorted(df[sat_col].astype(str).dropna().unique().tolist())
        else:
            sats = ['SAT_GLOBAL']

        return len(df), sats

    # -------------------------------------------------------------------------
    # Calibration & Model Memory API
    # -------------------------------------------------------------------------
    def evaluate_all_models(
        self,
        train_data: pd.DataFrame | str | Path,
        test_data: pd.DataFrame | str | Path,
        target_satellite_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stage 1: Evaluate all eligible models per satellite without future leakage."""
        cal_res = self.calibration_engine.calibrate(
            train_df_or_path=train_data,
            test_df_or_path=test_data,
            generate_report=True,
            target_satellite_id=target_satellite_id,
        )
        self.last_calibration_result = cal_res
        return cal_res

    def calibrate_satellite_models(
        self,
        train_data: pd.DataFrame | str | Path,
        test_data: pd.DataFrame | str | Path,
        target_satellite_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Alias for evaluate_all_models."""
        return self.evaluate_all_models(train_data, test_data, target_satellite_id=target_satellite_id)

    def select_model_for_satellite(
        self,
        satellite_id: str,
        selection: str,
    ) -> Dict[str, Any]:
        """Set manual model override for satellite in persistent memory."""
        return self.registry.set_manual_override(satellite_id=satellite_id, model_id=selection)

    def reset_model_for_satellite(self, satellite_id: str) -> Optional[Dict[str, Any]]:
        """Reset satellite selection back to automatic (highest candidate score)."""
        return self.registry.reset_to_automatic(satellite_id=satellite_id)

    def clear_registry(self) -> None:
        """Clear all satellite model entries and reset state to empty."""
        self.registry.clear()
        self.last_calibration_result = None
        self.current_forecast = None

    def get_model_for_satellite(self, satellite_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored metadata for given satellite."""
        return self.registry.get_satellite_entry(satellite_id)

    def get_all_satellite_memories(self) -> Dict[str, Any]:
        """Retrieve all satellite persistent model selections."""
        return self.registry._data.get("satellites", {})

    # -------------------------------------------------------------------------
    # Stage 2: Satellite-Aware Forecasting
    # -------------------------------------------------------------------------
    def predict_with_satellite_models(
        self,
        data: Optional[pd.DataFrame | str | Path] = None,
        satellite_id: Optional[str] = None,
        explicit_model: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run heterogeneous satellite forecasting routing each satellite to its memory."""
        target_data = data if data is not None else self.current_data
        if target_data is None:
            raise RuntimeError("No dataset available for prediction. Load dataset first.")

        chosen_explicit = explicit_model
        if chosen_explicit is None and self.current_model_mode != 'auto':
            chosen_explicit = self.current_model_mode

        forecast_df = predict_satellite_heterogeneous(
            data=target_data,
            registry=self.registry,
            satellite_id=satellite_id,
            explicit_model=chosen_explicit,
        )
        self.current_forecast = forecast_df
        return forecast_df

    def run_forecast(self, satellite_id: Optional[str] = None) -> pd.DataFrame:
        """Execute forecast inference for current dataset and loaded model mode."""
        if self.current_data is None:
            raise RuntimeError("No dataset is currently loaded. Call load_dataset() first.")

        if self.current_model_mode == 'auto':
            return self.predict_with_satellite_models(data=self.current_data, satellite_id=satellite_id)

        # Explicit single model mode
        if hasattr(self.current_model, 'predict'):
            res = self.current_model.predict(self.current_data, satellite_id=satellite_id)
            forecast_df = res if isinstance(res, pd.DataFrame) else pd.concat([r.to_dataframe() for r in res], ignore_index=True)
            self.current_forecast = forecast_df
            return forecast_df

        return self.predict_with_satellite_models(
            data=self.current_data,
            satellite_id=satellite_id,
            explicit_model=self.current_model_mode,
        )

    def export_forecast(self, export_path: str | Path) -> Path:
        """Export latest forecast predictions to CSV."""
        if self.current_forecast is None:
            raise RuntimeError("No forecast available to export. Run forecast first.")
        out = Path(export_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.current_forecast.to_csv(out, index=False)
        return out

    def generate_plots(
        self,
        output_dir: Optional[str | Path] = None,
        satellite_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate forecast component and 3D orbit error figures."""
        if self.current_forecast is None:
            raise RuntimeError("No forecast available to plot. Run forecast first.")

        sat = satellite_id
        if sat is None:
            for cand in ('Satellite_ID', 'satellite_id'):
                if cand in self.current_forecast.columns:
                    sat = str(self.current_forecast[cand].iloc[0])
                    break

        comp_path = Path(output_dir) / f'forecast_components_{sat}.png' if output_dir else None
        orbit_path = Path(output_dir) / f'orbit_3d_error_{sat}.png' if output_dir else None

        fig_comp = plot_forecast_components(self.current_forecast, satellite_id=sat, output_path=comp_path)
        fig_orbit = plot_3d_orbit_error(self.current_forecast, satellite_id=sat, output_path=orbit_path)

        return {
            'forecast_components': fig_comp,
            'orbit_3d': fig_orbit,
        }
