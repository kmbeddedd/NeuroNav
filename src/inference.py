"""Production Inference Engine for NeuroNav GNSS Orbit and Clock Error Forecasting.

Provides a clean, high-level, decoupled inference interface for Python/Tkinter GUI
and runtime services without exposing PyTorch internals, training scripts, or scalers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.artifacts import scaler_from_state
from src.config import (
    DEFAULT_SEED,
    EXPECTED_CADENCE_MINUTES,
    FORECAST_HORIZON,
    SEQ_LEN,
    SP3_CLOCK_SENTINEL_SECONDS,
    TARGET_COLS_4,
    resolve_device,
)
from src.data import engineer_features
from src.models.bilstm import BiLSTMGRUPyTorchModel
from src.models.transformer import GNSSForecaster
from src.models.adapters import get_adapter_by_id, MODEL_ADAPTER_CLASSES
from src.satellite_registry import SatelliteModelRegistry


@dataclass
class PredictionResult:
    """Structured container for GNSS orbit and clock forecast outputs."""
    satellite_id: str
    forecast_times: pd.DatetimeIndex
    forecast_steps: np.ndarray
    pred_error_x: np.ndarray
    pred_error_y: np.ndarray
    pred_error_z: np.ndarray
    pred_error_clock: np.ndarray
    pred_3d_orbit_error: np.ndarray
    model_name: str
    uncertainty: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    actual_error_x: Optional[np.ndarray] = None
    actual_error_y: Optional[np.ndarray] = None
    actual_error_z: Optional[np.ndarray] = None
    actual_error_clock: Optional[np.ndarray] = None
    actual_3d_orbit_error: Optional[np.ndarray] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert result into a tabular pandas DataFrame conforming to output contract."""
        data = {
            'forecast_step': self.forecast_steps,
            'forecast_time': self.forecast_times,
            'Satellite_ID': self.satellite_id,
            'pred_Error_X': self.pred_error_x,
            'pred_Error_Y': self.pred_error_y,
            'pred_Error_Z': self.pred_error_z,
            'pred_Error_Clock': self.pred_error_clock,
            'pred_3D_Orbit_Error': self.pred_3d_orbit_error,
        }
        if self.uncertainty:
            for target, (low, high) in self.uncertainty.items():
                data[f'pred_{target}_low'] = low
                data[f'pred_{target}_high'] = high

        if self.actual_error_x is not None:
            data['Error_X'] = self.actual_error_x
            data['Error_Y'] = self.actual_error_y
            data['Error_Z'] = self.actual_error_z
            data['Error_Clock'] = self.actual_error_clock
            if self.actual_3d_orbit_error is not None:
                data['3D_Orbit_Error'] = self.actual_3d_orbit_error
            data['residual_Error_X'] = self.pred_error_x - self.actual_error_x
            data['residual_Error_Y'] = self.pred_error_y - self.actual_error_y
            data['residual_Error_Z'] = self.pred_error_z - self.actual_error_z
            data['residual_Error_Clock'] = self.pred_error_clock - self.actual_error_clock
            if self.actual_3d_orbit_error is not None:
                data['residual_3D_Orbit_Error'] = self.pred_3d_orbit_error - self.actual_3d_orbit_error

        return pd.DataFrame(data)


class NeuroNavModel:
    """High-level deployment model wrapper managing inference, scalers, and validation."""

    def __init__(
        self,
        model: torch.nn.Module,
        feature_scaler: Any,
        target_scaler: Any,
        feature_cols: List[str],
        target_cols: List[str],
        satellite_classes: List[str],
        seq_len: int = SEQ_LEN,
        forecast_horizon: int = FORECAST_HORIZON,
        model_name: str = 'NeuroNavForecaster',
        model_type: str = 'bilstm',
        uncertainty_supported: bool = False,
        device: Optional[torch.device] = None,
        manifest: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.satellite_classes = list(satellite_classes)
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.model_name = model_name
        self.model_type = model_type
        self.uncertainty_supported = uncertainty_supported
        self.device = device or torch.device('cpu')
        self.manifest = manifest or {}
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load(
        cls,
        model_path_or_dir: Union[str, Path] = 'auto',
        device: str = 'auto',
    ) -> Union["NeuroNavModel", "SatelliteAutoForecaster"]:
        """Load a deployable NeuroNav model from directory, bundle path, or shortcut name."""
        norm = str(model_path_or_dir).lower()
        if norm in ('auto', 'satellite-auto', 'satellite_auto'):
            return SatelliteAutoForecaster()

        if norm in MODEL_ADAPTER_CLASSES and norm not in ('bilstm_gru', 'transformer'):
            return SatelliteAutoForecaster(explicit_model=norm)

        target = Path(model_path_or_dir)

        # Handle shortcuts
        if norm in ('bilstm', 'bilstm_gru', 'gnss-bilstm-gru'):
            target = Path('models/deploy/bilstm')
        elif norm in ('transformer', 'hybrid_transformer', 'gnss-hybrid-transformer'):
            target = Path('models/deploy/transformer')

        if target.is_dir():
            bundle_path = target / 'model.pt'
            manifest_path = target / 'manifest.json'
        else:
            bundle_path = target
            manifest_path = target.parent / 'manifest.json'

        if not bundle_path.exists():
            raise FileNotFoundError(f"Model bundle not found at: {bundle_path}")

        resolved_device = resolve_device(device)
        bundle = torch.load(bundle_path, map_location=resolved_device, weights_only=False)

        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            except Exception:
                pass

        model_class_name = bundle.get('model_class', '')
        model_config = bundle.get('model_config', {})

        # Restore scalers
        f_scaler = bundle['feature_scaler']
        if isinstance(f_scaler, dict):
            f_scaler = scaler_from_state(f_scaler)

        t_scaler = bundle['target_scaler']
        if isinstance(t_scaler, dict):
            t_scaler = scaler_from_state(t_scaler)

        feature_cols = bundle.get('feature_cols', [])
        target_cols = bundle.get('target_cols', TARGET_COLS_4)
        satellite_classes = bundle.get('satellite_classes', [])

        # Instantiate architecture
        if 'transformer' in model_class_name.lower() or 'hybrid' in model_class_name.lower() or 'GNSSForecaster' in model_class_name:
            model = GNSSForecaster(**model_config)
            model_type = 'transformer'
            uncertainty_supported = True
            default_name = 'GNSS-Hybrid-Transformer'
        else:
            model = BiLSTMGRUPyTorchModel(**model_config)
            model_type = 'bilstm'
            uncertainty_supported = False
            default_name = 'GNSS-BiLSTM-GRU'

        model.load_state_dict(bundle['model_state_dict'])
        model.to(resolved_device)
        model.eval()

        return cls(
            model=model,
            feature_scaler=f_scaler,
            target_scaler=t_scaler,
            feature_cols=feature_cols,
            target_cols=target_cols,
            satellite_classes=satellite_classes,
            seq_len=model_config.get('seq_len', SEQ_LEN),
            forecast_horizon=model_config.get('forecast_horizon', FORECAST_HORIZON),
            model_name=manifest.get('model_name', default_name),
            model_type=model_type,
            uncertainty_supported=uncertainty_supported,
            device=resolved_device,
            manifest=manifest,
        )

    def validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate input DataFrame against the inference contract."""
        data = df.copy()

        # Handle timestamp column alias
        time_col = None
        for candidate in ('Timestamp', 'utc_time', 'time', 'datetime'):
            if candidate in data.columns:
                time_col = candidate
                break

        if time_col is None:
            raise ValueError("Input data must contain a 'Timestamp' or 'utc_time' column.")

        data['Timestamp'] = pd.to_datetime(data[time_col], errors='coerce')
        if data['Timestamp'].isna().any():
            invalid_count = int(data['Timestamp'].isna().sum())
            raise ValueError(f"Input contains {invalid_count} invalid or unparsable timestamp entries.")

        # Handle satellite column alias
        sat_col = None
        for candidate in ('Satellite_ID', 'satellite_id', 'sat_id', 'prn', 'Satellite'):
            if candidate in data.columns:
                sat_col = candidate
                break
        if sat_col is None:
            raise ValueError("Input data must contain a 'Satellite_ID' column.")
        data['Satellite_ID'] = data[sat_col].astype(str)

        # Target alias mapping
        target_aliases = {
            'Error_X': ('Error_X', 'x_error_m', 'x_error (m)', 'error_x'),
            'Error_Y': ('Error_Y', 'y_error_m', 'y_error (m)', 'error_y'),
            'Error_Z': ('Error_Z', 'z_error_m', 'z_error (m)', 'error_z'),
            'Error_Clock': ('Error_Clock', 'clock_error_m', 'satclockerror (m)', 'satclockerror', 'error_clock'),
        }
        for std_col, candidates in target_aliases.items():
            if std_col not in data.columns:
                for c in candidates:
                    if c in data.columns:
                        data[std_col] = data[c]
                        break

        # Check required target error columns
        required_targets = ['Error_X', 'Error_Y', 'Error_Z', 'Error_Clock']
        missing_targets = [col for col in required_targets if col not in data.columns]
        if missing_targets:
            raise ValueError(f"Input data is missing required target columns: {missing_targets}")

        # Ensure numeric types
        for col in required_targets:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        return data

    def predict(
        self,
        data: Union[pd.DataFrame, str, Path],
        satellite_id: Optional[str] = None,
        return_dataframe: bool = True,
    ) -> Union[pd.DataFrame, List[PredictionResult]]:
        """Run GNSS orbit and clock error forecasting.

        Args:
            data: DataFrame or path to CSV file containing telemetry history.
            satellite_id: Optional specific satellite to predict (e.g. 'G01').
                          If None, predicts all satellites present in data.
            return_dataframe: If True, returns a consolidated pandas DataFrame.
                              If False, returns a list of PredictionResult dataclasses.

        Returns:
            pd.DataFrame or List[PredictionResult] containing multihorizon forecasts.
        """
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        validated_df = self.validate_input_data(df)

        available_sats = sorted(validated_df['Satellite_ID'].unique())
        target_sats = [satellite_id] if satellite_id is not None else available_sats

        results: List[PredictionResult] = []

        for sat in target_sats:
            sat_df = validated_df[validated_df['Satellite_ID'] == sat].sort_values('Timestamp').reset_index(drop=True)
            if len(sat_df) < self.seq_len:
                raise ValueError(
                    f"Satellite {sat} has {len(sat_df)} records, but model requires a minimum "
                    f"lookback history of {self.seq_len} contiguous steps (cadence: {EXPECTED_CADENCE_MINUTES}m)."
                )

            # Take the most recent sequence window for prediction
            history_df = sat_df.iloc[-self.seq_len:].copy().reset_index(drop=True)

            # Engineer features
            engineered = engineer_features(history_df)

            # Ensure all required features are present and finite
            for feat in self.feature_cols:
                if feat not in engineered.columns:
                    engineered[feat] = 0.0
                else:
                    engineered[feat] = engineered[feat].bfill().ffill().fillna(0.0)

            feat_df = engineered[self.feature_cols].copy()
            # Replace any residual non-finite values
            feat_df = feat_df.fillna(0.0)

            scaled_features = self.feature_scaler.transform(feat_df)
            scaled_features = np.nan_to_num(scaled_features, nan=0.0)

            x_tensor = torch.as_tensor(scaled_features, dtype=torch.float32, device=self.device).unsqueeze(0)

            last_timestamp = history_df['Timestamp'].iloc[-1]
            cadence = pd.Timedelta(minutes=EXPECTED_CADENCE_MINUTES)
            forecast_times = pd.date_range(
                start=last_timestamp + cadence,
                periods=self.forecast_horizon,
                freq=cadence,
            )
            forecast_steps = np.arange(1, self.forecast_horizon + 1)

            with torch.no_grad():
                if self.model_type == 'transformer':
                    sat_idx_val = self.satellite_classes.index(sat) if sat in self.satellite_classes else 0
                    sat_tensor = torch.tensor([sat_idx_val], dtype=torch.long, device=self.device)
                    mu, sigma, _, _ = self.model(x_tensor, sat_tensor)
                    raw_pred = mu.squeeze(0).cpu().numpy()
                    raw_sigma = sigma.squeeze(0).cpu().numpy()
                else:
                    raw_pred = self.model(x_tensor).squeeze(0).cpu().numpy()
                    raw_sigma = None

            # Inverse scale targets
            phys_pred = self.target_scaler.inverse_transform(raw_pred)

            pred_x = phys_pred[:, 0]
            pred_y = phys_pred[:, 1]
            pred_z = phys_pred[:, 2]
            pred_clock = phys_pred[:, 3]
            pred_3d = np.sqrt(pred_x ** 2 + pred_y ** 2 + pred_z ** 2)

            uncertainty_dict = None
            if raw_sigma is not None and self.uncertainty_supported:
                # Target scaler std scaling for uncertainty bounds (90% CI: z=1.645)
                scale_factors = self.target_scaler.scale_ if hasattr(self.target_scaler, 'scale_') else np.ones(4)
                phys_sigma = raw_sigma * scale_factors
                z_score = 1.645
                uncertainty_dict = {
                    'Error_X': (pred_x - z_score * phys_sigma[:, 0], pred_x + z_score * phys_sigma[:, 0]),
                    'Error_Y': (pred_y - z_score * phys_sigma[:, 1], pred_y + z_score * phys_sigma[:, 1]),
                    'Error_Z': (pred_z - z_score * phys_sigma[:, 2], pred_z + z_score * phys_sigma[:, 2]),
                    'Error_Clock': (pred_clock - z_score * phys_sigma[:, 3], pred_clock + z_score * phys_sigma[:, 3]),
                }

            # Check if ground truth exists for evaluation comparison
            actual_x = actual_y = actual_z = actual_clock = actual_3d = None
            future_mask = (sat_df['Timestamp'] >= forecast_times[0]) & (sat_df['Timestamp'] <= forecast_times[-1])
            future_df = sat_df[future_mask]
            if len(future_df) == self.forecast_horizon:
                actual_x = future_df['Error_X'].to_numpy(dtype=float)
                actual_y = future_df['Error_Y'].to_numpy(dtype=float)
                actual_z = future_df['Error_Z'].to_numpy(dtype=float)
                actual_clock = future_df['Error_Clock'].to_numpy(dtype=float)
                actual_3d = np.sqrt(actual_x ** 2 + actual_y ** 2 + actual_z ** 2)

            res = PredictionResult(
                satellite_id=sat,
                forecast_times=forecast_times,
                forecast_steps=forecast_steps,
                pred_error_x=pred_x,
                pred_error_y=pred_y,
                pred_error_z=pred_z,
                pred_error_clock=pred_clock,
                pred_3d_orbit_error=pred_3d,
                model_name=self.model_name,
                uncertainty=uncertainty_dict,
                actual_error_x=actual_x,
                actual_error_y=actual_y,
                actual_error_z=actual_z,
                actual_error_clock=actual_clock,
                actual_3d_orbit_error=actual_3d,
            )
            results.append(res)

        if return_dataframe:
            return pd.concat([r.to_dataframe() for r in results], ignore_index=True)
        return results


def predict_satellite_heterogeneous(
    data: Union[pd.DataFrame, str, Path],
    registry: Optional[SatelliteModelRegistry] = None,
    satellite_id: Optional[str] = None,
    horizon_steps: int = FORECAST_HORIZON,
    fallback_model: Optional[str] = None,
    explicit_model: Optional[str] = None,
) -> pd.DataFrame:
    """Execute satellite-aware heterogeneous forecasting across detected satellites.
    
    Each satellite is routed to its own stored model from persistent memory.
    Never uses a single global default.
    """
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    # Detect satellite column
    sat_col = None
    for candidate in ('Satellite_ID', 'satellite_id', 'sat_id', 'PRN', 'prn', 'Satellite'):
        if candidate in df.columns:
            sat_col = candidate
            break

    if satellite_id and sat_col is None:
        df['Satellite_ID'] = str(satellite_id).strip()
        sat_col = 'Satellite_ID'
        all_sats = [str(satellite_id).strip()]
    elif sat_col is not None:
        all_sats = sorted([str(s) for s in df[sat_col].dropna().unique()])
    else:
        all_sats = ['SAT_GLOBAL']
        df['Satellite_ID'] = 'SAT_GLOBAL'
        sat_col = 'Satellite_ID'

    target_sats = [satellite_id] if satellite_id is not None else all_sats
    reg = registry or SatelliteModelRegistry()

    forecast_frames: List[pd.DataFrame] = []

    for sat in target_sats:
        sat_df = df[df[sat_col].astype(str) == sat].copy()
        if len(sat_df) == 0:
            continue

        # Model resolution for this satellite
        if explicit_model and str(explicit_model).lower() not in ('auto', 'satellite-auto', 'satellite_auto', 'none'):
            chosen_model = str(explicit_model).lower()
            selection_mode = "explicit"
            adapter = get_adapter_by_id(chosen_model)
        else:
            entry = reg.get_satellite_entry(sat)
            if entry is None or not entry.get("selected_model"):
                if fallback_model:
                    chosen_model = fallback_model
                    selection_mode = "fallback"
                    adapter = get_adapter_by_id(chosen_model)
                else:
                    raise ValueError(
                        f"Satellite '{sat}' has NO_SELECTION in model memory. "
                        f"Calibrate models in Stage 1 first or specify an explicit model."
                    )
            else:
                chosen_model = entry["selected_model"]
                selection_mode = entry.get("selection_mode", "automatic")
                is_avail, avail_msg = reg.verify_model_availability(sat)
                if not is_avail:
                    raise RuntimeError(
                        f"Stored model '{chosen_model}' for satellite '{sat}' is unavailable: {avail_msg}"
                    )
                adapter = get_adapter_by_id(chosen_model)

        # Eligibility check
        is_elig, reason = adapter.check_eligibility(sat_df, sat)
        if not is_elig:
            raise ValueError(
                f"Satellite '{sat}' data is ineligible for model '{chosen_model}': {reason}"
            )

        # Fit model on historical 7-day data
        adapter.fit(sat_df, sat)

        # Predict future horizon
        pred_df = adapter.predict(
            train_df=sat_df,
            horizon_steps=horizon_steps,
            satellite_id=sat,
        )

        # Contract tagging
        pred_df['Satellite_ID'] = sat
        pred_df['satellite_id'] = sat
        pred_df['model_used'] = chosen_model
        pred_df['model_version'] = adapter.version
        pred_df['selection_mode'] = selection_mode

        time_val = pred_df['utc_time'] if 'utc_time' in pred_df.columns else (
            pred_df['forecast_time'] if 'forecast_time' in pred_df.columns else pred_df.get('Timestamp')
        )
        pred_df['timestamp'] = time_val

        # Ensure standard prediction columns exist
        pred_df['predicted_X'] = pred_df['pred_Error_X']
        pred_df['predicted_Y'] = pred_df['pred_Error_Y']
        pred_df['predicted_Z'] = pred_df['pred_Error_Z']
        pred_df['predicted_Clock'] = pred_df['pred_Error_Clock']

        forecast_frames.append(pred_df)

    if not forecast_frames:
        raise ValueError("No forecast predictions could be generated for the specified data.")

    return pd.concat(forecast_frames, ignore_index=True)


class SatelliteAutoForecaster:
    """High-level forecaster that routes each satellite to its own stored model from memory."""

    def __init__(
        self,
        registry: Optional[SatelliteModelRegistry] = None,
        explicit_model: Optional[str] = None,
    ):
        self.registry = registry or SatelliteModelRegistry()
        self.explicit_model = explicit_model
        self.model_name = f"SatelliteAutoForecaster({explicit_model or 'per-satellite-memory'})"
        self.model_type = "satellite_auto"
        self.seq_len = SEQ_LEN
        self.forecast_horizon = FORECAST_HORIZON
        self.uncertainty_supported = False

    def predict(
        self,
        data: Union[pd.DataFrame, str, Path],
        satellite_id: Optional[str] = None,
        horizon_steps: int = FORECAST_HORIZON,
        return_dataframe: bool = True,
    ) -> pd.DataFrame:
        """Run satellite-specific forecasting using persistent model memory."""
        return predict_satellite_heterogeneous(
            data=data,
            registry=self.registry,
            satellite_id=satellite_id,
            horizon_steps=horizon_steps,
            explicit_model=self.explicit_model,
        )
