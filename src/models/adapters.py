"""Forecast model adapters providing a unified interface across heterogeneous models."""
from __future__ import annotations

import json
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch

from src.baselines import persistence_forecast
from src.config import EXPECTED_CADENCE_MINUTES, FORECAST_HORIZON, SEQ_LEN

TARGETS = ('x_error_m', 'y_error_m', 'z_error_m', 'clock_error_m')
TARGET_COLS_CONTRACT = ('Error_X', 'Error_Y', 'Error_Z', 'Error_Clock')


def _extract_time_features(times: pd.Series | pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """Generate diurnal phase harmonics and polynomial elapsed time features."""
    index = pd.DatetimeIndex(times)
    elapsed_days = (index - origin).total_seconds().to_numpy(dtype=float) / 86400.0
    phase = 2.0 * np.pi * (index.hour.to_numpy() * 3600.0 + index.minute.to_numpy() * 60.0 + index.second.to_numpy()) / 86400.0
    columns = [elapsed_days, elapsed_days ** 2 / 49.0]
    for harmonic in range(1, 7):
        columns.extend((np.sin(harmonic * phase), np.cos(harmonic * phase)))
    return np.column_stack(columns).astype(np.float64)


def _get_query_times(
    train_df: pd.DataFrame,
    horizon_steps: int = FORECAST_HORIZON,
    target_times: Optional[pd.DatetimeIndex | pd.Series] = None,
    cadence_minutes: int = EXPECTED_CADENCE_MINUTES,
) -> pd.DatetimeIndex:
    """Derive future target forecast timestamps."""
    if target_times is not None and len(target_times) > 0:
        return pd.DatetimeIndex(target_times)

    time_col = 'utc_time' if 'utc_time' in train_df.columns else 'Timestamp'
    last_time = pd.to_datetime(train_df[time_col]).max()
    cadence = pd.Timedelta(minutes=cadence_minutes)
    return pd.date_range(start=last_time + cadence, periods=horizon_steps, freq=cadence)


def _standardize_target_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract 4-target matrix (X, Y, Z, Clock) using any available column aliases."""
    arr = np.zeros((len(df), 4), dtype=np.float64)
    target_candidates = [
        ('pred_Error_X', 'predicted_x_error_m', 'predicted_X', 'Error_X', 'x_error_m', 'x_error (m)', 'error_x'),
        ('pred_Error_Y', 'predicted_y_error_m', 'predicted_Y', 'Error_Y', 'y_error_m', 'y_error (m)', 'error_y'),
        ('pred_Error_Z', 'predicted_z_error_m', 'predicted_Z', 'Error_Z', 'z_error_m', 'z_error (m)', 'error_z'),
        ('pred_Error_Clock', 'predicted_clock_error_m', 'predicted_Clock', 'Error_Clock', 'clock_error_m', 'satclockerror (m)', 'satclockerror', 'error_clock'),
    ]
    for i, cands in enumerate(target_candidates):
        found = False
        for c in cands:
            if c in df.columns:
                arr[:, i] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).to_numpy()
                found = True
                break
        if not found:
            arr[:, i] = 0.0
    return arr


def _resolve_satellite_id(train_df: pd.DataFrame, satellite_id: Optional[str] = None) -> str:
    """Safely determine satellite identifier without risking AttributeError on str fallback."""
    if satellite_id:
        return str(satellite_id)
    for col in ('Satellite_ID', 'satellite_id', 'sat_id', 'PRN', 'prn', 'Satellite'):
        if col in train_df.columns and len(train_df[col]) > 0:
            return str(train_df[col].iloc[0])
    return "G01"


# -----------------------------------------------------------------------------
# Base Model Adapter
# -----------------------------------------------------------------------------
class ForecastModelAdapter(ABC):
    """Abstract base class for all satellite orbit & clock error forecasting models."""

    model_id: str
    model_name: str
    version: str
    description: str

    @abstractmethod
    def check_eligibility(self, train_df: pd.DataFrame, satellite_id: str) -> Tuple[bool, str]:
        """Check if satellite historical dataset meets model requirements."""
        pass

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, satellite_id: str) -> None:
        """Fit model strictly on allowed historical 7-day training data (zero future leakage)."""
        pass

    @abstractmethod
    def predict(
        self,
        train_df: pd.DataFrame,
        horizon_steps: int = FORECAST_HORIZON,
        target_times: Optional[pd.DatetimeIndex | pd.Series] = None,
        satellite_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Produce forecast dataframe for given horizon/target timestamps."""
        pass


# -----------------------------------------------------------------------------
# 1. Persistence Adapter
# -----------------------------------------------------------------------------
class PersistenceAdapter(ForecastModelAdapter):
    """Zero-parameter persistence baseline carrying forward last known observation."""

    model_id = "persistence"
    model_name = "Persistence Baseline"
    version = "1.0.0"
    description = "Causal persistence baseline repeating the last observed satellite errors"

    def __init__(self):
        self._last_state: Optional[np.ndarray] = None

    def check_eligibility(self, train_df: pd.DataFrame, satellite_id: str) -> Tuple[bool, str]:
        if len(train_df) < 1:
            return False, "Dataset is empty"
        return True, "Eligible"

    def fit(self, train_df: pd.DataFrame, satellite_id: str) -> None:
        targets = _standardize_target_matrix(train_df)
        self._last_state = targets[-1]

    def predict(
        self,
        train_df: pd.DataFrame,
        horizon_steps: int = FORECAST_HORIZON,
        target_times: Optional[pd.DatetimeIndex | pd.Series] = None,
        satellite_id: Optional[str] = None,
    ) -> pd.DataFrame:
        if self._last_state is None:
            self.fit(train_df, satellite_id or "G01")

        q_times = _get_query_times(train_df, horizon_steps, target_times)
        steps = len(q_times)
        preds = np.repeat(self._last_state[None, :], steps, axis=0)

        sat = _resolve_satellite_id(train_df, satellite_id)
        p3d = np.sqrt(preds[:, 0] ** 2 + preds[:, 1] ** 2 + preds[:, 2] ** 2)

        return pd.DataFrame({
            'forecast_step': np.arange(1, steps + 1),
            'utc_time': q_times,
            'forecast_time': q_times,
            'Satellite_ID': sat,
            'satellite_id': sat,
            'predicted_x_error_m': preds[:, 0],
            'predicted_y_error_m': preds[:, 1],
            'predicted_z_error_m': preds[:, 2],
            'predicted_clock_error_m': preds[:, 3],
            'pred_Error_X': preds[:, 0],
            'pred_Error_Y': preds[:, 1],
            'pred_Error_Z': preds[:, 2],
            'pred_Error_Clock': preds[:, 3],
            'pred_3D_Orbit_Error': p3d,
            'model_used': self.model_id,
            'model_version': self.version,
        })


# -----------------------------------------------------------------------------
# 2. Harmonic Ridge Adapter
# -----------------------------------------------------------------------------
class HarmonicRidgeAdapter(ForecastModelAdapter):
    """Linear Ridge regression fitted on diurnal orbit harmonics and elapsed time."""

    model_id = "harmonic_ridge"
    model_name = "Harmonic Ridge Regression"
    version = "1.0.0"
    description = "Physical diurnal orbit harmonics and polynomial trend with L2 regularisation"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model = None
        self._origin: Optional[pd.Timestamp] = None

    def check_eligibility(self, train_df: pd.DataFrame, satellite_id: str) -> Tuple[bool, str]:
        if len(train_df) < 10:
            return False, f"Insufficient observations ({len(train_df)} < 10)"
        return True, "Eligible"

    def fit(self, train_df: pd.DataFrame, satellite_id: str) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        time_col = 'utc_time' if 'utc_time' in train_df.columns else 'Timestamp'
        times = pd.to_datetime(train_df[time_col])
        self._origin = times.min().floor('D')

        x_train = _extract_time_features(times, self._origin)
        y_train = _standardize_target_matrix(train_df)

        self._model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self._model.fit(x_train, y_train)

    def predict(
        self,
        train_df: pd.DataFrame,
        horizon_steps: int = FORECAST_HORIZON,
        target_times: Optional[pd.DatetimeIndex | pd.Series] = None,
        satellite_id: Optional[str] = None,
    ) -> pd.DataFrame:
        if self._model is None:
            self.fit(train_df, satellite_id or "G01")

        q_times = _get_query_times(train_df, horizon_steps, target_times)
        x_query = _extract_time_features(q_times, self._origin)
        preds = self._model.predict(x_query)

        steps = len(q_times)
        sat = _resolve_satellite_id(train_df, satellite_id)
        p3d = np.sqrt(preds[:, 0] ** 2 + preds[:, 1] ** 2 + preds[:, 2] ** 2)

        return pd.DataFrame({
            'forecast_step': np.arange(1, steps + 1),
            'utc_time': q_times,
            'forecast_time': q_times,
            'Satellite_ID': sat,
            'satellite_id': sat,
            'predicted_x_error_m': preds[:, 0],
            'predicted_y_error_m': preds[:, 1],
            'predicted_z_error_m': preds[:, 2],
            'predicted_clock_error_m': preds[:, 3],
            'pred_Error_X': preds[:, 0],
            'pred_Error_Y': preds[:, 1],
            'pred_Error_Z': preds[:, 2],
            'pred_Error_Clock': preds[:, 3],
            'pred_3D_Orbit_Error': p3d,
            'model_used': self.model_id,
            'model_version': self.version,
        })


# -----------------------------------------------------------------------------
# 3. Random Forest Adapter
# -----------------------------------------------------------------------------
class RandomForestAdapter(ForecastModelAdapter):
    """Random Forest regressor capturing nonlinear dynamics and ephemeris residuals."""

    model_id = "random_forest"
    model_name = "Random Forest Regressor"
    version = "1.0.0"
    description = "Ensemble decision tree regressor fitted on orbit harmonic and lag features"

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = None
        self._origin: Optional[pd.Timestamp] = None

    def check_eligibility(self, train_df: pd.DataFrame, satellite_id: str) -> Tuple[bool, str]:
        if len(train_df) < 15:
            return False, f"Insufficient observations ({len(train_df)} < 15)"
        return True, "Eligible"

    def fit(self, train_df: pd.DataFrame, satellite_id: str) -> None:
        from sklearn.ensemble import RandomForestRegressor

        time_col = 'utc_time' if 'utc_time' in train_df.columns else 'Timestamp'
        times = pd.to_datetime(train_df[time_col])
        self._origin = times.min().floor('D')

        x_train = _extract_time_features(times, self._origin)
        y_train = _standardize_target_matrix(train_df)

        self._model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        self._model.fit(x_train, y_train)

    def predict(
        self,
        train_df: pd.DataFrame,
        horizon_steps: int = FORECAST_HORIZON,
        target_times: Optional[pd.DatetimeIndex | pd.Series] = None,
        satellite_id: Optional[str] = None,
    ) -> pd.DataFrame:
        if self._model is None:
            self.fit(train_df, satellite_id or "G01")

        q_times = _get_query_times(train_df, horizon_steps, target_times)
        x_query = _extract_time_features(q_times, self._origin)
        preds = self._model.predict(x_query)

        steps = len(q_times)
        sat = _resolve_satellite_id(train_df, satellite_id)
        p3d = np.sqrt(preds[:, 0] ** 2 + preds[:, 1] ** 2 + preds[:, 2] ** 2)

        return pd.DataFrame({
            'forecast_step': np.arange(1, steps + 1),
            'utc_time': q_times,
            'forecast_time': q_times,
            'Satellite_ID': sat,
            'satellite_id': sat,
            'predicted_x_error_m': preds[:, 0],
            'predicted_y_error_m': preds[:, 1],
            'predicted_z_error_m': preds[:, 2],
            'predicted_clock_error_m': preds[:, 3],
            'pred_Error_X': preds[:, 0],
            'pred_Error_Y': preds[:, 1],
            'pred_Error_Z': preds[:, 2],
            'pred_Error_Clock': preds[:, 3],
            'pred_3D_Orbit_Error': p3d,
            'model_used': self.model_id,
            'model_version': self.version,
        })


# -----------------------------------------------------------------------------
# 4. Gaussian Process Adapter
# -----------------------------------------------------------------------------
class GaussianProcessAdapter(ForecastModelAdapter):
    """Gaussian Process with RBF and White noise kernels for non-parametric forecasting."""

    model_id = "gaussian_process"
    model_name = "Gaussian Process Regressor"
    version = "1.0.0"
    description = "Non-parametric kernel regression with epistemic uncertainty estimation"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._models = []
        self._origin: Optional[pd.Timestamp] = None

    def check_eligibility(self, train_df: pd.DataFrame, satellite_id: str) -> Tuple[bool, str]:
        if len(train_df) < 10:
            return False, f"Insufficient observations ({len(train_df)} < 10)"
        return True, "Eligible"

    def fit(self, train_df: pd.DataFrame, satellite_id: str) -> None:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

        time_col = 'utc_time' if 'utc_time' in train_df.columns else 'Timestamp'
        times = pd.to_datetime(train_df[time_col])
        self._origin = times.min().floor('D')

        sample_df = train_df if len(train_df) <= 200 else train_df.iloc[-200:].copy()
        s_times = pd.to_datetime(sample_df[time_col])
        days = (s_times - self._origin).total_seconds().to_numpy(dtype=float).reshape(-1, 1) / 86400.0
        targets = _standardize_target_matrix(sample_df)

        self._models = []
        for i in range(4):
            kernel = ConstantKernel(1.0, (0.01, 100.0)) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(noise_level=0.1)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=0, random_state=self.random_state)
            gp.fit(days, targets[:, i])
            self._models.append(gp)

    def predict(
        self,
        train_df: pd.DataFrame,
        horizon_steps: int = FORECAST_HORIZON,
        target_times: Optional[pd.DatetimeIndex | pd.Series] = None,
        satellite_id: Optional[str] = None,
    ) -> pd.DataFrame:
        if not self._models:
            self.fit(train_df, satellite_id or "G01")

        q_times = _get_query_times(train_df, horizon_steps, target_times)
        days_query = (q_times - self._origin).total_seconds().to_numpy(dtype=float).reshape(-1, 1) / 86400.0

        preds = np.zeros((len(q_times), 4), dtype=np.float64)
        for i in range(4):
            preds[:, i] = self._models[i].predict(days_query)

        steps = len(q_times)
        sat = _resolve_satellite_id(train_df, satellite_id)
        p3d = np.sqrt(preds[:, 0] ** 2 + preds[:, 1] ** 2 + preds[:, 2] ** 2)

        return pd.DataFrame({
            'forecast_step': np.arange(1, steps + 1),
            'utc_time': q_times,
            'forecast_time': q_times,
            'Satellite_ID': sat,
            'satellite_id': sat,
            'predicted_x_error_m': preds[:, 0],
            'predicted_y_error_m': preds[:, 1],
            'predicted_z_error_m': preds[:, 2],
            'predicted_clock_error_m': preds[:, 3],
            'pred_Error_X': preds[:, 0],
            'pred_Error_Y': preds[:, 1],
            'pred_Error_Z': preds[:, 2],
            'pred_Error_Clock': preds[:, 3],
            'pred_3D_Orbit_Error': p3d,
            'model_used': self.model_id,
            'model_version': self.version,
        })


# -----------------------------------------------------------------------------
# 5. BiLSTM-GRU Adapter (Deep Neural Network)
# -----------------------------------------------------------------------------
class BiLSTMGRUAdapter(ForecastModelAdapter):
    """Production BiLSTM-GRU neural network bundle loaded from models/deploy/bilstm."""

    model_id = "bilstm_gru"
    model_name = "GNSS-BiLSTM-GRU"
    version = "1.0.0"
    description = "Bidirectional LSTM + GRU recurrent neural network for multi-horizon ephemeris forecasting"

    def __init__(self, model_dir: Optional[str | Path] = None):
        if model_dir is None:
            self.model_dir = Path(__file__).resolve().parent.parent.parent / 'models' / 'deploy' / 'bilstm'
        else:
            self.model_dir = Path(model_dir)
        self._model = None

    def check_eligibility(self, train_df: pd.DataFrame, satellite_id: str) -> Tuple[bool, str]:
        if not (self.model_dir / 'model.pt').exists():
            return False, f"Model weights missing: {self.model_dir / 'model.pt'}"
        if len(train_df) < SEQ_LEN:
            return False, f"Insufficient history length ({len(train_df)} < {SEQ_LEN} required epochs)"
        return True, "Eligible"

    def fit(self, train_df: pd.DataFrame, satellite_id: str) -> None:
        from src.inference import NeuroNavModel
        if self._model is None:
            self._model = NeuroNavModel.load(self.model_dir)

    def predict(
        self,
        train_df: pd.DataFrame,
        horizon_steps: int = FORECAST_HORIZON,
        target_times: Optional[pd.DatetimeIndex | pd.Series] = None,
        satellite_id: Optional[str] = None,
    ) -> pd.DataFrame:
        if self._model is None:
            self.fit(train_df, satellite_id or "G01")

        sat = _resolve_satellite_id(train_df, satellite_id)
        pred_df = self._model.predict(train_df, satellite_id=sat, return_dataframe=True)

        if 'forecast_time' in pred_df.columns:
            pred_df['utc_time'] = pd.to_datetime(pred_df['forecast_time'])

        if target_times is not None and len(target_times) > 0:
            target_dt = pd.to_datetime(target_times)
            aligned = pd.merge_asof(
                pd.DataFrame({'utc_time': target_dt}).sort_values('utc_time'),
                pred_df.sort_values('utc_time'),
                on='utc_time',
                direction='nearest',
            )
            pred_df = aligned

        pred_df['Satellite_ID'] = sat
        pred_df['satellite_id'] = sat
        pred_df['model_used'] = self.model_id
        pred_df['model_version'] = self.version
        pred_df['predicted_x_error_m'] = pred_df.get('pred_Error_X', 0.0)
        pred_df['predicted_y_error_m'] = pred_df.get('pred_Error_Y', 0.0)
        pred_df['predicted_z_error_m'] = pred_df.get('pred_Error_Z', 0.0)
        pred_df['predicted_clock_error_m'] = pred_df.get('pred_Error_Clock', 0.0)
        if 'pred_3D_Orbit_Error' not in pred_df.columns:
            pred_df['pred_3D_Orbit_Error'] = np.sqrt(
                pred_df['predicted_x_error_m'] ** 2 + pred_df['predicted_y_error_m'] ** 2 + pred_df['predicted_z_error_m'] ** 2
            )
        return pred_df


# -----------------------------------------------------------------------------
# 6. Hybrid Transformer Adapter
# -----------------------------------------------------------------------------
class TransformerAdapter(ForecastModelAdapter):
    """Production GNSS-Hybrid-Transformer bundle loaded from models/deploy/transformer."""

    model_id = "transformer"
    model_name = "GNSS-Hybrid-Transformer"
    version = "1.0.0"
    description = "Hybrid Transformer with temporal attention, spatial embeddings, and uncertainty bounds"

    def __init__(self, model_dir: Optional[str | Path] = None):
        if model_dir is None:
            self.model_dir = Path(__file__).resolve().parent.parent.parent / 'models' / 'deploy' / 'transformer'
        else:
            self.model_dir = Path(model_dir)
        self._model = None

    def check_eligibility(self, train_df: pd.DataFrame, satellite_id: str) -> Tuple[bool, str]:
        if not (self.model_dir / 'model.pt').exists():
            return False, f"Model weights missing: {self.model_dir / 'model.pt'}"
        if len(train_df) < SEQ_LEN:
            return False, f"Insufficient history length ({len(train_df)} < {SEQ_LEN} required epochs)"
        return True, "Eligible"

    def fit(self, train_df: pd.DataFrame, satellite_id: str) -> None:
        from src.inference import NeuroNavModel
        if self._model is None:
            self._model = NeuroNavModel.load(self.model_dir)

    def predict(
        self,
        train_df: pd.DataFrame,
        horizon_steps: int = FORECAST_HORIZON,
        target_times: Optional[pd.DatetimeIndex | pd.Series] = None,
        satellite_id: Optional[str] = None,
    ) -> pd.DataFrame:
        if self._model is None:
            self.fit(train_df, satellite_id or "G01")

        sat = _resolve_satellite_id(train_df, satellite_id)
        pred_df = self._model.predict(train_df, satellite_id=sat, return_dataframe=True)

        if 'forecast_time' in pred_df.columns:
            pred_df['utc_time'] = pd.to_datetime(pred_df['forecast_time'])

        if target_times is not None and len(target_times) > 0:
            target_dt = pd.to_datetime(target_times)
            aligned = pd.merge_asof(
                pd.DataFrame({'utc_time': target_dt}).sort_values('utc_time'),
                pred_df.sort_values('utc_time'),
                on='utc_time',
                direction='nearest',
            )
            pred_df = aligned

        pred_df['Satellite_ID'] = sat
        pred_df['satellite_id'] = sat
        pred_df['model_used'] = self.model_id
        pred_df['model_version'] = self.version
        pred_df['predicted_x_error_m'] = pred_df.get('pred_Error_X', 0.0)
        pred_df['predicted_y_error_m'] = pred_df.get('pred_Error_Y', 0.0)
        pred_df['predicted_z_error_m'] = pred_df.get('pred_Error_Z', 0.0)
        pred_df['predicted_clock_error_m'] = pred_df.get('pred_Error_Clock', 0.0)
        if 'pred_3D_Orbit_Error' not in pred_df.columns:
            pred_df['pred_3D_Orbit_Error'] = np.sqrt(
                pred_df['predicted_x_error_m'] ** 2 + pred_df['predicted_y_error_m'] ** 2 + pred_df['predicted_z_error_m'] ** 2
            )
        return pred_df


# -----------------------------------------------------------------------------
# 7. GEO Gated Mixture-of-Experts Adapter
# -----------------------------------------------------------------------------
class GEOGatedMoEAdapter(ForecastModelAdapter):
    """Causal regime-aware gated Mixture-of-Experts model for excursion and normal states."""

    model_id = "geo_gated_moe"
    model_name = "GEO Gated Mixture-of-Experts"
    version = "1.0.0"
    description = "Causal regime-gated mixture of experts with bidirectional GRU temporal encoder"

    def __init__(self, checkpoint_path: Optional[str | Path] = None):
        if checkpoint_path is None:
            self.checkpoint_path = Path(__file__).resolve().parent.parent.parent / 'research' / 'ps08' / 'models' / 'geo_gated_moe_day8.pt'
        else:
            self.checkpoint_path = Path(checkpoint_path)
        self._ckpt = None
        self._net = None
        self._base = None

    def check_eligibility(self, train_df: pd.DataFrame, satellite_id: str) -> Tuple[bool, str]:
        if not self.checkpoint_path.exists():
            return False, f"MoE checkpoint missing: {self.checkpoint_path}"
        if len(train_df) < 10:
            return False, f"Insufficient observations ({len(train_df)} < 10)"
        return True, "Eligible"

    def fit(self, train_df: pd.DataFrame, satellite_id: str) -> None:
        from scripts.benchmark.benchmark_ps08 import GEOGatedMoEModel, _fit_causal_baseline

        if self._ckpt is None:
            self._ckpt = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
            self._net = GEOGatedMoEModel(history_dim=19, query_dim=13, num_series=3)
            self._net.load_state_dict(self._ckpt['state_dict'])
            self._net.eval()

        time_col = 'utc_time' if 'utc_time' in train_df.columns else 'Timestamp'
        times = pd.to_datetime(train_df[time_col])
        origin = times.min().floor('D')
        self._origin = origin
        self._base = _fit_causal_baseline(train_df, origin)

    def predict(
        self,
        train_df: pd.DataFrame,
        horizon_steps: int = FORECAST_HORIZON,
        target_times: Optional[pd.DatetimeIndex | pd.Series] = None,
        satellite_id: Optional[str] = None,
    ) -> pd.DataFrame:
        from scripts.benchmark.benchmark_ps08 import (
            _compute_regime_probability,
            _physical_history_tensor,
            _physical_query_features,
            time_features,
        )

        if self._net is None:
            self.fit(train_df, satellite_id or "G01")

        q_times = _get_query_times(train_df, horizon_steps, target_times)
        sat = _resolve_satellite_id(train_df, satellite_id)
        series_name = 'GEO' if 'GEO' in sat.upper() else ('MEO-2' if '2' in sat else 'MEO-1')

        c_d, s_d = self._ckpt['delta_scalers'].get(series_name, self._ckpt['delta_scalers']['GEO'])
        detector = self._ckpt['detectors'].get(series_name, self._ckpt['detectors']['GEO'])
        cutoff = len(train_df) - 1

        time_col = 'utc_time' if 'utc_time' in train_df.columns else 'Timestamp'
        last_train_time = pd.to_datetime(train_df[time_col]).iloc[cutoff]

        hists, qs, sids, b_preds = [], [], [], []
        for q_t in q_times:
            hist, meta = _physical_history_tensor(train_df, cutoff, q_t, self._origin, self._base, c_d, s_d)
            rms = float(np.sqrt(np.mean(meta['orbit_norms'][-4:] ** 2)))
            flips = float(np.mean(np.diff(np.sign(meta['vals'][:, 0])) != 0)) if len(meta['vals']) > 1 else 0.0
            r_prob = _compute_regime_probability(meta['orbit_norms'], detector)
            q_feat = _physical_query_features(q_t, last_train_time, self._origin, r_prob, rms, flips)
            x_q = time_features(pd.Series([q_t]), self._origin)
            b_q = self._base.predict(x_q)[0]

            hists.append(hist)
            qs.append(q_feat)
            sids.append(0 if series_name == 'GEO' else (1 if '1' in series_name else 2))
            b_preds.append(b_q)

        with torch.no_grad():
            d_pred, _, _ = self._net(
                torch.as_tensor(np.array(hists), dtype=torch.float32),
                torch.as_tensor(np.array(qs), dtype=torch.float32),
                torch.as_tensor(np.array(sids), dtype=torch.long),
            )
            phys_delta = d_pred.numpy() * s_d + c_d

        preds = np.array(b_preds) + phys_delta
        steps = len(q_times)
        p3d = np.sqrt(preds[:, 0] ** 2 + preds[:, 1] ** 2 + preds[:, 2] ** 2)

        return pd.DataFrame({
            'forecast_step': np.arange(1, steps + 1),
            'utc_time': q_times,
            'forecast_time': q_times,
            'Satellite_ID': sat,
            'satellite_id': sat,
            'predicted_x_error_m': preds[:, 0],
            'predicted_y_error_m': preds[:, 1],
            'predicted_z_error_m': preds[:, 2],
            'predicted_clock_error_m': preds[:, 3],
            'pred_Error_X': preds[:, 0],
            'pred_Error_Y': preds[:, 1],
            'pred_Error_Z': preds[:, 2],
            'pred_Error_Clock': preds[:, 3],
            'pred_3D_Orbit_Error': p3d,
            'model_used': self.model_id,
            'model_version': self.version,
        })


# -----------------------------------------------------------------------------
# Adapter Registry & Discovery
# -----------------------------------------------------------------------------
MODEL_ADAPTER_CLASSES: Dict[str, Type[ForecastModelAdapter]] = {
    "persistence": PersistenceAdapter,
    "harmonic_ridge": HarmonicRidgeAdapter,
    "random_forest": RandomForestAdapter,
    "gaussian_process": GaussianProcessAdapter,
    "bilstm_gru": BiLSTMGRUAdapter,
    "transformer": TransformerAdapter,
    "geo_gated_moe": GEOGatedMoEAdapter,
}


def get_available_model_adapters() -> Dict[str, ForecastModelAdapter]:
    """Instantiate and return dictionary of all registered model adapters."""
    return {model_id: cls() for model_id, cls in MODEL_ADAPTER_CLASSES.items()}


def get_adapter_by_id(model_id: str) -> ForecastModelAdapter:
    """Retrieve an instantiated adapter by model_id with alias resolution."""
    normalized = model_id.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bilstm": "bilstm_gru",
        "gru": "bilstm_gru",
        "gnss_bilstm_gru": "bilstm_gru",
        "hybrid_transformer": "transformer",
        "gnss_hybrid_transformer": "transformer",
        "ridge": "harmonic_ridge",
        "rf": "random_forest",
        "gp": "gaussian_process",
        "moe": "geo_gated_moe",
        "gated_moe": "geo_gated_moe",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in MODEL_ADAPTER_CLASSES:
        raise ValueError(
            f"Unknown model identifier '{model_id}'. Supported models are: {list(MODEL_ADAPTER_CLASSES.keys())}"
        )
    return MODEL_ADAPTER_CLASSES[resolved]()
