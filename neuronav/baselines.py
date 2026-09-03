from __future__ import annotations
import warnings
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import numpy as np

def _normalise_history(history: np.ndarray) -> Tuple[np.ndarray, int]:
    values = np.asarray(history, dtype=np.float64)
    original_ndim = values.ndim
    if original_ndim == 1:
        values = values[None, :, None]
    elif original_ndim == 2:
        values = values[None, :, :]
    elif original_ndim != 3:
        raise ValueError(f'history must have shape (time,), (time, target), or (series, time, target); got {values.shape}')
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError(f'history cannot have an empty dimension; got {values.shape}')
    return (values, original_ndim)

def _restore_forecast_shape(forecast: np.ndarray, original_ndim: int) -> np.ndarray:
    if original_ndim == 1:
        return forecast[0, :, 0]
    if original_ndim == 2:
        return forecast[0]
    return forecast

def _validate_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or int(horizon) < 1:
        raise ValueError('horizon must be a positive integer')
    return int(horizon)

def _last_finite(values: np.ndarray) -> np.ndarray:
    result = np.full((values.shape[0], values.shape[2]), np.nan, dtype=np.float64)
    for series_index in range(values.shape[0]):
        for target_index in range(values.shape[2]):
            valid_indices = np.flatnonzero(np.isfinite(values[series_index, :, target_index]))
            if valid_indices.size:
                result[series_index, target_index] = values[series_index, valid_indices[-1], target_index]
    return result

def zero_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    values, original_ndim = _normalise_history(history)
    steps = _validate_horizon(horizon)
    forecast = np.zeros((values.shape[0], steps, values.shape[2]), dtype=np.float64)
    return _restore_forecast_shape(forecast, original_ndim)

def persistence_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    values, original_ndim = _normalise_history(history)
    steps = _validate_horizon(horizon)
    forecast = np.repeat(_last_finite(values)[:, None, :], steps, axis=1)
    return _restore_forecast_shape(forecast, original_ndim)

def seasonal_forecast(history: np.ndarray, horizon: int, season_length: int=96) -> np.ndarray:
    values, original_ndim = _normalise_history(history)
    steps = _validate_horizon(horizon)
    if isinstance(season_length, bool) or not isinstance(season_length, (int, np.integer)) or int(season_length) < 1:
        raise ValueError('season_length must be a positive integer')
    season_length = int(season_length)
    if values.shape[1] < season_length:
        raise ValueError(f'seasonal baseline needs at least {season_length} history steps; got {values.shape[1]}')
    final_season = values[:, -season_length:, :]
    indices = np.arange(steps) % season_length
    forecast = final_season[:, indices, :]
    fallback = np.repeat(_last_finite(values)[:, None, :], steps, axis=1)
    forecast = np.where(np.isfinite(forecast), forecast, fallback)
    return _restore_forecast_shape(forecast, original_ndim)

def drift_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    values, original_ndim = _normalise_history(history)
    steps = _validate_horizon(horizon)
    if values.shape[1] < 2:
        raise ValueError('drift baseline needs at least two history steps')
    slope = np.full((values.shape[0], values.shape[2]), np.nan, dtype=np.float64)
    anchor = _last_finite(values)
    for series_index in range(values.shape[0]):
        for target_index in range(values.shape[2]):
            valid_indices = np.flatnonzero(np.isfinite(values[series_index, :, target_index]))
            if valid_indices.size == 1:
                slope[series_index, target_index] = 0.0
            elif valid_indices.size > 1:
                first_index, last_index = (valid_indices[0], valid_indices[-1])
                slope[series_index, target_index] = (values[series_index, last_index, target_index] - values[series_index, first_index, target_index]) / float(last_index - first_index)
    lead = np.arange(1, steps + 1, dtype=np.float64)[None, :, None]
    forecast = anchor[:, None, :] + lead * slope[:, None, :]
    return _restore_forecast_shape(forecast, original_ndim)

def gaussian_process_forecast(history: np.ndarray, horizon: int, *, random_state: int=42) -> np.ndarray:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from sklearn.exceptions import ConvergenceWarning
    values, original_ndim = _normalise_history(history)
    steps = _validate_horizon(horizon)
    forecast = np.full((values.shape[0], steps, values.shape[2]), np.nan, dtype=np.float64)
    scale = float(max(values.shape[1] - 1, 1))
    future_time = (np.arange(values.shape[1], values.shape[1] + steps) / scale)[:, None]
    for series_index in range(values.shape[0]):
        for target_index in range(values.shape[2]):
            target = values[series_index, :, target_index]
            valid = np.isfinite(target)
            if valid.sum() < 3 or np.ptp(target[valid]) == 0:
                forecast[series_index, :, target_index] = _last_finite(values[series_index:series_index + 1, :, target_index:target_index + 1])[0, 0]
                continue
            training_time = (np.flatnonzero(valid) / scale)[:, None]
            kernel = ConstantKernel(1.0, (0.001, 1000.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.01, 100.0)) + WhiteKernel(noise_level=0.001, noise_level_bounds=(1e-08, 10.0))
            model = GaussianProcessRegressor(kernel=kernel, alpha=1e-08, normalize_y=True, n_restarts_optimizer=0, random_state=random_state)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                model.fit(training_time, target[valid])
            forecast[series_index, :, target_index] = model.predict(future_time)
    return _restore_forecast_shape(forecast, original_ndim)
forecast_zero = zero_forecast
forecast_persistence = persistence_forecast
forecast_seasonal = seasonal_forecast
forecast_drift = drift_forecast
DEFAULT_BASELINE_NAMES = ('zero', 'persistence', 'seasonal', 'drift')
BASELINE_NAMES = (*DEFAULT_BASELINE_NAMES, 'gaussian_process')

def generate_baseline_forecasts(history: np.ndarray, horizon: int, baselines: Sequence[str]=DEFAULT_BASELINE_NAMES, season_length: int=96) -> Dict[str, np.ndarray]:
    generators = {'zero': lambda: zero_forecast(history, horizon), 'persistence': lambda: persistence_forecast(history, horizon), 'seasonal': lambda: seasonal_forecast(history, horizon, season_length), 'drift': lambda: drift_forecast(history, horizon), 'gaussian_process': lambda: gaussian_process_forecast(history, horizon)}
    requested = [str(name).lower() for name in baselines]
    unknown = sorted(set(requested) - set(generators))
    if unknown:
        raise ValueError(f'unknown baselines {unknown}; supported baselines are {list(BASELINE_NAMES)}')
    if len(set(requested)) != len(requested):
        raise ValueError('baseline names must be unique')
    return {name: generators[name]() for name in requested}

def evaluate_baselines(history: np.ndarray, actual: np.ndarray, target_cols: Sequence[str], baselines: Sequence[str]=DEFAULT_BASELINE_NAMES, season_length: int=96, horizons: Optional[Mapping[str, int]]=None, satellite_ids: Optional[Sequence[Any]]=None, constellations: Optional[Sequence[Any]]=None, valid_mask: Optional[np.ndarray]=None) -> Dict[str, Dict[str, Any]]:
    from neuronav.evaluation import evaluate_forecasts
    actual_values = np.asarray(actual, dtype=np.float64)
    if actual_values.ndim not in (1, 2, 3):
        raise ValueError('actual has an unsupported shape')
    forecast_horizon = actual_values.shape[-2] if actual_values.ndim >= 2 else actual_values.shape[0]
    forecasts = generate_baseline_forecasts(history, forecast_horizon, baselines=baselines, season_length=season_length)
    return {name: evaluate_forecasts(actual_values, forecast, target_cols, horizons=horizons, satellite_ids=satellite_ids, constellations=constellations, valid_mask=valid_mask) for name, forecast in forecasts.items()}
