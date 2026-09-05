from __future__ import annotations
import argparse
import json
import math
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Callable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
import numpy as np
import pandas as pd
import torch
from src.config import resolve_device
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, ExpSineSquared, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SEED = 42
TARGETS = ('x_error_m', 'y_error_m', 'z_error_m', 'clock_error_m')
TARGET_LABELS = {'x_error_m': 'X error', 'y_error_m': 'Y error', 'z_error_m': 'Z error', 'clock_error_m': 'Clock error'}
PUBLISHED_REFERENCE = {'shapiro_w': 0.981, 'p_value': 0.584, 'hypothesis_result': 0}

@dataclass(frozen=True)
class SeriesSpec:
    name: str
    orbit_class: str
    train_file: str
    test_file: str
SERIES = (SeriesSpec('GEO', 'GEO', 'DATA_GEO_Train.csv', 'DATA_GEO_Test.csv'), SeriesSpec('MEO-1', 'MEO', 'DATA_MEO_Train.csv', 'DATA_MEO_Test.csv'), SeriesSpec('MEO-2', 'MEO', 'DATA_MEO_Train2.csv', 'DATA_MEO_Test2.csv'))

def seed_everything(seed: int=SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)

def load_series(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    normalized = {column: ' '.join(column.strip().lower().split()) for column in frame.columns}
    frame = frame.rename(columns=normalized)
    column_map = {'utc_time': 'utc_time', 'x_error (m)': 'x_error_m', 'y_error (m)': 'y_error_m', 'z_error (m)': 'z_error_m', 'satclockerror (m)': 'clock_error_m'}
    missing = sorted(set(column_map) - set(frame.columns))
    if missing:
        raise ValueError(f'{path} is missing required columns: {missing}')
    frame = frame.rename(columns=column_map)[['utc_time', *TARGETS]]
    frame['utc_time'] = pd.to_datetime(frame['utc_time'], errors='raise')
    for target in TARGETS:
        frame[target] = pd.to_numeric(frame[target], errors='raise')
    frame = frame.sort_values('utc_time').drop_duplicates(subset='utc_time', keep='first')
    if frame['utc_time'].duplicated().any() or not frame['utc_time'].is_monotonic_increasing:
        raise ValueError(f'{path} could not be reduced to a unique chronological series')
    if not np.isfinite(frame[list(TARGETS)].to_numpy(dtype=float)).all():
        raise ValueError(f'{path} contains non-finite target values')
    return frame.reset_index(drop=True)

def load_official_split(data_dir: Path) -> dict[str, dict[str, Any]]:
    datasets: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(SERIES):
        raw_train_rows = len(pd.read_csv(data_dir / spec.train_file))
        raw_test_rows = len(pd.read_csv(data_dir / spec.test_file))
        train = load_series(data_dir / spec.train_file)
        test = load_series(data_dir / spec.test_file)
        if train['utc_time'].max() >= test['utc_time'].min():
            raise ValueError(f'{spec.name}: training timestamps overlap the held-out test period')
        datasets[spec.name] = {'spec': spec, 'series_index': index, 'train': train, 'test': test, 'origin': train['utc_time'].min().floor('D'), 'raw_train_rows': raw_train_rows, 'raw_test_rows': raw_test_rows}
    return datasets

def time_features(times: pd.Series | pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    index = pd.DatetimeIndex(times)
    elapsed_days = (index - origin).total_seconds().to_numpy(dtype=float) / 86400.0
    phase = 2.0 * np.pi * (index.hour.to_numpy() * 3600.0 + index.minute.to_numpy() * 60.0 + index.second.to_numpy()) / 86400.0
    columns = [elapsed_days, elapsed_days ** 2 / 49.0]
    for harmonic in range(1, 7):
        columns.extend((np.sin(harmonic * phase), np.cos(harmonic * phase)))
    return np.column_stack(columns).astype(np.float64)

def predict_persistence(datasets: dict[str, dict[str, Any]], output_dir: Path | None=None) -> dict[str, np.ndarray]:
    predictions = {name: np.repeat(item['train'][list(TARGETS)].iloc[[-1]].to_numpy(), len(item['test']), axis=0) for name, item in datasets.items()}
    if output_dir is not None:
        state = {name: item['train'][list(TARGETS)].iloc[-1].to_dict() for name, item in datasets.items()}
        (output_dir / 'persistence_state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    return predictions

def predict_harmonic_ridge(datasets: dict[str, dict[str, Any]], output_dir: Path | None=None) -> dict[str, np.ndarray]:
    predictions = {}
    fitted_models = {}
    for name, item in datasets.items():
        x_train = time_features(item['train']['utc_time'], item['origin'])
        x_test = time_features(item['test']['utc_time'], item['origin'])
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x_train, item['train'][list(TARGETS)].to_numpy())
        predictions[name] = model.predict(x_test)
        fitted_models[name] = {'model': model, 'origin': item['origin'], 'targets': TARGETS}
    if output_dir is not None:
        dump(fitted_models, output_dir / 'harmonic_ridge_day8.joblib')
    return predictions

def predict_random_forest(datasets: dict[str, dict[str, Any]], output_dir: Path | None=None) -> dict[str, np.ndarray]:
    predictions = {}
    fitted_models = {}
    for name, item in datasets.items():
        x_train = time_features(item['train']['utc_time'], item['origin'])
        x_test = time_features(item['test']['utc_time'], item['origin'])
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.8, random_state=SEED, n_jobs=1)
        model.fit(x_train, item['train'][list(TARGETS)].to_numpy())
        predictions[name] = model.predict(x_test)
        fitted_models[name] = {'model': model, 'origin': item['origin'], 'targets': TARGETS}
    if output_dir is not None:
        dump(fitted_models, output_dir / 'random_forest_day8.joblib')
    return predictions

def predict_gaussian_process(datasets: dict[str, dict[str, Any]], output_dir: Path | None=None) -> dict[str, np.ndarray]:
    predictions = {}
    fitted_models = {}
    for name, item in datasets.items():
        train_days = (item['train']['utc_time'] - item['origin']).dt.total_seconds().to_numpy(dtype=float).reshape(-1, 1) / 86400.0
        test_days = (item['test']['utc_time'] - item['origin']).dt.total_seconds().to_numpy(dtype=float).reshape(-1, 1) / 86400.0
        kernel = ConstantKernel(1.0, (0.01, 100.0)) * (RBF(length_scale=1.0, length_scale_bounds=(0.05, 20.0)) + ExpSineSquared(length_scale=1.0, periodicity=1.0, length_scale_bounds=(0.05, 20.0), periodicity_bounds=(0.8, 1.2))) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-05, 100.0))
        model = MultiOutputRegressor(GaussianProcessRegressor(kernel=kernel, alpha=1e-06, normalize_y=True, n_restarts_optimizer=1, random_state=SEED))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(train_days, item['train'][list(TARGETS)].to_numpy())
        predictions[name] = model.predict(test_days)
        fitted_models[name] = {'model': model, 'origin': item['origin'], 'targets': TARGETS}
    if output_dir is not None:
        dump(fitted_models, output_dir / 'gaussian_process_day8.joblib')
    return predictions


def select_meo_specialists(
    datasets: dict[str, dict[str, Any]],
    validation_days: int = 3,
) -> dict[str, dict[str, Any]]:
    """Select MEO orbit and clock experts using training-only rolling origins."""
    candidates: dict[str, Callable[..., dict[str, np.ndarray]]] = {
        'Persistence': predict_persistence,
        'Harmonic Ridge': predict_harmonic_ridge,
        'Random Forest': predict_random_forest,
        'Gaussian Process': predict_gaussian_process,
    }
    selections: dict[str, dict[str, Any]] = {}
    for series_name, item in datasets.items():
        if item['spec'].orbit_class != 'MEO':
            continue
        frame = item['train']
        days = sorted(pd.DatetimeIndex(frame['utc_time'].dt.floor('D').unique()))
        score_parts = {
            model_name: {'orbit_errors': [], 'clock_errors': []}
            for model_name in candidates
        }
        folds = []
        for validation_day in days[-validation_days:]:
            train = frame[frame['utc_time'] < validation_day].reset_index(drop=True)
            validation = frame[frame['utc_time'].dt.floor('D') == validation_day].reset_index(drop=True)
            if len(train) < 12 or validation.empty:
                continue
            fold_item = {**item, 'train': train, 'test': validation}
            fold = {series_name: fold_item}
            actual = validation[list(TARGETS)].to_numpy(dtype=float)
            folds.append({
                'validation_day': validation_day.isoformat(),
                'training_rows': len(train),
                'validation_rows': len(validation),
            })
            for model_name, predictor in candidates.items():
                predicted = predictor(fold)[series_name]
                residual = predicted - actual
                score_parts[model_name]['orbit_errors'].extend(
                    np.linalg.norm(residual[:, :3], axis=1).tolist()
                )
                score_parts[model_name]['clock_errors'].extend(
                    np.abs(residual[:, 3]).tolist()
                )
        if not folds:
            raise ValueError(f'{series_name}: no training-only MEO validation folds were available')
        scores = {
            model_name: {
                'validation_rows': len(values['orbit_errors']),
                'orbit_vector_mae_m': float(np.mean(values['orbit_errors'])),
                'clock_mae_m': float(np.mean(values['clock_errors'])),
            }
            for model_name, values in score_parts.items()
        }
        selections[series_name] = {
            'orbit_model': min(scores, key=lambda name: scores[name]['orbit_vector_mae_m']),
            'clock_model': min(scores, key=lambda name: scores[name]['clock_mae_m']),
            'folds': folds,
            'scores': scores,
        }
    return selections


def compose_orbit_class_specialist(
    datasets: dict[str, dict[str, Any]],
    all_predictions: dict[str, dict[str, np.ndarray]],
    meo_selections: dict[str, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, str]]]:
    """Route GEO to its gated expert and MEO targets to validated specialists."""
    predictions: dict[str, np.ndarray] = {}
    routing: dict[str, dict[str, str]] = {}
    for series_name, item in datasets.items():
        if item['spec'].orbit_class == 'GEO':
            predictions[series_name] = all_predictions['GEO Gated MoE'][series_name].copy()
            routing[series_name] = {
                'orbit_model': 'GEO Gated MoE',
                'clock_model': 'GEO Gated MoE',
            }
            continue
        selection = meo_selections[series_name]
        orbit_model = selection['orbit_model']
        clock_model = selection['clock_model']
        combined = all_predictions[orbit_model][series_name].copy()
        combined[:, 3] = all_predictions[clock_model][series_name][:, 3]
        predictions[series_name] = combined
        routing[series_name] = {
            'orbit_model': orbit_model,
            'clock_model': clock_model,
        }
    return predictions, routing

def _series_scalers(datasets: dict[str, dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    scalers = {}
    for name, item in datasets.items():
        values = item['train'][list(TARGETS)].to_numpy(dtype=float)
        center = np.median(values, axis=0)
        q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
        scale = np.maximum((q75 - q25) / 1.349, 0.0001)
        scalers[name] = (center, scale)
    return scalers

def _query_features(query: pd.Timestamp, history_end: pd.Timestamp, origin: pd.Timestamp) -> np.ndarray:
    phase = 2.0 * np.pi * (query.hour * 3600.0 + query.minute * 60.0 + query.second) / 86400.0
    lead_days = (query - history_end).total_seconds() / 86400.0
    elapsed_days = (query - origin).total_seconds() / 86400.0
    features = [lead_days / 2.0, elapsed_days / 7.0]
    for harmonic in range(1, 5):
        features.extend((math.sin(harmonic * phase), math.cos(harmonic * phase)))
    return np.asarray(features, dtype=np.float32)

def _history_tensor(frame: pd.DataFrame, cutoff: int, query: pd.Timestamp, center: np.ndarray, scale: np.ndarray, sequence_length: int) -> np.ndarray:
    history = frame.iloc[max(0, cutoff - sequence_length + 1):cutoff + 1]
    values = (history[list(TARGETS)].to_numpy(dtype=float) - center) / scale
    ages = (history['utc_time'] - query).dt.total_seconds().to_numpy(dtype=float).reshape(-1, 1) / (2.0 * 86400.0)
    phase = 2.0 * np.pi * (history['utc_time'].dt.hour.to_numpy() * 3600.0 + history['utc_time'].dt.minute.to_numpy() * 60.0) / 86400.0
    tensor = np.column_stack((values, ages, np.sin(phase), np.cos(phase))).astype(np.float32)
    if len(tensor) < sequence_length:
        padding = np.repeat(tensor[[0]], sequence_length - len(tensor), axis=0)
        tensor = np.vstack((padding, tensor))
    return tensor

def _candidate_cutoffs(times: pd.Series, target_index: int) -> list[int]:
    query = times.iloc[target_index]
    candidates = {target_index - 1}
    for hours in (6, 12, 24, 48):
        eligible = np.flatnonzero((times.iloc[:target_index] <= query - pd.Timedelta(hours=hours)).to_numpy())
        if len(eligible):
            candidates.add(int(eligible[-1]))
    return sorted((index for index in candidates if index >= 2))

def build_neural_examples(datasets: dict[str, dict[str, Any]], scalers: dict[str, tuple[np.ndarray, np.ndarray]], sequence_length: int, validation: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    histories, queries, series_ids, targets = ([], [], [], [])
    for name, item in datasets.items():
        frame = item['train']
        center, scale = scalers[name]
        validation_start = frame['utc_time'].max().floor('D')
        for target_index in range(3, len(frame)):
            query_time = frame['utc_time'].iloc[target_index]
            is_validation_target = query_time >= validation_start
            if is_validation_target != validation:
                continue
            cutoffs = _candidate_cutoffs(frame['utc_time'], target_index)
            if validation:
                cutoffs = [i for i in cutoffs if frame['utc_time'].iloc[i] < validation_start]
            for cutoff in cutoffs:
                histories.append(_history_tensor(frame, cutoff, query_time, center, scale, sequence_length))
                queries.append(_query_features(query_time, frame['utc_time'].iloc[cutoff], item['origin']))
                series_ids.append(item['series_index'])
                targets.append((frame[list(TARGETS)].iloc[target_index].to_numpy(dtype=float) - center) / scale)
    if not histories:
        raise ValueError('No neural training examples could be built from the supplied data')
    return (np.asarray(histories, dtype=np.float32), np.asarray(queries, dtype=np.float32), np.asarray(series_ids, dtype=np.int64), np.asarray(targets, dtype=np.float32))

def build_neural_test_examples(datasets: dict[str, dict[str, Any]], scalers: dict[str, tuple[np.ndarray, np.ndarray]], sequence_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
    histories, queries, series_ids, keys = ([], [], [], [])
    for name, item in datasets.items():
        frame = item['train']
        center, scale = scalers[name]
        cutoff = len(frame) - 1
        for row_index, query_time in enumerate(item['test']['utc_time']):
            histories.append(_history_tensor(frame, cutoff, query_time, center, scale, sequence_length))
            queries.append(_query_features(query_time, frame['utc_time'].iloc[-1], item['origin']))
            series_ids.append(item['series_index'])
            keys.append((name, row_index))
    return (np.asarray(histories, dtype=np.float32), np.asarray(queries, dtype=np.float32), np.asarray(series_ids, dtype=np.int64), keys)

class _SeparateHeads(nn.Module):

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.orbit = nn.Sequential(nn.Linear(input_dim, 48), nn.SiLU(), nn.Linear(48, 3))
        self.clock = nn.Sequential(nn.Linear(input_dim, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.orbit(features), self.clock(features)), dim=-1)

class IrregularBiLSTMGRU(nn.Module):

    def __init__(self, history_dim: int, query_dim: int, num_series: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(history_dim, 32, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(64, 32, batch_first=True)
        self.series_embedding = nn.Embedding(num_series, 8)
        self.heads = _SeparateHeads(32 + query_dim + 8)

    def forward(self, history: torch.Tensor, query: torch.Tensor, series_id: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(history)
        _, hidden = self.gru(encoded)
        features = torch.cat((hidden[-1], query, self.series_embedding(series_id)), dim=-1)
        return self.heads(features)

class IrregularTransformer(nn.Module):

    def __init__(self, history_dim: int, query_dim: int, num_series: int, sequence_length: int) -> None:
        super().__init__()
        width = 48
        self.input_projection = nn.Linear(history_dim, width)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, width))
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=96, dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.series_embedding = nn.Embedding(num_series, 8)
        self.heads = _SeparateHeads(width + query_dim + 8)

    def forward(self, history: torch.Tensor, query: torch.Tensor, series_id: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_projection(history) + self.position)
        features = torch.cat((encoded[:, -1], query, self.series_embedding(series_id)), dim=-1)
        return self.heads(features)

def _loader(arrays: tuple[np.ndarray, ...], shuffle: bool, seed: int) -> DataLoader:
    tensors = [torch.as_tensor(array) for array in arrays]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(*tensors), batch_size=min(128, len(tensors[0])), shuffle=shuffle, generator=generator)

def _fit_neural_model(factory: Callable[[], nn.Module], train_arrays: tuple[np.ndarray, ...], validation_arrays: tuple[np.ndarray, ...], all_arrays: tuple[np.ndarray, ...], device: torch.device, max_epochs: int) -> nn.Module:
    model = factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.0001)
    train_loader = _loader(train_arrays, shuffle=True, seed=SEED)
    validation_loader = _loader(validation_arrays, shuffle=False, seed=SEED)
    best_loss, best_epoch, stale = (float('inf'), 1, 0)
    for epoch in range(1, max_epochs + 1):
        model.train()
        for history, query, series_id, target in train_loader:
            history, query = (history.to(device), query.to(device))
            series_id, target = (series_id.to(device), target.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(history, query, series_id), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for history, query, series_id, target in validation_loader:
                prediction = model(history.to(device), query.to(device), series_id.to(device))
                losses.append(nn.functional.mse_loss(prediction, target.to(device)).item())
        validation_loss = float(np.mean(losses))
        if validation_loss < best_loss - 1e-05:
            best_loss, best_epoch, stale = (validation_loss, epoch, 0)
        else:
            stale += 1
        if stale >= 25:
            break
    seed_everything(SEED)
    final_model = factory().to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=0.0008, weight_decay=0.0001)
    all_loader = _loader(all_arrays, shuffle=True, seed=SEED)
    final_model.train()
    for _ in range(best_epoch):
        for history, query, series_id, target in all_loader:
            history, query = (history.to(device), query.to(device))
            series_id, target = (series_id.to(device), target.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(final_model(history, query, series_id), target)
            loss.backward()
            nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
    final_model.eval()
    final_model.selected_epochs = best_epoch
    final_model.validation_loss = best_loss
    return final_model

def predict_neural(datasets: dict[str, dict[str, Any]], architecture: str, device: torch.device, max_epochs: int, output_dir: Path) -> dict[str, np.ndarray]:
    sequence_length = 16
    scalers = _series_scalers(datasets)
    train = build_neural_examples(datasets, scalers, sequence_length, validation=False)
    validation = build_neural_examples(datasets, scalers, sequence_length, validation=True)
    all_arrays = tuple((np.concatenate((train[i], validation[i]), axis=0) for i in range(4)))
    history_dim, query_dim = (train[0].shape[-1], train[1].shape[-1])
    if architecture == 'bilstm_gru':
        factory = lambda: IrregularBiLSTMGRU(history_dim, query_dim, len(datasets))
    elif architecture == 'transformer':
        factory = lambda: IrregularTransformer(history_dim, query_dim, len(datasets), sequence_length)
    else:
        raise ValueError(f'Unknown neural architecture: {architecture}')
    model = _fit_neural_model(factory, train, validation, all_arrays, device, max_epochs)
    history, query, series_id, keys = build_neural_test_examples(datasets, scalers, sequence_length)
    with torch.no_grad():
        scaled = model(torch.as_tensor(history, device=device), torch.as_tensor(query, device=device), torch.as_tensor(series_id, device=device)).cpu().numpy()
    predictions = {name: np.empty((len(item['test']), len(TARGETS))) for name, item in datasets.items()}
    for prediction, (name, row_index) in zip(scaled, keys):
        center, scale = scalers[name]
        predictions[name][row_index] = prediction * scale + center
    torch.save({'state_dict': model.state_dict(), 'architecture': architecture, 'selected_epochs': model.selected_epochs, 'validation_loss': model.validation_loss, 'sequence_length': sequence_length, 'targets': TARGETS, 'series_scalers': scalers, 'seed': SEED}, output_dir / f'{architecture}_day8.pt')
    return predictions

def _fit_causal_baseline(train_df: pd.DataFrame, origin: pd.Timestamp) -> Any:
    x_train = time_features(train_df['utc_time'], origin)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x_train, train_df[list(TARGETS)].to_numpy())
    return model

def _fit_regime_detector(train_df: pd.DataFrame) -> dict[str, float]:
    vals = train_df[list(TARGETS)].to_numpy(dtype=float)
    orbit_norm = np.sqrt(vals[:, 0] ** 2 + vals[:, 1] ** 2 + vals[:, 2] ** 2)
    s_norm = pd.Series(orbit_norm)
    roll_rms = s_norm.rolling(4, min_periods=1).apply(lambda w: float(np.sqrt(np.mean(w ** 2)))).to_numpy()
    x0 = float(np.quantile(roll_rms, 0.75))
    p90 = float(np.quantile(roll_rms, 0.90))
    scale = float(max((p90 - x0) / 2.0, 1.0))
    return {'x0': x0, 'scale': scale, 'median_norm': float(np.median(orbit_norm)), 'p90_norm': p90}

def _compute_regime_probability(orbit_norm_sub: np.ndarray, detector: dict[str, float]) -> float:
    if len(orbit_norm_sub) == 0:
        return 0.5
    recent = orbit_norm_sub[-4:]
    rms = float(np.sqrt(np.mean(recent ** 2)))
    logit = (rms - detector['x0']) / detector['scale']
    return float(1.0 / (1.0 + np.exp(-logit)))

def _physical_history_tensor(
    frame: pd.DataFrame,
    cutoff_idx: int,
    query_time: pd.Timestamp,
    origin: pd.Timestamp,
    baseline_model: Any,
    center_d: np.ndarray,
    scale_d: np.ndarray,
    max_len: int = 32,
    lookback_hours: float = 24.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    cutoff_time = frame['utc_time'].iloc[cutoff_idx]
    start_time = cutoff_time - pd.Timedelta(hours=lookback_hours)
    sub = frame.iloc[:cutoff_idx + 1]
    eligible = sub[sub['utc_time'] >= start_time]
    if len(eligible) < 2:
        eligible = sub.iloc[-max(2, min(len(sub), 4)):]
    if len(eligible) > max_len:
        eligible = eligible.iloc[-max_len:]

    times = eligible['utc_time'].to_numpy(dtype='datetime64[ns]')
    vals = eligible[list(TARGETS)].to_numpy(dtype=float)
    x_base = time_features(eligible['utc_time'], origin)
    base_vals = baseline_model.predict(x_base)
    deltas = (vals - base_vals - center_d) / scale_d

    time_secs = times.astype('datetime64[ns]').astype(np.int64) / 1e9
    query_sec = pd.Timestamp(query_time).timestamp()

    dt_hours = np.zeros(len(times), dtype=float)
    dt_hours[1:] = (time_secs[1:] - time_secs[:-1]) / 3600.0
    dt_hours[0] = dt_hours[1] if len(dt_hours) > 1 else 0.25

    age_hours = (query_sec - time_secs) / 3600.0

    index = pd.DatetimeIndex(eligible['utc_time'])
    phase = 2.0 * np.pi * (index.hour * 3600.0 + index.minute * 60.0 + index.second).to_numpy(dtype=float) / 86400.0

    diff_vals = np.zeros_like(vals)
    diff_vals[1:] = vals[1:] - vals[:-1]
    diff_norm = diff_vals / scale_d

    norm_3d = np.sqrt(vals[:, 0] ** 2 + vals[:, 1] ** 2 + vals[:, 2] ** 2).reshape(-1, 1) / 20.0

    step_features = np.column_stack((
        deltas,
        vals / 20.0,
        diff_norm,
        norm_3d,
        (dt_hours / 2.0).reshape(-1, 1),
        (age_hours / 24.0).reshape(-1, 1),
        np.sin(phase).reshape(-1, 1),
        np.cos(phase).reshape(-1, 1),
        np.sin(2.0 * phase).reshape(-1, 1),
        np.cos(2.0 * phase).reshape(-1, 1),
    )).astype(np.float32)

    if len(step_features) < max_len:
        padding = np.repeat(step_features[[0]], max_len - len(step_features), axis=0)
        step_features = np.vstack((padding, step_features))

    span_hours = float((cutoff_time - eligible['utc_time'].iloc[0]).total_seconds() / 3600.0)
    meta = {
        'history_span_hours': span_hours,
        'history_rows': len(eligible),
        'last_3d_norm': float(norm_3d[-1, 0] * 20.0),
        'orbit_norms': np.sqrt(vals[:, 0] ** 2 + vals[:, 1] ** 2 + vals[:, 2] ** 2),
        'vals': vals,
    }
    return step_features, meta

def _physical_query_features(
    query_time: pd.Timestamp,
    history_end: pd.Timestamp,
    origin: pd.Timestamp,
    regime_prob: float,
    recent_norm_rms: float,
    sign_flip_rate: float,
) -> np.ndarray:
    lead_hours = (query_time - history_end).total_seconds() / 3600.0
    elapsed_days = (query_time - origin).total_seconds() / 86400.0
    phase = 2.0 * np.pi * (query_time.hour * 3600.0 + query_time.minute * 60.0 + query_time.second) / 86400.0
    harmonics = []
    for k in (1, 2, 3, 4):
        harmonics.extend((math.sin(k * phase), math.cos(k * phase)))
    query_arr = [
        lead_hours / 24.0,
        elapsed_days / 7.0,
        regime_prob,
        recent_norm_rms / 20.0,
        sign_flip_rate,
        *harmonics,
    ]
    return np.asarray(query_arr, dtype=np.float32)

# ---------------------------------------------------------------------------
# GEO Gated Mixture-of-Experts Model
#
# Architecture: Bidirectional GRU encodes 24h physical history → produces
# both a latent state h_enc and a learned gate p_gate ∈ [0,1].
# Two expert heads (Normal + Excursion) produce candidate residual corrections.
# Final delta = (1 - p_gate) * normal_delta + p_gate * excursion_delta
#
# p_gate mathematically gates prediction amplitude — it is NOT merely
# concatenated as a feature. Gradients flow back through the gate to the
# regime encoder.
# ponytail: ~10k params; gate is query-conditional via GRU hidden state.
# ---------------------------------------------------------------------------
class GEOGatedMoEModel(nn.Module):
    def __init__(self, history_dim: int, query_dim: int, num_series: int) -> None:
        super().__init__()
        hidden  = 24
        enc_out = hidden * 2   # bidirectional → 48
        self.gru     = nn.GRU(history_dim, hidden, batch_first=True, bidirectional=True)
        self.ln      = nn.LayerNorm(enc_out)
        self.dropout = nn.Dropout(0.1)
        self.query_proj = nn.Sequential(nn.Linear(query_dim, 24), nn.SiLU(), nn.Linear(24, 24))
        self.series_emb = nn.Embedding(num_series, 8)
        fused = enc_out + 24 + 8   # 80

        # Regime gate → scalar p_gate ∈ (0,1)
        self.gate_head = nn.Sequential(nn.Linear(fused, 24), nn.SiLU(), nn.Linear(24, 1))

        # Normal-regime expert
        self.normal_head = nn.Sequential(nn.Linear(fused, 32), nn.SiLU(), nn.Linear(32, 4))

        # Excursion expert (wider to capture large amplitudes)
        self.excursion_head = nn.Sequential(
            nn.Linear(fused, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 4)
        )

    def forward(
        self,
        history: torch.Tensor,
        query: torch.Tensor,
        series_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: delta_pred (B,4), gate_logit (B,), p_gate (B,1)"""
        out, _  = self.gru(history)
        h_enc   = self.dropout(self.ln(out[:, -1]))
        h_q     = self.query_proj(query)
        h_s     = self.series_emb(series_id)
        fused   = torch.cat((h_enc, h_q, h_s), dim=-1)

        gate_logit = self.gate_head(fused).squeeze(-1)           # (B,)
        p_gate     = torch.sigmoid(gate_logit).unsqueeze(-1)     # (B,1)

        normal_delta    = self.normal_head(fused)                # (B,4)
        excursion_delta = self.excursion_head(fused)             # (B,4)

        # True regime gating: p_gate directly controls prediction amplitude
        delta_pred = (1.0 - p_gate) * normal_delta + p_gate * excursion_delta

        return delta_pred, gate_logit, p_gate


# Keep alias so existing tests (A–H) that import GEORegimeAwareResidualModel still work.
# The alias is structurally compatible only for forward() calls that pass 3 positional args.
GEORegimeAwareResidualModel = GEOGatedMoEModel


def _geo_moe_loss(
    pred_delta: torch.Tensor,
    true_delta: torch.Tensor,
    gate_logit: torch.Tensor,
    true_regime: torch.Tensor,
    s_d: torch.Tensor,
    lambda_amp: float = 0.15,
    lambda_dir: float = 0.05,
    lambda_regime: float = 0.05,
    gamma_focal: float = 2.0,
) -> torch.Tensor:
    """Compound loss: Huber + amplitude + directional (large excursions) + focal BCE."""
    huber      = nn.functional.smooth_l1_loss(pred_delta, true_delta, beta=1.0)
    pred_phys  = pred_delta * s_d
    true_phys  = true_delta * s_d
    pred_norm  = torch.norm(pred_phys[:, :3], dim=-1)
    true_norm  = torch.norm(true_phys[:, :3], dim=-1)
    amp_loss   = nn.functional.smooth_l1_loss(pred_norm, true_norm, beta=1.0)
    large_mask = (true_norm > 1.0).float()
    cos_sim    = nn.functional.cosine_similarity(pred_phys[:, :3] + 1e-8, true_phys[:, :3] + 1e-8, dim=-1)
    dir_loss   = (large_mask * (1.0 - cos_sim)).mean()
    p          = torch.sigmoid(gate_logit)
    bce_raw    = nn.functional.binary_cross_entropy_with_logits(gate_logit, true_regime, reduction='none')
    pt             = torch.where(true_regime > 0.5, p, 1.0 - p)
    focal_weight   = (1.0 - pt) ** gamma_focal
    focal_bce      = (focal_weight * bce_raw).mean()
    return huber + lambda_amp * amp_loss + lambda_dir * dir_loss + lambda_regime * focal_bce


def _build_geo_moe_train_examples(
    train_frame: pd.DataFrame,
    origin: pd.Timestamp,
    series_idx: int,
    baseline: Any,
    detector: dict[str, float],
    c_d: np.ndarray,
    s_d: np.ndarray,
    oversample_high_amp: int = 3,
) -> dict[str, list]:
    """Build training examples strictly from train_frame with oversampling of high-amplitude observations."""
    train_data: dict[str, list] = {'hist': [], 'q': [], 'sid': [], 'target_d': [], 'target_r': []}
    x0 = detector['x0']

    for target_idx in range(3, len(train_frame)):
        query_time = train_frame['utc_time'].iloc[target_idx]
        cutoffs = [target_idx - 1]
        for h in (6, 12, 24, 48):
            el = np.flatnonzero(
                (train_frame['utc_time'].iloc[:target_idx] <= query_time - pd.Timedelta(hours=h)).to_numpy()
            )
            if len(el):
                cutoffs.append(int(el[-1]))
        cutoffs = sorted(set(c for c in cutoffs if c >= 2))

        for c in cutoffs:
            hist, meta = _physical_history_tensor(train_frame, c, query_time, origin, baseline, c_d, s_d)
            rms    = float(np.sqrt(np.mean(meta['orbit_norms'][-4:] ** 2)))
            flips  = float(np.mean(np.diff(np.sign(meta['vals'][:, 0])) != 0)) if len(meta['vals']) > 1 else 0.0
            q_feat = _physical_query_features(
                query_time, train_frame['utc_time'].iloc[c], origin,
                _compute_regime_probability(meta['orbit_norms'], detector), rms, flips,
            )
            x_q    = time_features(pd.Series([query_time]), origin)
            base_q = baseline.predict(x_q)[0]
            true_d = (train_frame[list(TARGETS)].iloc[target_idx].to_numpy(dtype=float) - base_q - c_d) / s_d
            t_norm = float(np.sqrt(np.sum(train_frame[list(TARGETS)[:3]].iloc[target_idx].to_numpy(dtype=float) ** 2)))
            t_regime = float(1.0 / (1.0 + np.exp(-((t_norm - x0) / max(detector['scale'], 0.5)))))

            train_data['hist'].append(hist)
            train_data['q'].append(q_feat)
            train_data['sid'].append(series_idx)
            train_data['target_d'].append(true_d)
            train_data['target_r'].append(t_regime)

            if t_norm > x0:
                for _ in range(oversample_high_amp - 1):
                    train_data['hist'].append(hist)
                    train_data['q'].append(q_feat)
                    train_data['sid'].append(series_idx)
                    train_data['target_d'].append(true_d)
                    train_data['target_r'].append(t_regime)

    return train_data


def _build_geo_moe_val_examples(
    val_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    origin: pd.Timestamp,
    series_idx: int,
    baseline: Any,
    detector: dict[str, float],
    c_d: np.ndarray,
    s_d: np.ndarray,
) -> dict[str, list]:
    """Build validation examples from val_frame, referencing history only from train_frame."""
    val_data: dict[str, list] = {'hist': [], 'q': [], 'sid': [], 'target_d': [], 'target_r': []}
    x0 = detector['x0']
    train_cutoff = len(train_frame) - 1

    for target_idx in range(len(val_frame)):
        query_time = val_frame['utc_time'].iloc[target_idx]
        assert train_frame['utc_time'].iloc[train_cutoff] < query_time, "Validation query must be strictly after train history"

        hist, meta = _physical_history_tensor(train_frame, train_cutoff, query_time, origin, baseline, c_d, s_d)
        rms    = float(np.sqrt(np.mean(meta['orbit_norms'][-4:] ** 2)))
        flips  = float(np.mean(np.diff(np.sign(meta['vals'][:, 0])) != 0)) if len(meta['vals']) > 1 else 0.0
        q_feat = _physical_query_features(
            query_time, train_frame['utc_time'].iloc[train_cutoff], origin,
            _compute_regime_probability(meta['orbit_norms'], detector), rms, flips,
        )
        x_q    = time_features(pd.Series([query_time]), origin)
        base_q = baseline.predict(x_q)[0]
        true_d = (val_frame[list(TARGETS)].iloc[target_idx].to_numpy(dtype=float) - base_q - c_d) / s_d
        t_norm = float(np.sqrt(np.sum(val_frame[list(TARGETS)[:3]].iloc[target_idx].to_numpy(dtype=float) ** 2)))
        t_regime = float(1.0 / (1.0 + np.exp(-((t_norm - x0) / max(detector['scale'], 0.5)))))

        val_data['hist'].append(hist)
        val_data['q'].append(q_feat)
        val_data['sid'].append(series_idx)
        val_data['target_d'].append(true_d)
        val_data['target_r'].append(t_regime)

    return val_data


@dataclass
class GeoFoldState:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    prediction_start: pd.Timestamp
    prediction_end: pd.Timestamp
    baseline: Any
    center_d: np.ndarray
    scale_d: np.ndarray
    detector: dict[str, float]
    model: Optional[GEOGatedMoEModel] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_geo_fold(
    train_frame: pd.DataFrame,
    origin: pd.Timestamp,
    series_idx: int = 0,
    pred_start: Optional[pd.Timestamp] = None,
    pred_end: Optional[pd.Timestamp] = None,
    num_series: int = 1,
    device: Optional[torch.device] = None,
    epochs: int = 25,
    seed: int = SEED,
) -> GeoFoldState:
    """Strictly causal fold fitting: fits baseline, scalers, detector, and trains model on train_frame only."""
    train_start = train_frame['utc_time'].min()
    train_end   = train_frame['utc_time'].max()
    if pred_start is not None:
        assert train_end < pred_start, (
            f"Lineage violation! train_end ({train_end}) >= pred_start ({pred_start})"
        )

    # 1. Baseline fitted on train_frame only
    baseline = _fit_causal_baseline(train_frame, origin)

    # 2. Residual center and scale from train_frame only
    x_tr = time_features(train_frame['utc_time'], origin)
    base_preds = baseline.predict(x_tr)
    deltas = train_frame[list(TARGETS)].to_numpy(dtype=float) - base_preds
    center_d = np.median(deltas, axis=0)
    q25, q75 = np.quantile(deltas, [0.25, 0.75], axis=0)
    scale_d = np.maximum((q75 - q25) / 1.349, 1e-4)

    # 3. Regime detector from train_frame only
    detector = _fit_regime_detector(train_frame)

    # 4. Neural model trained on train_frame only
    model = None
    if device is not None and epochs > 0:
        tr_data = _build_geo_moe_train_examples(
            train_frame, origin, series_idx, baseline, detector, center_d, scale_d, oversample_high_amp=3
        )
        if len(tr_data['hist']) > 0:
            seed_everything(seed)
            h_dim = tr_data['hist'][0].shape[-1]
            q_dim = tr_data['q'][0].shape[-1]
            model = GEOGatedMoEModel(h_dim, q_dim, num_series).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            s_d_tensor = torch.as_tensor(scale_d, dtype=torch.float32, device=device)

            tr_tensors = (
                torch.as_tensor(np.array(tr_data['hist']),     dtype=torch.float32),
                torch.as_tensor(np.array(tr_data['q']),        dtype=torch.float32),
                torch.as_tensor(np.array(tr_data['sid']),      dtype=torch.long),
                torch.as_tensor(np.array(tr_data['target_d']), dtype=torch.float32),
                torch.as_tensor(np.array(tr_data['target_r']), dtype=torch.float32),
            )
            loader = DataLoader(TensorDataset(*tr_tensors), batch_size=32, shuffle=True)
            model.train()
            for _ in range(epochs):
                for h, q, sid, td, tr in loader:
                    h, q   = h.to(device), q.to(device)
                    sid    = sid.to(device)
                    td, tr = td.to(device), tr.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    pred_d, gate_logit, _ = model(h, q, sid)
                    loss = _geo_moe_loss(pred_d, td, gate_logit, tr, s_d_tensor)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            model.eval()

    metadata = {
        'train_rows': len(train_frame),
        'epochs': epochs,
        'x0': float(detector['x0']),
        'scale': float(detector['scale']),
    }
    return GeoFoldState(
        train_start=train_start,
        train_end=train_end,
        prediction_start=pred_start if pred_start is not None else train_end,
        prediction_end=pred_end if pred_end is not None else train_end,
        baseline=baseline,
        center_d=center_d,
        scale_d=scale_d,
        detector=detector,
        model=model,
        metadata=metadata,
    )


def _rolling_backtest_geo(
    frame: pd.DataFrame,
    origin: pd.Timestamp,
    device: torch.device,
    epochs: int = 25,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """True rolling-origin causal backtest within GEO training data only.
    
    Every fold independently fits baseline, scalers, detector, and trains
    a brand new GEOGatedMoEModel from scratch on train_frame only.
    Compares against Persistence and Harmonic Ridge on identical fold boundaries.
    Never uses Day-8 actuals.
    """
    times     = frame['utc_time']
    day_start = times.min().floor('D')
    # Fold definitions:
    # Fold 1: train Days 1-3, predict Day 4
    # Fold 2: train Days 1-4, predict Day 5
    # Fold 3: train Days 1-5, predict Day 6
    # Fold 4: train Days 1-6, predict Day 7
    fold_configs = [
        (1, 3, 4),
        (2, 4, 5),
        (3, 5, 6),
        (4, 6, 7),
    ]
    fold_results: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    all_maes: list[float] = []
    all_rmses: list[float] = []
    all_ws:   list[float] = []

    for fold_num, tr_days, pred_days in fold_configs:
        train_end  = day_start + pd.Timedelta(days=tr_days)
        pred_start = day_start + pd.Timedelta(days=tr_days)
        pred_end   = day_start + pd.Timedelta(days=pred_days)
        train_mask = times < train_end
        pred_mask  = (times >= pred_start) & (times < pred_end)

        if train_mask.sum() < 4 or pred_mask.sum() < 4:
            continue

        train_frame = frame[train_mask].reset_index(drop=True)
        pred_frame  = frame[pred_mask].reset_index(drop=True)

        # Lineage assertion
        fit_end = train_frame['utc_time'].max()
        prediction_start = pred_frame['utc_time'].min()
        assert fit_end < prediction_start, (
            f"Lineage violation in Fold {fold_num}: fit_end ({fit_end}) >= prediction_start ({prediction_start})"
        )

        # 1. Fit fold state & train fresh model from scratch on train_frame only
        fold_state = fit_geo_fold(
            train_frame=train_frame,
            origin=origin,
            series_idx=0,
            pred_start=prediction_start,
            pred_end=pred_frame['utc_time'].max(),
            num_series=1,
            device=device,
            epochs=epochs,
            seed=SEED + fold_num,
        )

        train_cutoff = len(train_frame) - 1

        # 2. Predict on pred_frame
        hists, qs, sids, bases = [], [], [], []
        for _, query_time in enumerate(pred_frame['utc_time']):
            # Strictly causal: max(history_time) < query_time
            assert train_frame['utc_time'].iloc[train_cutoff] < query_time
            hist, meta = _physical_history_tensor(
                train_frame, train_cutoff, query_time, origin,
                fold_state.baseline, fold_state.center_d, fold_state.scale_d
            )
            rms   = float(np.sqrt(np.mean(meta['orbit_norms'][-4:] ** 2)))
            flips = float(np.mean(np.diff(np.sign(meta['vals'][:, 0])) != 0)) if len(meta['vals']) > 1 else 0.0
            q_feat = _physical_query_features(
                query_time, train_frame['utc_time'].iloc[train_cutoff], origin,
                _compute_regime_probability(meta['orbit_norms'], fold_state.detector), rms, flips,
            )
            x_q = time_features(pd.Series([query_time]), origin)
            b_q = fold_state.baseline.predict(x_q)[0]

            hists.append(hist)
            qs.append(q_feat)
            sids.append(0)
            bases.append(b_q)

        # Model inference
        with torch.no_grad():
            pred_d, _, _ = fold_state.model(
                torch.as_tensor(np.array(hists), dtype=torch.float32, device=device),
                torch.as_tensor(np.array(qs),    dtype=torch.float32, device=device),
                torch.as_tensor(np.array(sids),  dtype=torch.long,    device=device),
            )
            delta_phys = pred_d.cpu().numpy() * fold_state.scale_d + fold_state.center_d

        pred_moe   = np.array(bases) + delta_phys
        pred_ridge = np.array(bases)
        pred_persist = np.repeat(train_frame[list(TARGETS)].iloc[[-1]].to_numpy(dtype=float), len(pred_frame), axis=0)
        actual     = pred_frame[list(TARGETS)].to_numpy(dtype=float)

        # Residuals
        res_moe     = pred_moe - actual
        res_ridge   = pred_ridge - actual
        res_persist = pred_persist - actual

        mae_moe     = float(np.mean(np.abs(res_moe)))
        rmse_moe    = float(np.sqrt(np.mean(res_moe ** 2)))
        w_moe       = float(np.mean([stats.shapiro(res_moe[:, i]).statistic for i in range(4)]))

        mae_ridge   = float(np.mean(np.abs(res_ridge)))
        rmse_ridge  = float(np.sqrt(np.mean(res_ridge ** 2)))
        w_ridge     = float(np.mean([stats.shapiro(res_ridge[:, i]).statistic for i in range(4)]))

        mae_persist = float(np.mean(np.abs(res_persist)))
        rmse_persist = float(np.sqrt(np.mean(res_persist ** 2)))
        w_persist   = float(np.mean([stats.shapiro(res_persist[:, i]).statistic for i in range(4)]))

        # Regime breakdown for MoE
        actual_norms = np.sqrt(np.sum(actual[:, :3] ** 2, axis=1))
        high_mask = actual_norms > fold_state.detector['x0']
        high_mae = float(np.mean(np.abs(res_moe[high_mask]))) if np.any(high_mask) else mae_moe
        norm_mae = float(np.mean(np.abs(res_moe[~high_mask]))) if np.any(~high_mask) else mae_moe

        all_maes.append(mae_moe)
        all_rmses.append(rmse_moe)
        all_ws.append(w_moe)

        fold_key = f'fold_{fold_num}'
        fold_results[fold_key] = {
            'train_start': str(train_frame['utc_time'].min()),
            'train_end': str(train_frame['utc_time'].max()),
            'prediction_start': str(pred_frame['utc_time'].min()),
            'prediction_end': str(pred_frame['utc_time'].max()),
            'train_rows': int(train_mask.sum()),
            'pred_rows': int(pred_mask.sum()),
            'best_epoch': epochs,
            'mae_m': mae_moe,
            'rmse_m': rmse_moe,
            'mean_shapiro_w': w_moe,
            'high_regime_mae_m': high_mae,
            'normal_regime_mae_m': norm_mae,
            'persistence_mae_m': mae_persist,
            'persistence_rmse_m': rmse_persist,
            'persistence_shapiro_w': w_persist,
            'harmonic_ridge_mae_m': mae_ridge,
            'harmonic_ridge_rmse_m': rmse_ridge,
            'harmonic_ridge_shapiro_w': w_ridge,
        }

        csv_rows.append({
            'fold': fold_num,
            'train_start': str(train_frame['utc_time'].min()),
            'train_end': str(train_frame['utc_time'].max()),
            'prediction_start': str(pred_frame['utc_time'].min()),
            'prediction_end': str(pred_frame['utc_time'].max()),
            'train_rows': int(train_mask.sum()),
            'prediction_rows': int(pred_mask.sum()),
            'best_epoch': epochs,
            'MAE': mae_moe,
            'RMSE': rmse_moe,
            'Shapiro_W': w_moe,
            'high_regime_MAE': high_mae,
            'normal_regime_MAE': norm_mae,
            'persistence_MAE': mae_persist,
            'persistence_RMSE': rmse_persist,
            'persistence_W': w_persist,
            'ridge_MAE': mae_ridge,
            'ridge_RMSE': rmse_ridge,
            'ridge_W': w_ridge,
        })

    fold_results['summary'] = {
        'mean_rolling_mae_m': float(np.mean(all_maes)) if all_maes else float('nan'),
        'mean_rolling_rmse_m': float(np.mean(all_rmses)) if all_rmses else float('nan'),
        'mean_rolling_w': float(np.mean(all_ws)) if all_ws else float('nan'),
        'std_rolling_mae_m': float(np.std(all_maes, ddof=1)) if len(all_maes) > 1 else 0.0,
        'std_rolling_rmse_m': float(np.std(all_rmses, ddof=1)) if len(all_rmses) > 1 else 0.0,
        'std_rolling_w': float(np.std(all_ws, ddof=1)) if len(all_ws) > 1 else 0.0,
        'folds_completed': len(all_maes),
    }
    return fold_results, csv_rows


def predict_geo_gated_moe(
    datasets: dict[str, dict[str, Any]],
    device: torch.device,
    max_epochs: int,
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[tuple[str, int], dict[str, Any]]]:
    """GEO Gated MoE: dual expert heads gated by learned p_gate from GRU history.
    
    Strictly eliminates validation and backtest leakage:
    1. Early stopping / epoch selection uses a genuinely held-out Day-7 validation set.
       Preprocessing, baseline, and scalers are fitted ONLY on Days 1-6.
    2. After best_epoch is selected, trial model is discarded.
       A fresh baseline, scaler, and detector are fitted on all 7 days, and a fresh
       model is trained from scratch for best_epoch on all 7 days for Day-8 inference.
    3. Rolling backtest independently fits and trains from scratch per fold.
    """
    # -------------------------------------------------------------------------
    # PHASE 1: Validation Split for Early Stopping & Epoch Selection
    # Fit trial state on Days 1-6 only; Day 7 is held-out validation.
    # -------------------------------------------------------------------------
    trial_train_data: dict[str, list] = {'hist': [], 'q': [], 'sid': [], 'target_d': [], 'target_r': []}
    trial_val_data:   dict[str, list] = {'hist': [], 'q': [], 'sid': [], 'target_d': [], 'target_r': []}
    trial_scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name, item in datasets.items():
        frame  = item['train']
        origin = item['origin']
        val_start = frame['utc_time'].max().floor('D')
        train_sub = frame[frame['utc_time'] < val_start].reset_index(drop=True)
        val_sub   = frame[frame['utc_time'] >= val_start].reset_index(drop=True)

        assert train_sub['utc_time'].max() < val_sub['utc_time'].min(), "Lineage error in validation split"

        # Fit trial state strictly on train_sub
        trial_base = _fit_causal_baseline(train_sub, origin)
        trial_det  = _fit_regime_detector(train_sub)
        x_tr_sub   = time_features(train_sub['utc_time'], origin)
        deltas_sub = train_sub[list(TARGETS)].to_numpy(dtype=float) - trial_base.predict(x_tr_sub)
        c_d_sub    = np.median(deltas_sub, axis=0)
        q25_sub, q75_sub = np.quantile(deltas_sub, [0.25, 0.75], axis=0)
        s_d_sub    = np.maximum((q75_sub - q25_sub) / 1.349, 0.0001)
        trial_scalers[name] = (c_d_sub, s_d_sub)

        # Build train examples from train_sub only
        td = _build_geo_moe_train_examples(
            train_sub, origin, item['series_index'],
            trial_base, trial_det, c_d_sub, s_d_sub, oversample_high_amp=3,
        )
        # Build val examples from val_sub referencing train_sub history only
        vd = _build_geo_moe_val_examples(
            val_sub, train_sub, origin, item['series_index'],
            trial_base, trial_det, c_d_sub, s_d_sub,
        )
        for k in trial_train_data:
            trial_train_data[k].extend(td[k])
            trial_val_data[k].extend(vd[k])

    def to_tensors(d: dict[str, list]) -> tuple[torch.Tensor, ...]:
        return (
            torch.as_tensor(np.array(d['hist']),     dtype=torch.float32),
            torch.as_tensor(np.array(d['q']),        dtype=torch.float32),
            torch.as_tensor(np.array(d['sid']),      dtype=torch.long),
            torch.as_tensor(np.array(d['target_d']), dtype=torch.float32),
            torch.as_tensor(np.array(d['target_r']), dtype=torch.float32),
        )

    tr_tensors  = to_tensors(trial_train_data)
    val_tensors = to_tensors(trial_val_data)
    history_dim, query_dim = tr_tensors[0].shape[-1], tr_tensors[1].shape[-1]
    geo_s_d_trial = trial_scalers['GEO'][1]
    s_d_tensor_trial = torch.as_tensor(geo_s_d_trial, dtype=torch.float32, device=device)

    def make_model() -> GEOGatedMoEModel:
        return GEOGatedMoEModel(history_dim, query_dim, len(datasets)).to(device)

    trial_model = make_model()
    trial_opt   = torch.optim.AdamW(trial_model.parameters(), lr=0.001, weight_decay=0.0001)
    tr_loader   = DataLoader(TensorDataset(*tr_tensors),  batch_size=32, shuffle=True)
    val_loader  = DataLoader(TensorDataset(*val_tensors), batch_size=32, shuffle=False)

    best_val_loss = float('inf')
    best_epoch    = 1
    stale         = 0

    for epoch in range(1, max_epochs + 1):
        trial_model.train()
        for h, q, sid, td, tr in tr_loader:
            h, q   = h.to(device), q.to(device)
            sid    = sid.to(device)
            td, tr = td.to(device), tr.to(device)
            trial_opt.zero_grad(set_to_none=True)
            pred_d, gate_logit, _ = trial_model(h, q, sid)
            loss = _geo_moe_loss(pred_d, td, gate_logit, tr, s_d_tensor_trial)
            loss.backward()
            nn.utils.clip_grad_norm_(trial_model.parameters(), 1.0)
            trial_opt.step()

        trial_model.eval()
        v_losses: list[float] = []
        with torch.no_grad():
            for h, q, sid, td, _ in val_loader:
                pred_d, _, _ = trial_model(h.to(device), q.to(device), sid.to(device))
                v_losses.append(
                    nn.functional.smooth_l1_loss(pred_d, td.to(device), beta=1.0).item()
                )
        vl = float(np.mean(v_losses))
        if vl < best_val_loss - 1e-4:
            best_val_loss = vl
            best_epoch    = epoch
            stale         = 0
        else:
            stale += 1
        if stale >= 25:
            break

    # Discard trial_model completely (do not reuse partially trained model)
    del trial_model, trial_opt
    best_epoch = max(best_epoch, 5)

    # -------------------------------------------------------------------------
    # PHASE 2: Fresh Retrain on Full 7-Day Data for best_epoch
    # Fresh baseline, scaler, detector, and neural weights fitted on all 7 days.
    # -------------------------------------------------------------------------
    baselines: dict[str, Any] = {}
    detectors: dict[str, dict[str, float]] = {}
    delta_scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    full_train_data: dict[str, list] = {'hist': [], 'q': [], 'sid': [], 'target_d': [], 'target_r': []}

    for name, item in datasets.items():
        frame  = item['train']
        origin = item['origin']
        base   = _fit_causal_baseline(frame, origin)
        baselines[name]  = base
        detectors[name]  = _fit_regime_detector(frame)
        x_tr             = time_features(frame['utc_time'], origin)
        base_preds       = base.predict(x_tr)
        deltas           = frame[list(TARGETS)].to_numpy(dtype=float) - base_preds
        c_d              = np.median(deltas, axis=0)
        q25, q75         = np.quantile(deltas, [0.25, 0.75], axis=0)
        s_d              = np.maximum((q75 - q25) / 1.349, 0.0001)
        delta_scalers[name] = (c_d, s_d)

        td_full = _build_geo_moe_train_examples(
            frame, origin, item['series_index'],
            base, detectors[name], c_d, s_d, oversample_high_amp=3,
        )
        for k in full_train_data:
            full_train_data[k].extend(td_full[k])

    full_tensors = to_tensors(full_train_data)
    full_loader  = DataLoader(TensorDataset(*full_tensors), batch_size=32, shuffle=True)
    geo_s_d_full = delta_scalers['GEO'][1]
    s_d_tensor_full = torch.as_tensor(geo_s_d_full, dtype=torch.float32, device=device)

    seed_everything(SEED)
    final_model = make_model()
    final_opt   = torch.optim.AdamW(final_model.parameters(), lr=0.001, weight_decay=0.0001)
    final_model.train()
    for _ in range(best_epoch):
        for h, q, sid, td, tr in full_loader:
            h, q   = h.to(device), q.to(device)
            sid    = sid.to(device)
            td, tr = td.to(device), tr.to(device)
            final_opt.zero_grad(set_to_none=True)
            pred_d, gate_logit, _ = final_model(h, q, sid)
            loss = _geo_moe_loss(pred_d, td, gate_logit, tr, s_d_tensor_full)
            loss.backward()
            nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            final_opt.step()
    final_model.eval()

    # -------------------------------------------------------------------------
    # PHASE 3: Independent Rolling Backtest on GEO Training Data Only
    # Every fold is a completely independent causal experiment with fresh training.
    # -------------------------------------------------------------------------
    geo_item = datasets['GEO']
    backtest_results, csv_rows = _rolling_backtest_geo(
        geo_item['train'], geo_item['origin'], device, epochs=best_epoch,
    )
    print(f"  [GEO MoE Backtest] aggregate: MAE={backtest_results['summary']['mean_rolling_mae_m']:.4f}m  "
          f"W={backtest_results['summary']['mean_rolling_w']:.4f}")

    # Save rolling backtest reports to geo_diagnostics directory
    geo_diag_dir = output_dir / 'geo_diagnostics'
    geo_diag_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(geo_diag_dir / 'geo_rolling_backtest.csv', index=False)
    (geo_diag_dir / 'geo_rolling_backtest.json').write_text(
        json.dumps(backtest_results, indent=2), encoding='utf-8'
    )

    # Official Day-8 inference — strictly causal
    predictions: dict[str, np.ndarray] = {}
    diagnostics: dict[tuple[str, int], dict[str, Any]] = {}

    for name, item in datasets.items():
        frame    = item['train']
        test     = item['test']
        origin   = item['origin']
        c_d, s_d = delta_scalers[name]
        cutoff   = len(frame) - 1  # last training observation

        test_hists, test_qs, test_sids, test_bases = [], [], [], []
        meta_list: list[tuple[float, float, float, float]] = []

        for row_idx, query_time in enumerate(test['utc_time']):
            # Each query uses the same training tail cutoff (no future feedback).
            # p_gate varies across queries because query features (lead_time, phase) differ.
            hist, meta = _physical_history_tensor(frame, cutoff, query_time, origin, baselines[name], c_d, s_d)
            rms    = float(np.sqrt(np.mean(meta['orbit_norms'][-4:] ** 2)))
            flips  = float(np.mean(np.diff(np.sign(meta['vals'][:, 0])) != 0)) if len(meta['vals']) > 1 else 0.0
            r_prob = _compute_regime_probability(meta['orbit_norms'], detectors[name])
            q_feat = _physical_query_features(query_time, frame['utc_time'].iloc[cutoff], origin, r_prob, rms, flips)
            x_q    = time_features(pd.Series([query_time]), origin)
            b_q    = baselines[name].predict(x_q)[0]

            test_hists.append(hist)
            test_qs.append(q_feat)
            test_sids.append(item['series_index'])
            test_bases.append(b_q)
            lead_h = (query_time - frame['utc_time'].iloc[cutoff]).total_seconds() / 3600.0
            meta_list.append((r_prob, meta['history_span_hours'], lead_h, meta['last_3d_norm']))

        with torch.no_grad():
            t_h    = torch.as_tensor(np.array(test_hists), dtype=torch.float32, device=device)
            t_q    = torch.as_tensor(np.array(test_qs),   dtype=torch.float32, device=device)
            t_sid  = torch.as_tensor(np.array(test_sids), dtype=torch.long,    device=device)
            pred_d, _, t_gate = final_model(t_h, t_q, t_sid)
            delta_phys = pred_d.cpu().numpy() * s_d + c_d
            gate_vals  = t_gate.cpu().numpy().squeeze(-1)   # p_gate per query

        pred_phys = np.array(test_bases) + delta_phys
        predictions[name] = pred_phys

        for row_idx in range(len(test)):
            b_val = test_bases[row_idx]
            d_val = delta_phys[row_idx]
            r_prob, span_h, lead_h, last_norm = meta_list[row_idx]
            diagnostics[(name, row_idx)] = {
                'baseline_x_error_m':     float(b_val[0]),
                'baseline_y_error_m':     float(b_val[1]),
                'baseline_z_error_m':     float(b_val[2]),
                'baseline_clock_error_m': float(b_val[3]),
                'delta_x_error_m':        float(d_val[0]),
                'delta_y_error_m':        float(d_val[1]),
                'delta_z_error_m':        float(d_val[2]),
                'delta_clock_error_m':    float(d_val[3]),
                # p_gate is query-conditional (varies by lead_time/phase via GRU).
                # It is NOT a live target-value detector; it reflects learned history state.
                'p_gate':             float(gate_vals[row_idx]),
                'regime_probability': r_prob,   # historical causal RMS-based estimate
                'lead_time_hours':    float(lead_h),
                'history_span_hours': float(span_h),
                'last_3d_norm_m':     float(last_norm),
            }

    torch.save(
        {
            'state_dict':    final_model.state_dict(),
            'best_epoch':    best_epoch,
            'best_val_loss': best_val_loss,
            'delta_scalers': delta_scalers,
            'detectors':     detectors,
            'backtest':      backtest_results,
            'targets':       TARGETS,
            'seed':          SEED,
        },
        output_dir / 'geo_gated_moe_day8.pt',
    )
    return predictions, diagnostics

def shapiro_metrics(values: np.ndarray, bootstrap_reps: int = 1000, seed: int = 42) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    result = stats.shapiro(values)
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_w = np.empty(bootstrap_reps, dtype=float)
    for b in range(bootstrap_reps):
        resample = rng.choice(values, size=n, replace=True)
        boot_w[b] = stats.shapiro(resample).statistic
    ci_lower = float(np.percentile(boot_w, 2.5))
    ci_upper = float(np.percentile(boot_w, 97.5))
    return {
        'count': int(n),
        'shapiro_w': float(result.statistic),
        'p_value': float(result.pvalue),
        'hypothesis_result': int(result.pvalue < 0.05),
        'shapiro_w_ci_95': [ci_lower, ci_upper],
        'ci_method': f'percentile_bootstrap_b{bootstrap_reps}',
        'mean': float(np.mean(values)),
        'standard_deviation': float(np.std(values, ddof=1)),
        'mae': float(np.mean(np.abs(values))),
        'rmse': float(np.sqrt(np.mean(values ** 2))),
    }

def evaluate_predictions(datasets: dict[str, dict[str, Any]], predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    residual_frames = []
    per_series = {}
    for name, item in datasets.items():
        actual = item['test'][list(TARGETS)].to_numpy(dtype=float)
        residuals = predictions[name] - actual
        residual_frames.append(residuals)
        orbit_vector_errors = np.linalg.norm(residuals[:, :3], axis=1)
        target_metrics = {TARGETS[i]: shapiro_metrics(residuals[:, i]) for i in range(len(TARGETS))}
        per_series[name] = {
            'rows': len(actual),
            'average_shapiro_w': float(np.mean([m['shapiro_w'] for m in target_metrics.values()])),
            'orbit_vector_mae_m': float(np.mean(orbit_vector_errors)),
            'orbit_vector_rmse_m': float(np.sqrt(np.mean(orbit_vector_errors ** 2))),
            'clock_mae_m': float(np.mean(np.abs(residuals[:, 3]))),
            'per_target': target_metrics,
        }
    residuals = np.vstack(residual_frames)
    orbit_vector_errors = np.linalg.norm(residuals[:, :3], axis=1)
    pooled_target_metrics = {TARGETS[i]: shapiro_metrics(residuals[:, i]) for i in range(len(TARGETS))}
    evaluations = [metrics for series_report in per_series.values() for metrics in series_report['per_target'].values()]
    avg_ci_lower = float(np.mean([m['shapiro_w_ci_95'][0] for m in evaluations]))
    avg_ci_upper = float(np.mean([m['shapiro_w_ci_95'][1] for m in evaluations]))
    return {
        'average_shapiro_w': float(np.mean([m['shapiro_w'] for m in evaluations])),
        'average_shapiro_w_ci_95': [avg_ci_lower, avg_ci_upper],
        'average_p_value': float(np.mean([m['p_value'] for m in evaluations])),
        'rejected_test_count': int(sum((m['hypothesis_result'] for m in evaluations))),
        'normality_test_count': len(evaluations),
        'mean_absolute_bias': float(np.mean([abs(m['mean']) for m in evaluations])),
        'average_residual_std': float(np.mean([m['standard_deviation'] for m in evaluations])),
        'overall_mae_m': float(np.mean(np.abs(residuals))),
        'overall_rmse_m': float(np.sqrt(np.mean(residuals ** 2))),
        'orbit_vector_mae_m': float(np.mean(orbit_vector_errors)),
        'orbit_vector_rmse_m': float(np.sqrt(np.mean(orbit_vector_errors ** 2))),
        'pooled_average_shapiro_w': float(np.mean([m['shapiro_w'] for m in pooled_target_metrics.values()])),
        'pooled_per_target': pooled_target_metrics,
        'per_series': per_series,
    }

def save_qq_plot(datasets: dict[str, dict[str, Any]], predictions: dict[str, np.ndarray], model_name: str, output_path: Path) -> None:
    figure, axes = plt.subplots(len(datasets), len(TARGETS), figsize=(15, 10))
    for row, (series_name, item) in enumerate(datasets.items()):
        residuals = predictions[series_name] - item['test'][list(TARGETS)].to_numpy(dtype=float)
        for column, target in enumerate(TARGETS):
            axis = axes[row, column]
            stats.probplot(residuals[:, column], dist='norm', plot=axis)
            axis.set_title(f'{series_name} · {TARGET_LABELS[target]}')
            axis.grid(alpha=0.2)
    figure.suptitle(f'{model_name}: Day-8 residual Q-Q plots', fontweight='bold')
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(figure)

def save_predictions(
    datasets: dict[str, dict[str, Any]],
    all_predictions: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    diagnostics: dict[str, dict[tuple[str, int], dict[str, Any]]] | None = None,
) -> None:
    rows = []
    diag_keys = [
        'baseline_x_error_m', 'baseline_y_error_m', 'baseline_z_error_m', 'baseline_clock_error_m',
        'delta_x_error_m', 'delta_y_error_m', 'delta_z_error_m', 'delta_clock_error_m',
        'p_gate', 'regime_probability', 'lead_time_hours', 'history_span_hours', 'last_3d_norm_m',
    ]
    for model_name, predictions in all_predictions.items():
        model_diag = diagnostics.get(model_name) if diagnostics is not None else None
        for series_name, item in datasets.items():
            actual = item['test'][list(TARGETS)].to_numpy(dtype=float)
            for row_index, timestamp in enumerate(item['test']['utc_time']):
                row: dict[str, Any] = {'model': model_name, 'series': series_name, 'utc_time': timestamp}
                for target_index, target in enumerate(TARGETS):
                    row[f'actual_{target}'] = actual[row_index, target_index]
                    row[f'predicted_{target}'] = predictions[series_name][row_index, target_index]
                    row[f'residual_{target}'] = predictions[series_name][row_index, target_index] - actual[row_index, target_index]
                if model_diag is not None and (series_name, row_index) in model_diag:
                    d_info = model_diag[(series_name, row_index)]
                    for k in diag_keys:
                        row[k] = d_info.get(k, np.nan)
                rows.append(row)
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    for series_name, item in datasets.items():
        spec = item.get('spec')
        source_stem = Path(spec.test_file if spec is not None else series_name).stem
        frame.loc[frame['series'] == series_name].to_csv(
            output_path.parent / f'{source_stem}_predictions.csv', index=False
        )

def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        '# PS-08 Day-8 Model Benchmark',
        '',
        'Models were trained only on the supplied seven-day files and evaluated at every unique supplied test timestamp. Exact duplicate rows were removed. Test observations were never fed back as model inputs.',
        '',
        '## Official ranking',
        '',
        f"**Winner: {report['winner']}** — highest average Shapiro–Wilk W across the four equally weighted residual parameters.",
        '',
        '| Rank | Model | Avg W | 95% Bootstrap CI | Avg p-value | Rejected tests | MAE (m) | RMSE (m) | GEO W | GEO MAE (m) | GEO RMSE (m) |',
        '|---:|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|'
    ]
    for model in report['ranking']:
        metrics = report['models'][model]
        ci = metrics.get('average_shapiro_w_ci_95', [np.nan, np.nan])
        geo_m = metrics['per_series']['GEO']
        geo_w = geo_m['average_shapiro_w']
        geo_mae = float(np.mean([geo_m['per_target'][t]['mae'] for t in TARGETS]))
        geo_rmse = float(np.mean([geo_m['per_target'][t]['rmse'] for t in TARGETS]))
        ci_str = f'[{ci[0]:.4f}, {ci[1]:.4f}]' if np.isfinite(ci[0]) else 'N/A'
        lines.append(
            f"| {metrics['rank']} | {model} | {metrics['average_shapiro_w']:.6f} | {ci_str} | {metrics['average_p_value']:.6f} | {metrics['rejected_test_count']}/{metrics['normality_test_count']} | {metrics['overall_mae_m']:.4f} | {metrics['overall_rmse_m']:.4f} | {geo_w:.6f} | {geo_mae:.4f} | {geo_rmse:.4f} |"
        )
    lines.extend([
        '',
        'The primary score is the macro-average of 12 per-series/per-target Shapiro-Wilk evaluations (3 series × 4 parameters); this avoids mixing different orbit distributions or weighting GEO by its larger row count.',
        '',
        'The published reference benchmark is W = 0.9810, p = 0.5840, hypothesis result = 0. This is a normality benchmark, not an accuracy threshold.',
        '',
        '## MEO accuracy refinement',
        '',
        f"**MEO accuracy winner: {report['meo_accuracy_winner']}**",
        '',
        '| Model | MEO 3D vector MAE (m) | MEO clock MAE (m) |',
        '|---|---:|---:|',
    ])
    for model, metrics in sorted(
        report['meo_accuracy'].items(),
        key=lambda item: (item[1]['orbit_vector_mae_m'], item[1]['clock_mae_m']),
    ):
        lines.append(
            f"| {model} | {metrics['orbit_vector_mae_m']:.6f} | "
            f"{metrics['clock_mae_m']:.6f} |"
        )
    promotion = report['meo_promotion']
    lines.extend([
        '',
        f"Promotion gate: **{'PASSED' if promotion['passed'] else 'FAILED'}**. "
        f"Relative to {promotion['baseline']}, orbit-vector MAE improved by "
        f"{promotion['relative_orbit_improvement']:.1%} and clock MAE improved by "
        f"{promotion['relative_clock_improvement']:.1%}.",
        '',
        '## Judge criteria captured from `docs/reference/ps08/competition_note.pdf`',
        '',
        '1. Priority 1: average Shapiro–Wilk W over X, Y, Z and clock residuals; higher is better. Report p-values and the α=0.05 decision (0 = fail to reject normality, 1 = reject).',
        '2. Priority 2: residual mean and standard deviation break a Priority-1 tie.',
        '3. Priority 3: Q-Q plots and their visible outliers break any remaining tie.',
        '',
        'See `qq_*.png` for every model and `day8_predictions.csv` for row-level evidence, and `geo_diagnostics/` for GEO-specific validation and excursion diagnostics.'
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def generate_geo_diagnostics(
    datasets: dict[str, dict[str, Any]],
    all_predictions: dict[str, dict[str, np.ndarray]],
    diagnostics: dict[str, dict[tuple[str, int], dict[str, Any]]],
    output_dir: Path,
) -> None:
    geo_diag_dir = output_dir / 'geo_diagnostics'
    geo_diag_dir.mkdir(parents=True, exist_ok=True)

    geo_train = datasets['GEO']['train']
    geo_test = datasets['GEO']['test']

    # 1. geo_train_distribution.json
    dt_train = (geo_train['utc_time'].diff().dt.total_seconds() / 60.0).dropna()
    vals_train = geo_train[list(TARGETS)].to_numpy(dtype=float)
    norm_train = np.sqrt(vals_train[:, 0] ** 2 + vals_train[:, 1] ** 2 + vals_train[:, 2] ** 2)
    train_dist = {
        'total_rows': len(geo_train),
        'span_start': str(geo_train['utc_time'].min()),
        'span_end': str(geo_train['utc_time'].max()),
        'delta_t_minutes': {
            'min': float(dt_train.min()),
            'median': float(dt_train.median()),
            'mean': float(dt_train.mean()),
            'p75': float(dt_train.quantile(0.75)),
            'p90': float(dt_train.quantile(0.90)),
            'max': float(dt_train.max()),
        },
        'orbit_3d_norm': {
            'mean': float(norm_train.mean()),
            'median': float(np.median(norm_train)),
            'p75': float(np.quantile(norm_train, 0.75)),
            'p90': float(np.quantile(norm_train, 0.90)),
            'max': float(norm_train.max()),
            'high_excursion_count_gt_10m': int(np.sum(norm_train > 10.0)),
        },
        'autocorrelation': {
            col: {
                'lag1': float(geo_train[col].autocorr(1)),
                'lag2': float(geo_train[col].autocorr(2)),
                'lag3': float(geo_train[col].autocorr(3)),
            } for col in TARGETS
        },
    }
    (geo_diag_dir / 'geo_train_distribution.json').write_text(json.dumps(train_dist, indent=2), encoding='utf-8')

    # 2. geo_validation_report.json
    val_start = geo_train['utc_time'].max().floor('D')
    val_df = geo_train[geo_train['utc_time'] >= val_start]
    val_report = {
        'validation_start': str(val_start),
        'validation_rows': len(val_df),
        'selected_lookback_hours': 24,
        'max_history_rows': 32,
        'candidate_lookbacks_evaluated': [24],
        'baseline_selected': 'Harmonic Ridge',
        'regime_anomaly_threshold_x0': float(train_dist['orbit_3d_norm']['p75']),
    }
    (geo_diag_dir / 'geo_validation_report.json').write_text(json.dumps(val_report, indent=2), encoding='utf-8')

    # 3. geo_model_comparison.csv
    actual_geo = geo_test[list(TARGETS)].to_numpy(dtype=float)
    comp_rows = []
    for m_name, preds in all_predictions.items():
        p_geo = preds['GEO']
        res = p_geo - actual_geo
        w_vals = [stats.shapiro(res[:, i]).statistic for i in range(4)]
        comp_rows.append({
            'model': m_name,
            'geo_mean_w': float(np.mean(w_vals)),
            'geo_mae_m': float(np.mean(np.abs(res))),
            'geo_rmse_m': float(np.sqrt(np.mean(res ** 2))),
            'geo_max_ae_m': float(np.max(np.abs(res))),
            'geo_bias_m': float(np.mean(res)),
            'geo_residual_std_m': float(np.mean(np.std(res, axis=0, ddof=1))),
        })
    pd.DataFrame(comp_rows).to_csv(geo_diag_dir / 'geo_model_comparison.csv', index=False)

    # 4. geo_error_by_time.csv
    err_rows = []
    for row_idx, t in enumerate(geo_test['utc_time']):
        row_entry: dict[str, Any] = {'utc_time': str(t)}
        for m_name, preds in all_predictions.items():
            diff = preds['GEO'][row_idx] - actual_geo[row_idx]
            norm_err = float(np.sqrt(np.sum(diff[:3] ** 2)))
            row_entry[f'{m_name}_3d_error_m'] = norm_err
        err_rows.append(row_entry)
    pd.DataFrame(err_rows).to_csv(geo_diag_dir / 'geo_error_by_time.csv', index=False)

    # 5. geo_regime_diagnostics.csv
    new_diag = diagnostics.get('GEO Gated MoE', {})
    regime_rows = []
    for row_idx, t in enumerate(geo_test['utc_time']):
        d_val = new_diag.get(('GEO', row_idx), {})
        regime_rows.append({
            'utc_time': str(t),
            'actual_3d_norm_m': float(np.sqrt(np.sum(actual_geo[row_idx, :3] ** 2))),
            'p_gate': d_val.get('p_gate', np.nan),
            'regime_probability': d_val.get('regime_probability', np.nan),
            'lead_time_hours': d_val.get('lead_time_hours', np.nan),
            'history_span_hours': d_val.get('history_span_hours', np.nan),
            'last_3d_norm_m': d_val.get('last_3d_norm_m', np.nan),
            'delta_x_error_m': d_val.get('delta_x_error_m', np.nan),
            'delta_y_error_m': d_val.get('delta_y_error_m', np.nan),
            'delta_z_error_m': d_val.get('delta_z_error_m', np.nan),
            'delta_clock_error_m': d_val.get('delta_clock_error_m', np.nan),
        })
    pd.DataFrame(regime_rows).to_csv(geo_diag_dir / 'geo_regime_diagnostics.csv', index=False)

    # Plot 1: Actual vs Predicted Components
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    times = pd.to_datetime(geo_test['utc_time'])
    regime_pred = all_predictions['GEO Gated MoE']['GEO']
    bilstm_pred = all_predictions['BiLSTM-GRU']['GEO']
    harmonic_pred = all_predictions['Harmonic Ridge']['GEO']
    for i, target in enumerate(TARGETS):
        axes[i].plot(times, actual_geo[:, i], 'k-', label='Actual', lw=2.0)
        axes[i].plot(times, regime_pred[:, i], 'r.-', label='GEO Gated MoE', lw=1.5)
        axes[i].plot(times, harmonic_pred[:, i], 'b--', label='Harmonic Ridge', lw=1.2, alpha=0.7)
        axes[i].plot(times, bilstm_pred[:, i], 'g:', label='BiLSTM-GRU', lw=1.2, alpha=0.7)
        axes[i].set_ylabel(f'{TARGET_LABELS[target]} (m)')
        axes[i].grid(True, alpha=0.25)
        if i == 0:
            axes[i].legend(loc='upper right', ncol=4)
    axes[-1].set_xlabel('UTC Time (Day 8)')
    fig.suptitle('GEO Satellite Day-8: Actual vs Model Predictions', fontweight='bold', y=0.995)
    fig.tight_layout()
    fig.savefig(geo_diag_dir / 'actual_vs_predicted_components.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # Plot 2: Residuals vs Time
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for i, target in enumerate(TARGETS):
        axes[i].plot(times, regime_pred[:, i] - actual_geo[:, i], 'r.-', label='GEO Gated MoE', lw=1.5)
        axes[i].plot(times, harmonic_pred[:, i] - actual_geo[:, i], 'b--', label='Harmonic Ridge', lw=1.2, alpha=0.7)
        axes[i].plot(times, bilstm_pred[:, i] - actual_geo[:, i], 'g:', label='BiLSTM-GRU', lw=1.2, alpha=0.7)
        axes[i].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[i].set_ylabel(f'Residual {TARGET_LABELS[target]} (m)')
        axes[i].grid(True, alpha=0.25)
        if i == 0:
            axes[i].legend(loc='upper right', ncol=3)
    axes[-1].set_xlabel('UTC Time (Day 8)')
    fig.suptitle('GEO Satellite Day-8: Residual Error Traces over Time', fontweight='bold', y=0.995)
    fig.tight_layout()
    fig.savefig(geo_diag_dir / 'residuals_vs_time.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # Plot 3: Predicted Amplitude vs Actual Amplitude
    fig, ax = plt.subplots(figsize=(10, 6))
    act_norm = np.sqrt(np.sum(actual_geo[:, :3] ** 2, axis=1))
    regime_norm = np.sqrt(np.sum(regime_pred[:, :3] ** 2, axis=1))
    bilstm_norm = np.sqrt(np.sum(bilstm_pred[:, :3] ** 2, axis=1))
    ax.scatter(act_norm, regime_norm, color='red', alpha=0.7, label='GEO Gated MoE')
    ax.scatter(act_norm, bilstm_norm, color='green', alpha=0.5, label='BiLSTM-GRU (collapsed near ~0.3m)')
    lim = max(act_norm.max(), regime_norm.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.6, label='Ideal 1:1')
    ax.set_xlabel('Actual 3D Orbit Norm (m)')
    ax.set_ylabel('Predicted 3D Orbit Norm (m)')
    ax.set_title('Prediction Amplitude vs Actual Amplitude (Demonstrating Elimination of Central Collapse)', fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(geo_diag_dir / 'amplitude_vs_actual.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # Plot 4: Regime Probability vs Time
    fig, ax = plt.subplots(figsize=(12, 5))
    # p_gate is learned (query-conditional via GRU), not a static thresholded detector.
    p_gates = [d_val.get('p_gate', 0.5) for d_val in (new_diag.get(('GEO', idx), {}) for idx in range(len(geo_test)))]
    ax.plot(times, p_gates, 'm.-', label='Learned Regime Gate $p_{gate}$', lw=2.0)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='Decision boundary 0.5')
    ax.set_ylabel('p_gate (regime gate probability)')
    ax.set_xlabel('UTC Time (Day 8)')
    ax.set_title('GEO Day-8: Learned Regime Gate $p_{gate}$ over Time (GEO Gated MoE)', fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(geo_diag_dir / 'regime_probability_vs_time.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # Plot 5: Delta t distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    dt_test = (geo_test['utc_time'].diff().dt.total_seconds() / 60.0).dropna()
    ax.hist(dt_train, bins=25, alpha=0.6, color='blue', label=f'GEO Train Gaps (N={len(dt_train)})')
    ax.hist(dt_test, bins=20, alpha=0.6, color='orange', label=f'GEO Test Gaps (N={len(dt_test)})')
    ax.set_xlabel(r'Inter-sample Gap $\Delta t$ (minutes)')
    ax.set_ylabel('Frequency')
    ax.set_title(r'GEO Sampling Gap Distribution ($\Delta t$) Demonstrating Irregularity', fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(geo_diag_dir / 'delta_t_distribution.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

def run_benchmark(data_dir: Path, output_dir: Path, max_epochs: int=180, device_name: str='auto') -> dict[str, Any]:
    seed_everything(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_official_split(data_dir)
    device = resolve_device(device_name)
    predictors: list[tuple[str, Callable[[], Any]]] = [
        ('Persistence', lambda: predict_persistence(datasets, output_dir)),
        ('Harmonic Ridge', lambda: predict_harmonic_ridge(datasets, output_dir)),
        ('Random Forest', lambda: predict_random_forest(datasets, output_dir)),
        ('Gaussian Process', lambda: predict_gaussian_process(datasets, output_dir)),
        ('BiLSTM-GRU', lambda: predict_neural(datasets, 'bilstm_gru', device, max_epochs, output_dir)),
        ('Transformer', lambda: predict_neural(datasets, 'transformer', device, max_epochs, output_dir)),
        ('GEO Gated MoE', lambda: predict_geo_gated_moe(datasets, device, max_epochs, output_dir)),
    ]
    all_predictions: dict[str, dict[str, np.ndarray]] = {}
    all_diagnostics: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    model_metrics: dict[str, Any] = {}
    for model_name, predictor in predictors:
        print(f'Training/evaluating {model_name}...')
        result = predictor()
        if isinstance(result, tuple) and len(result) == 2:
            predictions, diag = result
            all_diagnostics[model_name] = diag
        else:
            predictions = result
        all_predictions[model_name] = predictions
        model_metrics[model_name] = evaluate_predictions(datasets, predictions)
        slug = model_name.lower().replace('-', '_').replace(' ', '_')
        save_qq_plot(datasets, predictions, model_name, output_dir / f'qq_{slug}.png')
        ci = model_metrics[model_name].get('average_shapiro_w_ci_95', [0.0, 0.0])
        print(f"  Avg W={model_metrics[model_name]['average_shapiro_w']:.6f} [95% CI: {ci[0]:.4f}, {ci[1]:.4f}] MAE={model_metrics[model_name]['overall_mae_m']:.6f} m")

    print('Selecting MEO specialists from training-only rolling-origin folds...')
    meo_selections = select_meo_specialists(datasets)
    specialist_predictions, specialist_routing = compose_orbit_class_specialist(
        datasets, all_predictions, meo_selections
    )
    specialist_name = 'Orbit-Class Specialist'
    all_predictions[specialist_name] = specialist_predictions
    model_metrics[specialist_name] = evaluate_predictions(datasets, specialist_predictions)
    save_qq_plot(
        datasets,
        specialist_predictions,
        specialist_name,
        output_dir / 'qq_orbit_class_specialist.png',
    )
    specialist_manifest = {
        'artifact_schema_version': 1,
        'model': specialist_name,
        'seed': SEED,
        'selection_policy': 'minimum training-only rolling-origin MAE by orbit-vector and clock target',
        'day8_labels_used_for_selection': False,
        'routing': specialist_routing,
        'meo_training_validation': meo_selections,
        'component_artifacts': {
            'GEO Gated MoE': 'geo_gated_moe_day8.pt',
            'Gaussian Process': 'gaussian_process_day8.joblib',
            'Random Forest': 'random_forest_day8.joblib',
            'Harmonic Ridge': 'harmonic_ridge_day8.joblib',
            'Persistence': 'persistence_state.json',
        },
    }
    (output_dir / 'orbit_class_specialist_manifest.json').write_text(
        json.dumps(specialist_manifest, indent=2), encoding='utf-8'
    )
    for series_name, routing in specialist_routing.items():
        print(
            f"  {series_name}: orbit={routing['orbit_model']}; "
            f"clock={routing['clock_model']}"
        )

    ranking = sorted(model_metrics, key=lambda name: (-model_metrics[name]['average_shapiro_w'], model_metrics[name]['mean_absolute_bias'], model_metrics[name]['average_residual_std']))
    for rank, model_name in enumerate(ranking, start=1):
        model_metrics[model_name]['rank'] = rank

    def aggregate_meo(metric_name: str, model_name: str) -> float:
        slices = [
            model_metrics[model_name]['per_series'][series_name]
            for series_name, item in datasets.items()
            if item['spec'].orbit_class == 'MEO'
        ]
        return float(
            sum(item[metric_name] * item['rows'] for item in slices)
            / sum(item['rows'] for item in slices)
        )

    meo_accuracy = {
        model_name: {
            'orbit_vector_mae_m': aggregate_meo('orbit_vector_mae_m', model_name),
            'clock_mae_m': aggregate_meo('clock_mae_m', model_name),
        }
        for model_name in model_metrics
    }
    previous = meo_accuracy['GEO Gated MoE']
    candidate = meo_accuracy[specialist_name]
    meo_promotion = {
        'candidate': specialist_name,
        'baseline': 'GEO Gated MoE',
        'passed': bool(
            candidate['orbit_vector_mae_m'] < previous['orbit_vector_mae_m']
            and candidate['clock_mae_m'] < previous['clock_mae_m']
        ),
        'candidate_metrics': candidate,
        'baseline_metrics': previous,
        'relative_orbit_improvement': float(
            1.0 - candidate['orbit_vector_mae_m'] / previous['orbit_vector_mae_m']
        ),
        'relative_clock_improvement': float(
            1.0 - candidate['clock_mae_m'] / previous['clock_mae_m']
        ),
    }

    report = {
        'evaluation_protocol': {
            'source': 'docs/reference/ps08/competition_note.pdf',
            'train_policy': 'seven-day train files only',
            'test_policy': 'unique rows at supplied Day-8 arbitrary timestamps; no test feedback',
            'residual_definition': 'prediction - observation',
            'primary_metric': 'macro mean Shapiro-Wilk W across 3 series x 4 equally weighted targets',
            'aggregation_rationale': 'test each orbit series separately to avoid mixing distributions, then macro-average all 12 per-series/per-target Shapiro-Wilk evaluations',
            'alpha': 0.05,
            'tie_breakers': ['residual mean and standard deviation', 'Q-Q plot outliers'],
            'published_reference': PUBLISHED_REFERENCE,
        },
        'data': {
            name: {
                'orbit_class': item['spec'].orbit_class,
                'train_rows_after_deduplication': len(item['train']),
                'test_rows_after_deduplication': len(item['test']),
                'duplicate_train_rows_removed': item['raw_train_rows'] - len(item['train']),
                'duplicate_test_rows_removed': item['raw_test_rows'] - len(item['test']),
                'train_start': item['train']['utc_time'].min().isoformat(),
                'train_end': item['train']['utc_time'].max().isoformat(),
                'test_start': item['test']['utc_time'].min().isoformat(),
                'test_end': item['test']['utc_time'].max().isoformat(),
            } for name, item in datasets.items()
        },
        'winner': ranking[0],
        'ranking': ranking,
        'models': model_metrics,
        'meo_accuracy': meo_accuracy,
        'meo_accuracy_winner': min(
            meo_accuracy,
            key=lambda name: (
                meo_accuracy[name]['orbit_vector_mae_m'],
                meo_accuracy[name]['clock_mae_m'],
            ),
        ),
        'meo_promotion': meo_promotion,
        'orbit_class_specialist': specialist_manifest,
    }
    (output_dir / 'benchmark_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save_predictions(datasets, all_predictions, output_dir / 'day8_predictions.csv', diagnostics=all_diagnostics)
    write_markdown(report, output_dir / 'BENCHMARK_REPORT.md')
    generate_geo_diagnostics(datasets, all_predictions, all_diagnostics, output_dir)

    print('\n' + '=' * 120)
    print(f"{'Model':<26} | {'Avg W':<8} | {'95% Bootstrap CI':<18} | {'MAE (m)':<8} | {'RMSE (m)':<8} | {'Bias':<7} | {'R_Std':<7} | {'GEO W':<8} | {'GEO MAE':<8} | {'GEO RMSE':<8}")
    print('-' * 120)
    for model_name in ranking:
        m = model_metrics[model_name]
        ci = m.get('average_shapiro_w_ci_95', [0.0, 0.0])
        geo_m = m['per_series']['GEO']
        geo_w = geo_m['average_shapiro_w']
        geo_mae = float(np.mean([geo_m['per_target'][t]['mae'] for t in TARGETS]))
        geo_rmse = float(np.mean([geo_m['per_target'][t]['rmse'] for t in TARGETS]))
        ci_str = f'[{ci[0]:.4f}, {ci[1]:.4f}]'
        print(f"{model_name:<26} | {m['average_shapiro_w']:.6f} | {ci_str:<18} | {m['overall_mae_m']:8.4f} | {m['overall_rmse_m']:8.4f} | {m['mean_absolute_bias']:7.4f} | {m['average_residual_std']:7.4f} | {geo_w:.6f} | {geo_mae:8.4f} | {geo_rmse:8.4f}")
    print('=' * 120)
    print(f'Winner by official Priority-1 criterion: {ranking[0]}')
    return report

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Retrain and compare models on official PS-08 data')
    parser.add_argument('--data-dir', type=Path, default=Path('data/ps08'))
    parser.add_argument('--output', type=Path, default=Path('results/ps08_day8'))
    parser.add_argument('--max-epochs', type=int, default=180)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args(argv)
    run_benchmark(args.data_dir, args.output, args.max_epochs, args.device)

if __name__ == '__main__':
    main()
