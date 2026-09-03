from __future__ import annotations
import argparse
import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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

def shapiro_metrics(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    result = stats.shapiro(values)
    return {'count': int(len(values)), 'shapiro_w': float(result.statistic), 'p_value': float(result.pvalue), 'hypothesis_result': int(result.pvalue < 0.05), 'mean': float(np.mean(values)), 'standard_deviation': float(np.std(values, ddof=1)), 'mae': float(np.mean(np.abs(values))), 'rmse': float(np.sqrt(np.mean(values ** 2)))}

def evaluate_predictions(datasets: dict[str, dict[str, Any]], predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    residual_frames = []
    per_series = {}
    for name, item in datasets.items():
        actual = item['test'][list(TARGETS)].to_numpy(dtype=float)
        residuals = predictions[name] - actual
        residual_frames.append(residuals)
        target_metrics = {TARGETS[i]: shapiro_metrics(residuals[:, i]) for i in range(len(TARGETS))}
        per_series[name] = {'rows': len(actual), 'average_shapiro_w': float(np.mean([m['shapiro_w'] for m in target_metrics.values()])), 'per_target': target_metrics}
    residuals = np.vstack(residual_frames)
    pooled_target_metrics = {TARGETS[i]: shapiro_metrics(residuals[:, i]) for i in range(len(TARGETS))}
    independent_tests = [metrics for series_report in per_series.values() for metrics in series_report['per_target'].values()]
    return {'average_shapiro_w': float(np.mean([m['shapiro_w'] for m in independent_tests])), 'average_p_value': float(np.mean([m['p_value'] for m in independent_tests])), 'rejected_test_count': int(sum((m['hypothesis_result'] for m in independent_tests))), 'normality_test_count': len(independent_tests), 'mean_absolute_bias': float(np.mean([abs(m['mean']) for m in independent_tests])), 'average_residual_std': float(np.mean([m['standard_deviation'] for m in independent_tests])), 'overall_mae_m': float(np.mean(np.abs(residuals))), 'overall_rmse_m': float(np.sqrt(np.mean(residuals ** 2))), 'pooled_average_shapiro_w': float(np.mean([m['shapiro_w'] for m in pooled_target_metrics.values()])), 'pooled_per_target': pooled_target_metrics, 'per_series': per_series}

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

def save_predictions(datasets: dict[str, dict[str, Any]], all_predictions: dict[str, dict[str, np.ndarray]], output_path: Path) -> None:
    rows = []
    for model_name, predictions in all_predictions.items():
        for series_name, item in datasets.items():
            actual = item['test'][list(TARGETS)].to_numpy(dtype=float)
            for row_index, timestamp in enumerate(item['test']['utc_time']):
                row: dict[str, Any] = {'model': model_name, 'series': series_name, 'utc_time': timestamp}
                for target_index, target in enumerate(TARGETS):
                    row[f'actual_{target}'] = actual[row_index, target_index]
                    row[f'predicted_{target}'] = predictions[series_name][row_index, target_index]
                    row[f'residual_{target}'] = predictions[series_name][row_index, target_index] - actual[row_index, target_index]
                rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)

def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ['# PS-08 Day-8 Model Benchmark', '', 'Models were trained only on the supplied seven-day files and evaluated at every unique supplied test timestamp. Exact duplicate rows were removed. Test observations were never fed back as model inputs.', '', '## Official ranking', '', f"**Winner: {report['winner']}** — highest average Shapiro–Wilk W across the four equally weighted residual parameters.", '', '| Rank | Model | Avg W | Avg p-value | Rejected tests | MAE (m) | RMSE (m) |', '|---:|---|---:|---:|---:|---:|---:|']
    for model in report['ranking']:
        metrics = report['models'][model]
        lines.append(f"| {metrics['rank']} | {model} | {metrics['average_shapiro_w']:.6f} | {metrics['average_p_value']:.6f} | {metrics['rejected_test_count']}/{metrics['normality_test_count']} | {metrics['overall_mae_m']:.6f} | {metrics['overall_rmse_m']:.6f} |")
    lines.extend(['', 'The primary score is the macro-average of 12 independent tests (3 series × 4 parameters); this avoids mixing different orbit distributions or weighting GEO by its larger row count.', '', 'The published reference benchmark is W = 0.9810, p = 0.5840, hypothesis result = 0. This is a normality benchmark, not an accuracy threshold.', '', '## Judge criteria captured from `Data_PS-08/Note.pdf`', '', '1. Priority 1: average Shapiro–Wilk W over X, Y, Z and clock residuals; higher is better. Report p-values and the α=0.05 decision (0 = fail to reject normality, 1 = reject).', '2. Priority 2: residual mean and standard deviation break a Priority-1 tie.', '3. Priority 3: Q-Q plots and their visible outliers break any remaining tie.', '', 'See `qq_*.png` for every model and `day8_predictions.csv` for row-level evidence.'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def run_benchmark(data_dir: Path, output_dir: Path, max_epochs: int=180, device_name: str='auto') -> dict[str, Any]:
    seed_everything(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_official_split(data_dir)
    device = resolve_device(device_name)
    predictors: list[tuple[str, Callable[[], dict[str, np.ndarray]]]] = [('Persistence', lambda: predict_persistence(datasets, output_dir)), ('Harmonic Ridge', lambda: predict_harmonic_ridge(datasets, output_dir)), ('Random Forest', lambda: predict_random_forest(datasets, output_dir)), ('Gaussian Process', lambda: predict_gaussian_process(datasets, output_dir)), ('BiLSTM-GRU', lambda: predict_neural(datasets, 'bilstm_gru', device, max_epochs, output_dir)), ('Transformer', lambda: predict_neural(datasets, 'transformer', device, max_epochs, output_dir))]
    all_predictions: dict[str, dict[str, np.ndarray]] = {}
    model_metrics: dict[str, Any] = {}
    for model_name, predictor in predictors:
        print(f'Training/evaluating {model_name}...')
        predictions = predictor()
        all_predictions[model_name] = predictions
        model_metrics[model_name] = evaluate_predictions(datasets, predictions)
        slug = model_name.lower().replace('-', '_').replace(' ', '_')
        save_qq_plot(datasets, predictions, model_name, output_dir / f'qq_{slug}.png')
        print(f"  W={model_metrics[model_name]['average_shapiro_w']:.6f} MAE={model_metrics[model_name]['overall_mae_m']:.6f} m")
    ranking = sorted(model_metrics, key=lambda name: (-model_metrics[name]['average_shapiro_w'], model_metrics[name]['mean_absolute_bias'], model_metrics[name]['average_residual_std']))
    for rank, model_name in enumerate(ranking, start=1):
        model_metrics[model_name]['rank'] = rank
    report = {'evaluation_protocol': {'source': 'Data_PS-08/Note.pdf', 'train_policy': 'seven-day train files only', 'test_policy': 'unique rows at supplied Day-8 arbitrary timestamps; no test feedback', 'residual_definition': 'prediction - observation', 'primary_metric': 'macro mean Shapiro-Wilk W across 3 series x 4 equally weighted targets', 'aggregation_rationale': 'test each orbit series separately to avoid mixing distributions, then macro-average all 12 W values', 'alpha': 0.05, 'tie_breakers': ['residual mean and standard deviation', 'Q-Q plot outliers'], 'published_reference': PUBLISHED_REFERENCE}, 'data': {name: {'orbit_class': item['spec'].orbit_class, 'train_rows_after_deduplication': len(item['train']), 'test_rows_after_deduplication': len(item['test']), 'duplicate_train_rows_removed': item['raw_train_rows'] - len(item['train']), 'duplicate_test_rows_removed': item['raw_test_rows'] - len(item['test']), 'train_start': item['train']['utc_time'].min().isoformat(), 'train_end': item['train']['utc_time'].max().isoformat(), 'test_start': item['test']['utc_time'].min().isoformat(), 'test_end': item['test']['utc_time'].max().isoformat()} for name, item in datasets.items()}, 'winner': ranking[0], 'ranking': ranking, 'models': model_metrics}
    (output_dir / 'benchmark_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save_predictions(datasets, all_predictions, output_dir / 'day8_predictions.csv')
    write_markdown(report, output_dir / 'BENCHMARK_REPORT.md')
    print(f'Winner by official Priority-1 criterion: {ranking[0]}')
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description='Retrain and compare models on official PS-08 data')
    parser.add_argument('--data-dir', type=Path, default=Path('Data_PS-08'))
    parser.add_argument('--output', type=Path, default=Path('results/ps08_day8'))
    parser.add_argument('--max-epochs', type=int, default=180)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    run_benchmark(args.data_dir, args.output, args.max_epochs, args.device)
if __name__ == '__main__':
    main()
