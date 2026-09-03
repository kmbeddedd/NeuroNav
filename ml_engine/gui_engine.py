from __future__ import annotations
import gzip
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch import nn

warnings.filterwarnings('ignore', category=UserWarning)

TARGETS = ('x_error_m', 'y_error_m', 'z_error_m', 'clock_error_m')
TARGET_LABELS = {
    'x_error_m': 'X Error (m)',
    'y_error_m': 'Y Error (m)',
    'z_error_m': 'Z Error (m)',
    'clock_error_m': 'Clock Error (m)'
}

SP3_MISSING_CLOCK_SENTINEL = 999999.999999

# -------------------------------------------------------------------------
# Neural Network Architectures (matching PS-08 Benchmark)
# -------------------------------------------------------------------------
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
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=96,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.series_embedding = nn.Embedding(num_series, 8)
        self.heads = _SeparateHeads(width + query_dim + 8)

    def forward(self, history: torch.Tensor, query: torch.Tensor, series_id: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_projection(history) + self.position)
        features = torch.cat((encoded[:, -1], query, self.series_embedding(series_id)), dim=-1)
        return self.heads(features)


# -------------------------------------------------------------------------
# Feature Extraction & Time Helpers
# -------------------------------------------------------------------------
def time_features(times: pd.Series | pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    index = pd.DatetimeIndex(times)
    elapsed_days = (index - origin).total_seconds().to_numpy(dtype=float) / 86400.0
    phase = 2.0 * np.pi * (index.hour.to_numpy() * 3600.0 + index.minute.to_numpy() * 60.0 + index.second.to_numpy()) / 86400.0
    columns = [elapsed_days, elapsed_days ** 2 / 49.0]
    for harmonic in range(1, 7):
        columns.extend((np.sin(harmonic * phase), np.cos(harmonic * phase)))
    return np.column_stack(columns).astype(np.float64)


# -------------------------------------------------------------------------
# File Parsers: CSV, SP3, RNX
# -------------------------------------------------------------------------
def parse_csv_dataset(file_path: Path | str) -> pd.DataFrame:
    """Parse CSV dataset with flexible column detection for PS-08 & GNSS benchmark formats."""
    file_path = Path(file_path)
    df = pd.read_csv(file_path)
    
    # Normalize column names: lowercase and stripped
    col_map = {col: ' '.join(col.strip().lower().split()) for col in df.columns}
    norm_df = df.rename(columns=col_map)
    
    # Time column detection
    time_col = None
    for candidate in ('utc_time', 'timestamp', 'time', 'datetime', 'epoch', 'date_time'):
        if candidate in norm_df.columns:
            time_col = candidate
            break
    
    if time_col is None:
        raise ValueError(f"Could not identify a timestamp column in {file_path.name}. Found columns: {list(df.columns)}")
    
    result = pd.DataFrame()
    result['utc_time'] = pd.to_datetime(norm_df[time_col], errors='coerce')
    result = result.dropna(subset=['utc_time'])
    
    # Target mapping for PS-08 format
    target_mappings = {
        'x_error_m': ['x_error (m)', 'x_error_m', 'error_x', 'x_error', 'broadcast_x_error'],
        'y_error_m': ['y_error (m)', 'y_error_m', 'error_y', 'y_error', 'broadcast_y_error'],
        'z_error_m': ['z_error (m)', 'z_error_m', 'error_z', 'z_error', 'broadcast_z_error'],
        'clock_error_m': ['satclockerror (m)', 'satclockerror', 'clock_error_m', 'error_clock', 'clock_error', 'sat_clock_error']
    }
    
    for target_key, candidate_list in target_mappings.items():
        matched = None
        for candidate in candidate_list:
            if candidate in norm_df.columns:
                matched = candidate
                break
        if matched is not None:
            result[target_key] = pd.to_numeric(norm_df[matched], errors='coerce')
    
    # Preserve satellite ID or constellation if present
    for extra in ('satellite_id', 'satellite', 'sat_id', 'prn'):
        if extra in norm_df.columns:
            result['satellite_id'] = norm_df[extra].astype(str)
            break
            
    result = result.sort_values('utc_time').drop_duplicates(subset='utc_time', keep='first').reset_index(drop=True)
    return result


def parse_sp3_file(sp3_path: Path | str) -> pd.DataFrame:
    """Parse standard IGS SP3 (Precise Orbit & Clock) format files."""
    sp3_path = Path(sp3_path)
    records = []
    current_dt: Optional[datetime] = None
    
    opener = gzip.open if str(sp3_path).endswith('.gz') else open
    mode = 'rt' if str(sp3_path).endswith('.gz') else 'r'
    
    with opener(sp3_path, mode, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('*'):
                parts = line[1:].split()
                if len(parts) >= 6:
                    try:
                        year, month, day, hour, minute = map(int, parts[:5])
                        sec = float(parts[5])
                        current_dt = datetime(year, month, day, hour, minute, int(sec))
                    except ValueError:
                        continue
            elif line.startswith('P') and current_dt is not None:
                sat_id = line[1:4].strip()
                tokens = line[4:].split()
                if len(tokens) >= 4:
                    try:
                        x_km = float(tokens[0])
                        y_km = float(tokens[1])
                        z_km = float(tokens[2])
                        clk_us = float(tokens[3])
                        
                        # Convert km to meters
                        x_m = x_km * 1000.0
                        y_m = y_km * 1000.0
                        z_m = z_km * 1000.0
                        
                        # Handle sentinel missing clock
                        if math.isclose(clk_us, SP3_MISSING_CLOCK_SENTINEL, abs_tol=0.001):
                            clk_s = np.nan
                            clk_m = np.nan
                        else:
                            clk_s = clk_us * 1e-06
                            clk_m = clk_s * 299792458.0  # meters (c * dt)
                            
                        records.append({
                            'utc_time': current_dt,
                            'satellite_id': sat_id,
                            'x_error_m': x_m,
                            'y_error_m': y_m,
                            'z_error_m': z_m,
                            'clock_error_m': clk_m
                        })
                    except ValueError:
                        continue
                        
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid SP3 coordinate records found in {sp3_path.name}")
    df = df.sort_values('utc_time').reset_index(drop=True)
    return df


def parse_rnx_file(rnx_path: Path | str) -> pd.DataFrame:
    """Parse RINEX broadcast navigation files (.rnx, .nav, .YYn, .YYp)."""
    rnx_path = Path(rnx_path)
    records = []
    
    opener = gzip.open if str(rnx_path).endswith('.gz') else open
    mode = 'rt' if str(rnx_path).endswith('.gz') else 'r'
    
    in_header = True
    with opener(rnx_path, mode, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if in_header:
                if 'END OF HEADER' in line:
                    in_header = False
                continue
            
            # Epoch record line in RINEX Nav
            line_str = line.rstrip()
            if len(line_str) < 20:
                continue
                
            sat_id = line_str[:3].strip()
            try:
                date_part = line_str[3:22].strip().split()
                if len(date_part) >= 6:
                    year, month, day, hour, minute = map(int, date_part[:5])
                    sec = float(date_part[5])
                    if year < 100:
                        year += 2000 if year < 80 else 1900
                    epoch_dt = datetime(year, month, day, hour, minute, int(sec))
                    
                    # Clock bias (seconds converted to meters)
                    val_str = line_str[22:41].strip().replace('D', 'e').replace('d', 'e')
                    clk_bias = float(val_str) if val_str else 0.0
                    clk_m = clk_bias * 299792458.0
                    
                    records.append({
                        'utc_time': epoch_dt,
                        'satellite_id': sat_id,
                        'x_error_m': 0.0,
                        'y_error_m': 0.0,
                        'z_error_m': 0.0,
                        'clock_error_m': clk_m
                    })
            except Exception:
                continue
                
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid RINEX navigation records found in {rnx_path.name}")
    df = df.sort_values('utc_time').drop_duplicates(subset=['utc_time', 'satellite_id']).reset_index(drop=True)
    return df


def load_dataset_file(file_path: Path | str) -> pd.DataFrame:
    """Universal dataset loader for CSV, SP3, and RNX."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = path.suffix.lower()
    if ext == '.gz':
        stem_ext = Path(path.stem).suffix.lower()
        if stem_ext in ('.sp3',):
            return parse_sp3_file(path)
        elif stem_ext in ('.rnx', '.nav') or stem_ext.endswith('n') or stem_ext.endswith('p'):
            return parse_rnx_file(path)
        else:
            return parse_csv_dataset(path)
    elif ext in ('.csv', '.txt'):
        return parse_csv_dataset(path)
    elif ext in ('.sp3', '.eph'):
        return parse_sp3_file(path)
    elif ext in ('.rnx', '.nav') or ext.endswith('n') or ext.endswith('p'):
        return parse_rnx_file(path)
    else:
        try:
            return parse_csv_dataset(path)
        except Exception:
            return parse_sp3_file(path)


# -------------------------------------------------------------------------
# Model Inference Engine
# -------------------------------------------------------------------------
def detect_series_type(df: pd.DataFrame, file_path: Optional[Path | str] = None) -> str:
    """Infer whether data belongs to GEO, MEO-1, or MEO-2 based on filename or timestamps."""
    name = str(file_path).upper() if file_path else ""
    if 'GEO' in name:
        return 'GEO'
    elif 'MEO_TEST2' in name or 'MEO_TRAIN2' in name or 'MEO-2' in name or 'MEO2' in name:
        return 'MEO-2'
    elif 'MEO' in name:
        return 'MEO-1'
    
    if len(df) > 100:
        return 'GEO'
    return 'MEO-1'


def compute_model_predictions(
    input_df: pd.DataFrame,
    model_name: str,
    target_series: Optional[str] = None,
    models_dir: Optional[Path | str] = None,
    forecast_horizon_hours: int = 24,
    forecast_step_minutes: int = 15,
    custom_forecast_times: Optional[pd.DatetimeIndex | pd.Series] = None
) -> pd.DataFrame:
    """
    Generate model predictions using the requested model.
    If custom_forecast_times is provided (e.g. from 8th-day ground truth), predicts at those exact epochs.
    Otherwise generates forecast across the 24-hour Day-8 horizon.
    """
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent.parent / 'results' / 'ps08_day8'
    else:
        models_dir = Path(models_dir)
        
    series_name = target_series or detect_series_type(input_df)
    
    # Determine forecast query timestamps
    if custom_forecast_times is not None:
        query_times = pd.to_datetime(custom_forecast_times)
    else:
        last_train_time = input_df['utc_time'].max()
        start_time = (last_train_time + timedelta(days=1)).floor('D')
        if start_time <= last_train_time:
            start_time = last_train_time + timedelta(minutes=forecast_step_minutes)
            
        total_steps = int(forecast_horizon_hours * 60 / forecast_step_minutes)
        query_times = pd.date_range(start=start_time, periods=total_steps, freq=f'{forecast_step_minutes}min')
        
    origin = input_df['utc_time'].min().floor('D')
    
    # 1. Harmonic Ridge
    if 'ridge' in model_name.lower():
        model_file = models_dir / 'harmonic_ridge_day8.joblib'
        if model_file.exists():
            models_dict = joblib.load(model_file)
            series_key = series_name if series_name in models_dict else list(models_dict.keys())[0]
            fitted = models_dict[series_key]
            model = fitted['model']
            m_origin = fitted.get('origin', origin)
            x_query = time_features(query_times, m_origin)
            preds = model.predict(x_query)
        else:
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            x_train = time_features(input_df['utc_time'], origin)
            y_train = input_df[list(TARGETS)].to_numpy()
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(x_train, y_train)
            x_query = time_features(query_times, origin)
            preds = model.predict(x_query)

    # 2. Random Forest
    elif 'random forest' in model_name.lower() or 'rf' in model_name.lower():
        model_file = models_dir / 'random_forest_day8.joblib'
        if model_file.exists():
            models_dict = joblib.load(model_file)
            series_key = series_name if series_name in models_dict else list(models_dict.keys())[0]
            fitted = models_dict[series_key]
            model = fitted['model']
            m_origin = fitted.get('origin', origin)
            x_query = time_features(query_times, m_origin)
            preds = model.predict(x_query)
        else:
            from sklearn.ensemble import RandomForestRegressor
            x_train = time_features(input_df['utc_time'], origin)
            y_train = input_df[list(TARGETS)].to_numpy()
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(x_train, y_train)
            x_query = time_features(query_times, origin)
            preds = model.predict(x_query)

    # 3. Gaussian Process
    elif 'gaussian process' in model_name.lower() or 'gp' in model_name.lower():
        model_file = models_dir / 'gaussian_process_day8.joblib'
        if model_file.exists():
            models_dict = joblib.load(model_file)
            series_key = series_name if series_name in models_dict else list(models_dict.keys())[0]
            fitted = models_dict[series_key]
            model = fitted['model']
            m_origin = fitted.get('origin', origin)
            test_days = (pd.DatetimeIndex(query_times) - m_origin).total_seconds().to_numpy(dtype=float).reshape(-1, 1) / 86400.0
            preds = model.predict(test_days)
        else:
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            x_train = time_features(input_df['utc_time'], origin)
            y_train = input_df[list(TARGETS)].to_numpy()
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(x_train, y_train)
            preds = model.predict(time_features(query_times, origin))

    # 4. Neural Network (BiLSTM-GRU or Transformer)
    elif 'bilstm' in model_name.lower() or 'transformer' in model_name.lower():
        arch_file = 'bilstm_gru_day8.pt' if 'bilstm' in model_name.lower() else 'transformer_day8.pt'
        model_path = models_dir / arch_file
        if model_path.exists():
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            scalers = ckpt['series_scalers']
            series_key = series_name if series_name in scalers else list(scalers.keys())[0]
            center, scale = scalers[series_key]
            
            series_idx_map = {'GEO': 0, 'MEO-1': 1, 'MEO-2': 2}
            s_idx = series_idx_map.get(series_key, 0)
            
            last_train = input_df['utc_time'].iloc[-1]
            history_rows = input_df.iloc[-ckpt['sequence_length']:]
            val_norm = (history_rows[list(TARGETS)].to_numpy(dtype=float) - center) / scale
            
            seq_len = ckpt['sequence_length']
            scaled_preds = []
            
            if 'bilstm' in model_name.lower():
                net = IrregularBiLSTMGRU(history_dim=7, query_dim=10, num_series=3)
            else:
                net = IrregularTransformer(history_dim=7, query_dim=10, num_series=3, sequence_length=seq_len)
            net.load_state_dict(ckpt['state_dict'])
            net.eval()
            
            with torch.no_grad():
                for q_t in query_times:
                    phase = 2.0 * np.pi * (q_t.hour * 3600.0 + q_t.minute * 60.0 + q_t.second) / 86400.0
                    lead_days = (q_t - last_train).total_seconds() / 86400.0
                    elapsed_days = (q_t - origin).total_seconds() / 86400.0
                    q_feats = [lead_days / 2.0, elapsed_days / 7.0]
                    for harmonic in range(1, 5):
                        q_feats.extend((math.sin(harmonic * phase), math.cos(harmonic * phase)))
                    q_tensor = torch.tensor([q_feats], dtype=torch.float32)
                    
                    ages = (history_rows['utc_time'] - q_t).dt.total_seconds().to_numpy(dtype=float).reshape(-1, 1) / (2.0 * 86400.0)
                    h_phase = 2.0 * np.pi * (history_rows['utc_time'].dt.hour.to_numpy() * 3600.0 + history_rows['utc_time'].dt.minute.to_numpy() * 60.0) / 86400.0
                    h_arr = np.column_stack((val_norm, ages, np.sin(h_phase), np.cos(h_phase))).astype(np.float32)
                    if len(h_arr) < seq_len:
                        padding = np.repeat(h_arr[[0]], seq_len - len(h_arr), axis=0)
                        h_arr = np.vstack((padding, h_arr))
                    h_tensor = torch.tensor([h_arr], dtype=torch.float32)
                    s_tensor = torch.tensor([s_idx], dtype=torch.int64)
                    
                    out = net(h_tensor, q_tensor, s_tensor).cpu().numpy()[0]
                    scaled_preds.append(out * scale + center)
                    
            preds = np.asarray(scaled_preds)
        else:
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            x_train = time_features(input_df['utc_time'], origin)
            y_train = input_df[list(TARGETS)].to_numpy()
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(x_train, y_train)
            preds = model.predict(time_features(query_times, origin))

    # Default / Baseline Persistence
    else:
        last_vals = input_df[list(TARGETS)].iloc[-1].to_numpy()
        preds = np.repeat(last_vals.reshape(1, -1), len(query_times), axis=0)

    # Format result DataFrame
    res_df = pd.DataFrame({'utc_time': query_times})
    for i, target in enumerate(TARGETS):
        res_df[f'predicted_{target}'] = preds[:, i]
        
    return res_df


# -------------------------------------------------------------------------
# Comparison & Shapiro-Wilk Statistical Evaluation
# -------------------------------------------------------------------------
@dataclass
class TargetMetrics:
    target: str
    target_label: str
    count: int
    shapiro_w: float
    p_value: float
    alpha: float
    reject_normality: bool
    hypothesis_result_str: str
    mean_bias: float
    std_dev: float
    mae: float
    rmse: float


def compute_shapiro_wilk_metrics(residuals: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Compute Shapiro-Wilk W, p-value, and hypothesis test decision."""
    vals = np.asarray(residuals, dtype=float)
    vals = vals[np.isfinite(vals)]
    
    if len(vals) < 3:
        return {
            'count': len(vals),
            'shapiro_w': 1.0,
            'p_value': 1.0,
            'alpha': alpha,
            'reject_normality': False,
            'hypothesis_decision': 'Insufficient Data',
            'mean_bias': 0.0,
            'std_dev': 0.0,
            'mae': 0.0,
            'rmse': 0.0
        }
        
    test_vals = vals if len(vals) <= 5000 else np.random.choice(vals, 5000, replace=False)
    res = stats.shapiro(test_vals)
    w_stat = float(res.statistic)
    p_val = float(res.pvalue)
    reject = bool(p_val < alpha)
    
    decision = "Reject H0 (Non-Gaussian) [Reject]" if reject else "Fail to Reject H0 (Normal) [Pass]"
    
    return {
        'count': len(vals),
        'shapiro_w': w_stat,
        'p_value': p_val,
        'alpha': alpha,
        'reject_normality': reject,
        'hypothesis_decision': decision,
        'mean_bias': float(np.mean(vals)),
        'std_dev': float(np.std(vals, ddof=1)),
        'mae': float(np.mean(np.abs(vals))),
        'rmse': float(np.sqrt(np.mean(vals ** 2)))
    }


def compare_and_evaluate(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, List[TargetMetrics], Dict[str, Any]]:
    """
    Join model predictions with ground truth on timestamp.
    Compute residuals, Shapiro-Wilk test scores, p-values, and hypothesis decisions.
    Returns:
      (merged_df, per_target_metrics, summary_metrics)
    """
    p_df = predictions_df.copy()
    g_df = ground_truth_df.copy()
    
    p_df['utc_time'] = pd.to_datetime(p_df['utc_time'])
    g_df['utc_time'] = pd.to_datetime(g_df['utc_time'])
    
    # Merge on exact timestamp or nearest within 30 minutes tolerance
    merged = pd.merge_asof(
        g_df.sort_values('utc_time'),
        p_df.sort_values('utc_time'),
        on='utc_time',
        direction='nearest',
        tolerance=pd.Timedelta(minutes=30)
    )
    
    merged = merged.dropna(subset=[f'predicted_{t}' for t in TARGETS if f'predicted_{t}' in merged.columns])
    
    if merged.empty:
        # Fallback to row index join if timestamps differ widely
        min_len = min(len(p_df), len(g_df))
        merged = pd.concat([
            g_df.iloc[:min_len].reset_index(drop=True),
            p_df.iloc[:min_len].reset_index(drop=True).drop(columns=['utc_time'], errors='ignore')
        ], axis=1)

    target_metrics_list: List[TargetMetrics] = []
    all_residuals = []
    
    for target in TARGETS:
        pred_col = f'predicted_{target}'
        actual_col = target
        
        if pred_col not in merged.columns or actual_col not in merged.columns:
            continue
            
        res_col = f'residual_{target}'
        merged[res_col] = merged[pred_col] - merged[actual_col]
        res_vals = merged[res_col].to_numpy(dtype=float)
        all_residuals.append(res_vals)
        
        m = compute_shapiro_wilk_metrics(res_vals, alpha=alpha)
        target_metrics_list.append(TargetMetrics(
            target=target,
            target_label=TARGET_LABELS.get(target, target),
            count=m['count'],
            shapiro_w=m['shapiro_w'],
            p_value=m['p_value'],
            alpha=m['alpha'],
            reject_normality=m['reject_normality'],
            hypothesis_result_str=m['hypothesis_decision'],
            mean_bias=m['mean_bias'],
            std_dev=m['std_dev'],
            mae=m['mae'],
            rmse=m['rmse']
        ))
        
    avg_w = float(np.mean([m.shapiro_w for m in target_metrics_list])) if target_metrics_list else 0.0
    avg_p = float(np.mean([m.p_value for m in target_metrics_list])) if target_metrics_list else 0.0
    overall_mae = float(np.mean([m.mae for m in target_metrics_list])) if target_metrics_list else 0.0
    overall_rmse = float(np.mean([m.rmse for m in target_metrics_list])) if target_metrics_list else 0.0
    rejected_count = sum(1 for m in target_metrics_list if m.reject_normality)
    total_tests = len(target_metrics_list)
    
    summary = {
        'average_shapiro_w': avg_w,
        'average_p_value': avg_p,
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'rejected_count': rejected_count,
        'total_tests': total_tests,
        'normality_passed': rejected_count == 0,
        'rows_matched': len(merged)
    }
    
    return merged, target_metrics_list, summary
