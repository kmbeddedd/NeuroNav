"""Satellite-Specific Model Calibration and Evaluation Engine for NeuroNav."""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

from src.models.adapters import (
    ForecastModelAdapter,
    _standardize_target_matrix,
    get_available_model_adapters,
)
from src.satellite_registry import (
    SatelliteModelRegistry,
    compute_dataset_hash,
)

SATELLITE_COLS = ("Satellite_ID", "satellite_id", "sat_id", "PRN", "prn", "Satellite")
TIME_COLS = ("utc_time", "Timestamp", "timestamp", "time", "Date_Time", "datetime")


def detect_satellite_col(df: pd.DataFrame) -> Optional[str]:
    """Identify the satellite identifier column in dataframe."""
    for col in SATELLITE_COLS:
        if col in df.columns:
            return col
    return None


def detect_time_col(df: pd.DataFrame) -> Optional[str]:
    """Identify the timestamp column in dataframe."""
    for col in TIME_COLS:
        if col in df.columns:
            return col
    return None


def compute_target_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute complete statistical evaluation metrics for 1D target residuals."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "mae": 0.0, "rmse": 0.0, "bias": 0.0, "std": 0.0,
            "r2": 0.0, "max_ae": 0.0, "shapiro_w": 1.0, "shapiro_p": 1.0,
            "residuals": [],
        }

    # Match lengths
    min_len = min(len(y_true), len(y_pred))
    yt = y_true[:min_len]
    yp = y_pred[:min_len]
    residuals = yp - yt

    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    bias = float(np.mean(residuals))
    std = float(np.std(residuals))
    max_ae = float(np.max(np.abs(residuals)))

    try:
        r2 = float(r2_score(yt, yp))
    except Exception:
        r2 = 0.0

    # Shapiro-Wilk normality test on residuals
    shapiro_w = 1.0
    shapiro_p = 1.0
    if len(residuals) >= 3:
        try:
            # Subsample if length > 5000 as per scipy requirement
            sample = residuals if len(residuals) <= 5000 else np.random.choice(residuals, 5000, replace=False)
            if np.all(sample == sample[0]):
                shapiro_w, shapiro_p = 1.0, 1.0
            else:
                stat, p_val = stats.shapiro(sample)
                shapiro_w = float(stat) if not np.isnan(stat) else 1.0
                shapiro_p = float(p_val) if not np.isnan(p_val) else 1.0
        except Exception:
            shapiro_w, shapiro_p = 1.0, 1.0

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "std": std,
        "r2": r2,
        "max_ae": max_ae,
        "shapiro_w": shapiro_w,
        "shapiro_p": shapiro_p,
        "residuals": [float(r) for r in residuals],
    }


def compute_multi_target_evaluation(
    true_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    target_names: Tuple[str, ...] = ("X", "Y", "Z", "Clock"),
) -> Dict[str, Any]:
    """Compute per-target metrics and composite multi-target selection score.
    
    Composite Selection Score:
        Score = Shapiro_W_mean / (1.0 + MAE_3D + MAE_Clock_norm)
    Higher is better (balanced orbital accuracy, clock accuracy, and residual normality).
    """
    min_len = min(len(true_matrix), len(pred_matrix))
    t_mat = true_matrix[:min_len]
    p_mat = pred_matrix[:min_len]

    per_target = {}
    w_list = []
    p_list = []

    for idx, name in enumerate(target_names):
        m = compute_target_metrics(t_mat[:, idx], p_mat[:, idx])
        per_target[name] = m
        w_list.append(m["shapiro_w"])
        p_list.append(m["shapiro_p"])

    # 3D Orbit Error
    diff_xyz = p_mat[:, :3] - t_mat[:, :3]
    orbit_3d_errors = np.sqrt(np.sum(diff_xyz ** 2, axis=1))
    mae_3d = float(np.mean(orbit_3d_errors))
    rmse_3d = float(np.sqrt(np.mean(orbit_3d_errors ** 2)))

    mae_clock = per_target["Clock"]["mae"]
    w_mean = float(np.mean(w_list))
    p_mean = float(np.mean(p_list))

    # Scale clock error so meters / meters are comparable
    clock_scale = 1.0
    mae_clock_norm = mae_clock * clock_scale

    # Composite score formula:
    # bounded, penalizes 3D and clock errors, rewards high Shapiro-Wilk Gaussianity
    composite_score = float(w_mean / (1.0 + mae_3d + mae_clock_norm))

    return {
        "composite_score": composite_score,
        "mae_3d": mae_3d,
        "rmse_3d": rmse_3d,
        "mae_clock": mae_clock,
        "shapiro_w_mean": w_mean,
        "shapiro_p_mean": p_mean,
        "per_target": per_target,
    }


class SatelliteCalibrationEngine:
    """End-to-end calibrator: evaluates all candidate models per satellite without data leakage."""

    def __init__(self, registry: Optional[SatelliteModelRegistry] = None):
        self.registry = registry or SatelliteModelRegistry()
        self.adapters: Dict[str, ForecastModelAdapter] = get_available_model_adapters()

    def calibrate(
        self,
        train_df_or_path: pd.DataFrame | str | Path,
        test_df_or_path: pd.DataFrame | str | Path,
        generate_report: bool = True,
        target_satellite_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute Stage 1 Calibration across all satellites present in dataset.
        
        Guarantees zero-leakage:
        - Only train_df is provided to adapter.check_eligibility() and adapter.fit().
        - test_df is strictly used as post-prediction ground truth.
        """
        # Load dataframes if paths provided
        train_df = pd.read_csv(train_df_or_path) if not isinstance(train_df_or_path, pd.DataFrame) else train_df_or_path.copy()
        test_df = pd.read_csv(test_df_or_path) if not isinstance(test_df_or_path, pd.DataFrame) else test_df_or_path.copy()

        dataset_hash = compute_dataset_hash(train_df)
        sat_col_train = detect_satellite_col(train_df)
        sat_col_test = detect_satellite_col(test_df)
        time_col_train = detect_time_col(train_df)
        time_col_test = detect_time_col(test_df)

        # Standardize satellite lists
        if target_satellite_id and str(target_satellite_id).strip():
            clean_sat_id = str(target_satellite_id).strip()
            satellites = [clean_sat_id]
            train_df["Satellite_ID"] = clean_sat_id
            test_df["Satellite_ID"] = clean_sat_id
            sat_col_train = "Satellite_ID"
            sat_col_test = "Satellite_ID"
        elif sat_col_train and sat_col_train in train_df.columns:
            satellites = sorted([str(s) for s in train_df[sat_col_train].dropna().unique()])
        else:
            satellites = ["SAT_GLOBAL"]
            train_df["Satellite_ID"] = "SAT_GLOBAL"
            sat_col_train = "Satellite_ID"

        results_by_satellite: Dict[str, Any] = {}
        comparison_matrix: Dict[str, Dict[str, float]] = {m_id: {} for m_id in self.adapters}

        for sat_id in satellites:
            # 1. Slice training data for satellite (strictly 7-day data)
            if sat_col_train:
                sat_train = train_df[train_df[sat_col_train].astype(str) == sat_id].copy()
            else:
                sat_train = train_df.copy()

            if sat_col_test and sat_col_test in test_df.columns:
                sat_test = test_df[test_df[sat_col_test].astype(str) == sat_id].copy()
            else:
                sat_test = test_df.copy()

            if len(sat_train) == 0:
                continue

            # Sort temporally
            if time_col_train and time_col_train in sat_train.columns:
                sat_train[time_col_train] = pd.to_datetime(sat_train[time_col_train])
                sat_train = sat_train.sort_values(by=time_col_train).reset_index(drop=True)

            if time_col_test and time_col_test in sat_test.columns:
                sat_test[time_col_test] = pd.to_datetime(sat_test[time_col_test])
                sat_test = sat_test.sort_values(by=time_col_test).reset_index(drop=True)

            # Target ground-truth matrix for 8th day
            test_truth_matrix = _standardize_target_matrix(sat_test)
            target_times = sat_test[time_col_test] if (time_col_test and time_col_test in sat_test.columns) else None
            horizon_steps = len(sat_test) if len(sat_test) > 0 else 96

            candidate_scores: Dict[str, float] = {}
            candidate_metrics: Dict[str, Any] = {}
            model_eligibility: Dict[str, Dict[str, Any]] = {}

            # Evaluate each registered model
            for model_id, adapter in self.adapters.items():
                # Check eligibility
                is_eligible, reason = adapter.check_eligibility(sat_train, sat_id)
                model_eligibility[model_id] = {
                    "eligible": is_eligible,
                    "reason": reason,
                }

                if not is_eligible:
                    continue

                try:
                    # Fit strictly on historical training data
                    adapter.fit(sat_train, sat_id)

                    # Predict future 8th-day horizon
                    pred_df = adapter.predict(
                        train_df=sat_train,
                        horizon_steps=horizon_steps,
                        target_times=target_times,
                        satellite_id=sat_id,
                    )

                    pred_matrix = _standardize_target_matrix(pred_df)

                    # Evaluate predictions vs ground truth
                    eval_res = compute_multi_target_evaluation(test_truth_matrix, pred_matrix)
                    score = eval_res["composite_score"]

                    candidate_scores[model_id] = score
                    candidate_metrics[model_id] = eval_res
                    comparison_matrix[model_id][sat_id] = score

                except Exception as exc:
                    # Low-data or model execution error handled gracefully
                    model_eligibility[model_id] = {
                        "eligible": False,
                        "reason": f"Execution error: {str(exc)}",
                    }

            # Determine best model for THIS satellite
            if candidate_scores:
                best_model = max(candidate_scores.keys(), key=lambda m: candidate_scores[m])
                best_score = candidate_scores[best_model]
            else:
                # Fallback to persistence if available
                best_model = "persistence"
                best_score = 0.0
                candidate_scores["persistence"] = 0.0

            best_adapter = self.adapters.get(best_model)
            version = best_adapter.version if best_adapter else "1.0.0"

            # Save to persistent satellite memory
            entry = self.registry.save_calibration_result(
                satellite_id=sat_id,
                best_model=best_model,
                score=best_score,
                candidate_models=candidate_scores,
                validation_metrics=candidate_metrics,
                model_version=version,
                dataset_hash=dataset_hash,
            )

            results_by_satellite[sat_id] = {
                "satellite_id": sat_id,
                "winner": entry.get("selected_model"),
                "selection_mode": entry.get("selection_mode"),
                "score": entry.get("score"),
                "candidate_scores": candidate_scores,
                "metrics": candidate_metrics,
                "eligibility": model_eligibility,
                "entry": entry,
            }

        # Generate machine-readable audit report
        report_paths = {}
        if generate_report:
            report_paths = self.registry.generate_audit_report()

        return {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset_hash": dataset_hash,
            "satellites": results_by_satellite,
            "comparison_matrix": comparison_matrix,
            "audit_reports": report_paths,
        }
