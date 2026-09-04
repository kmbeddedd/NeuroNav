"""Calibration and Operational Forecasting Pipeline.

Implements:
- Phase A: Calibration Pipeline (7-day train + 8th-day truth -> evaluate -> rank per sat -> select -> persist in registry -> auditable report)
- Phase B: Operational Forecast Pipeline (new 7-day data -> lookup registry per sat -> route & predict -> record model used)
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.forecasting.base import ForecastModel
from src.forecasting.eligibility import compute_eligibility_matrix
from src.forecasting.models import MODEL_REGISTRY, create_model, get_available_model_names
from src.forecasting.registry import SatelliteModelRegistry, SatelliteSelection
from src.forecasting.router import PredictionRouter
from src.forecasting.validation import load_telemetry_source, validate_dataset
from src.physics import compute_sisre, ecef_error_to_ric, nominal_satellite_orbit

logger = logging.getLogger(__name__)

TARGET_COLS = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]


def compute_metrics_for_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    orbit_class: str = "MEO",
    timestamps: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, Any]:
    """Computes full suite of required scientific metrics across all 4 targets and 3D vector norm."""
    residuals = predicted - actual  # (N, 4)
    n_samples = len(residuals)

    res_x = residuals[:, 0]
    res_y = residuals[:, 1]
    res_z = residuals[:, 2]
    res_clk = residuals[:, 3]

    norm_3d_actual = np.sqrt(np.sum(actual[:, :3] ** 2, axis=1))
    norm_3d_pred = np.sqrt(np.sum(predicted[:, :3] ** 2, axis=1))
    # True vector residual error norm: ||pred - act||
    vector_3d_errors = np.sqrt(np.sum(residuals[:, :3] ** 2, axis=1))

    per_target = {}
    target_names = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]
    shapiro_w_list = []
    shapiro_p_list = []

    for idx, name in enumerate(target_names):
        r = residuals[:, idx]
        valid_r = r[np.isfinite(r)]
        mae = float(np.mean(np.abs(valid_r)))
        rmse = float(np.sqrt(np.mean(np.square(valid_r))))
        bias = float(np.mean(valid_r))
        std = float(np.std(valid_r))
        max_ae = float(np.max(np.abs(valid_r)))

        # R2 score
        act_col = actual[:, idx]
        ss_res = np.sum(np.square(valid_r))
        ss_tot = np.sum(np.square(act_col - np.mean(act_col)))
        r2 = float(1.0 - (ss_res / max(ss_tot, 1e-12))) if ss_tot > 1e-12 else 0.0

        # Shapiro-Wilk test for normality
        if len(valid_r) >= 3:
            # SciPy handles up to 5000 samples
            sample_sub = valid_r if len(valid_r) <= 5000 else np.random.choice(valid_r, 5000, replace=False)
            w_stat, p_val = stats.shapiro(sample_sub)
            w_stat, p_val = float(w_stat), float(p_val)
        else:
            w_stat, p_val = 0.0, 0.0

        shapiro_w_list.append(w_stat)
        shapiro_p_list.append(p_val)

        per_target[name] = {
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "std": std,
            "max_ae": max_ae,
            "r2": r2,
            "shapiro_w": w_stat,
            "shapiro_p": p_val,
            "p_value": p_val,
            "hypothesis_result": int(p_val < 0.05),
            "is_normal": bool(p_val >= 0.05),
        }

    # SISRE calculation: convert coordinates to RIC if timestamps provided
    if timestamps is not None and len(timestamps) == n_samples:
        pos, vel = nominal_satellite_orbit(timestamps, orbit_class=orbit_class)
        ric_res = ecef_error_to_ric(residuals[:, :3], pos, vel)
        sisre_arr = compute_sisre(
            radial_error=ric_res[:, 0],
            along_track_error=ric_res[:, 1],
            cross_track_error=ric_res[:, 2],
            clock_error=residuals[:, 3],
            orbit_class=orbit_class,
        )
    else:
        # Fallback estimation of radial vs along/cross if timestamps absent
        sisre_arr = compute_sisre(
            radial_error=residuals[:, 0] * 0.5,
            along_track_error=residuals[:, 1],
            cross_track_error=residuals[:, 2],
            clock_error=residuals[:, 3],
            orbit_class=orbit_class,
        )

    return {
        "n_samples": n_samples,
        "per_target": per_target,
        "orbit_3d_vector_mae_m": float(np.mean(vector_3d_errors)),
        "orbit_3d_vector_rmse_m": float(np.sqrt(np.mean(np.square(vector_3d_errors)))),
        "orbit_3d_vector_max_ae_m": float(np.max(vector_3d_errors)),
        "clock_mae_m": per_target["clock_error_m"]["mae"],
        "mean_shapiro_w": float(np.mean(shapiro_w_list)),
        "mean_shapiro_p": float(np.mean(shapiro_p_list)),
        "sisre_mean_m": float(np.mean(sisre_arr)),
        "sisre_rms_m": float(np.sqrt(np.mean(np.square(sisre_arr)))),
    }


class CalibrationPipeline:
    """Orchestrates Phase A Model Calibration across all satellites."""

    def __init__(
        self,
        registry: Optional[SatelliteModelRegistry] = None,
        artifacts_dir: Union[str, Path] = "models/registry/artifacts",
        reports_dir: Union[str, Path] = "reports/calibration",
        primary_metric: str = "orbit_3d_vector_mae_m",
        candidate_models: Optional[List[str]] = None,
    ):
        self.registry = registry or SatelliteModelRegistry()
        self.artifacts_dir = Path(artifacts_dir)
        self.reports_dir = Path(reports_dir)
        self.primary_metric = primary_metric
        self.candidate_models = candidate_models or [
            "persistence",
            "harmonic_ridge",
            "random_forest",
            "gaussian_process",
            "geo_moe",
            "bilstm_gru",
            "transformer",
            "decoupled_clock",
            "nhits",
        ]

    def run_calibration(
        self,
        train_data: Union[str, Path, pd.DataFrame, Dict[str, pd.DataFrame]],
        test_data: Union[str, Path, pd.DataFrame, Dict[str, pd.DataFrame]],
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes complete Phase A calibration."""
        now = datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        run_id = run_id or f"calib_{timestamp_str}"
        report_dir = self.reports_dir / run_id
        report_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # 1. Validate train and test datasets
        train_val = validate_dataset(train_data, min_history_rows=8)
        test_val = validate_dataset(test_data, min_history_rows=1, is_test_dataset=True)

        if not train_val.is_valid:
            return {
                "status": "failed",
                "stage": "train_data_validation",
                "validation_result": train_val.to_dict(),
            }
        if not test_val.is_valid:
            return {
                "status": "failed",
                "stage": "test_data_validation",
                "validation_result": test_val.to_dict(),
            }

        train_series = train_val.normalized_data or {}
        test_series = test_val.normalized_data or {}

        # Detect satellites common to both train and test
        common_sats = sorted(set(train_series.keys()).intersection(set(test_series.keys())))
        if not common_sats:
            return {
                "status": "failed",
                "reason": f"No overlapping satellites between train ({list(train_series.keys())}) and test ({list(test_series.keys())})",
            }

        # 2. Check model eligibility matrix
        eligibility_matrix = compute_eligibility_matrix(train_series, self.candidate_models)

        calibration_summary = {
            "run_id": run_id,
            "timestamp": now.isoformat(),
            "primary_metric": self.primary_metric,
            "satellites_evaluated": common_sats,
            "candidate_models": self.candidate_models,
            "satellite_winners": {},
            "detailed_scores": {},
        }

        comparison_rows = []
        detailed_rows = []

        # 3. Train, predict, and evaluate eligible models per satellite
        for sat_id in common_sats:
            sat_train = train_series[sat_id]
            sat_test = test_series[sat_id]
            test_times = pd.to_datetime(sat_test["utc_time"]).reset_index(drop=True)
            test_actuals = sat_test[TARGET_COLS].to_numpy(dtype=np.float64)

            # Compute fingerprint/hash of training data
            train_bytes = sat_train[TARGET_COLS].to_numpy().tobytes()
            train_hash = hashlib.sha256(train_bytes).hexdigest()[:16]

            candidate_scores: Dict[str, float] = {}
            candidate_metrics: Dict[str, Dict[str, Any]] = {}
            trained_models: Dict[str, ForecastModel] = {}

            for model_name in self.candidate_models:
                elig = eligibility_matrix[sat_id].get(model_name)
                if elig and not elig.eligible:
                    logger.info(f"Skipping {model_name} for {sat_id}: {elig.reason}")
                    continue

                try:
                    # Create and fit candidate
                    model_inst = create_model(model_name)
                    model_inst.fit(sat_train)

                    # Predict on 8th-day timestamps (zero leakage: targets unseen)
                    preds = model_inst.predict(sat_train, test_times)

                    # Compute comprehensive evaluation metrics
                    metrics = compute_metrics_for_residuals(
                        actual=test_actuals,
                        predicted=preds,
                        orbit_class=sat_id,
                        timestamps=test_times,
                    )

                    score = metrics[self.primary_metric]
                    candidate_scores[model_name] = score
                    candidate_metrics[model_name] = metrics
                    trained_models[model_name] = model_inst

                    # Log rows for comparison CSV
                    comparison_rows.append({
                        "run_id": run_id,
                        "satellite_id": sat_id,
                        "model": model_name,
                        "primary_score": score,
                        "orbit_3d_vector_mae_m": metrics["orbit_3d_vector_mae_m"],
                        "clock_mae_m": metrics["clock_mae_m"],
                        "sisre_mean_m": metrics["sisre_mean_m"],
                        "mean_shapiro_w": metrics["mean_shapiro_w"],
                        "mean_p_value": metrics["mean_shapiro_p"],
                        "rejected_normality_tests": sum(t_m["hypothesis_result"] for t_m in metrics["per_target"].values()),
                    })

                    # Log detailed per-target rows
                    for t_name in TARGET_COLS:
                        t_m = metrics["per_target"][t_name]
                        detailed_rows.append({
                            "run_id": run_id,
                            "satellite_id": sat_id,
                            "model": model_name,
                            "target": t_name,
                            "mae": t_m["mae"],
                            "rmse": t_m["rmse"],
                            "bias": t_m["bias"],
                            "std": t_m["std"],
                            "shapiro_w": t_m["shapiro_w"],
                            "shapiro_p": t_m["shapiro_p"],
                            "p_value": t_m["p_value"],
                            "hypothesis_result": t_m["hypothesis_result"],
                            "is_normal": t_m["is_normal"],
                        })

                except Exception as exc:
                    logger.warning(f"Failed evaluation for {model_name} on {sat_id}: {exc}")
                    continue

            if not candidate_scores:
                logger.error(f"No candidate model succeeded for satellite {sat_id}")
                continue

            # 4. Deterministic Model Selection per Satellite
            # Lower is better for error metrics (MAE, RMSE, SISRE); higher is better for Shapiro-Wilk W
            if "shapiro" in self.primary_metric:
                winner_name = max(candidate_scores, key=lambda m: candidate_scores[m])
            else:
                winner_name = min(candidate_scores, key=lambda m: candidate_scores[m])

            winner_score = candidate_scores[winner_name]
            winner_model = trained_models[winner_name]

            # 5. Persist winning model artifact
            artifact_filename = f"{sat_id}_{winner_name}.pt" if "moe" in winner_name or "nhits" in winner_name or "bilstm" in winner_name or "transformer" in winner_name else f"{sat_id}_{winner_name}.joblib"
            artifact_dest = self.artifacts_dir / artifact_filename
            winner_model.save(artifact_dest)

            # 6. Update Persistent Satellite Model Registry
            selection = self.registry.register_calibration_winner(
                satellite_id=sat_id,
                winner_model=winner_name,
                score=winner_score,
                candidate_scores=candidate_scores,
                training_dataset_hash=train_hash,
                model_artifact=str(artifact_dest),
                primary_metric=self.primary_metric,
            )

            calibration_summary["satellite_winners"][sat_id] = {
                "selected_model": selection.selected_model,
                "selection_mode": selection.selection_mode,
                "score": winner_score,
                "candidate_scores": candidate_scores,
                "artifact": str(artifact_dest),
            }
            calibration_summary["detailed_scores"][sat_id] = candidate_metrics

        # 7. Write Machine-Readable Calibration Reports
        summary_file = report_dir / "summary.json"
        summary_file.write_text(json.dumps(calibration_summary, indent=2), encoding="utf-8")

        if comparison_rows:
            comp_df = pd.DataFrame(comparison_rows)
            comp_df.to_csv(report_dir / "satellite_model_comparison.csv", index=False)

        if detailed_rows:
            det_df = pd.DataFrame(detailed_rows)
            det_df.to_csv(report_dir / "detailed_metrics.csv", index=False)

        eligibility_data = {
            s: {m: e.to_dict() for m, e in m_dict.items()}
            for s, m_dict in eligibility_matrix.items()
        }
        (report_dir / "eligibility.json").write_text(json.dumps(eligibility_data, indent=2), encoding="utf-8")

        config_data = {
            "primary_metric": self.primary_metric,
            "candidate_models": self.candidate_models,
            "artifacts_dir": str(self.artifacts_dir),
            "reports_dir": str(self.reports_dir),
        }
        (report_dir / "configuration.json").write_text(json.dumps(config_data, indent=2), encoding="utf-8")

        calibration_summary["status"] = "success"
        calibration_summary["report_dir"] = str(report_dir)
        return calibration_summary
