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
from dataclasses import asdict

from src.forecasting.base import ForecastModel
from src.forecasting.eligibility import compute_eligibility_matrix
from src.forecasting.models import MODEL_REGISTRY, create_model, get_available_model_names
from src.forecasting.registry import (
    SatelliteModelRegistry,
    SatelliteSelection,
    get_satellite_artifact_dir,
)
from src.forecasting.router import PredictionRouter
from src.forecasting.validation import (
    SatelliteDataset,
    load_telemetry_source,
    normalize_dataframe_columns,
    validate_dataset,
)
from src.physics import compute_sisre, ecef_error_to_ric, nominal_satellite_orbit

logger = logging.getLogger(__name__)

TARGET_COLS = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]
TARGET_ALIASES = {"x_error_m": "X", "y_error_m": "Y", "z_error_m": "Z", "clock_error_m": "Clock"}


def evaluate_residuals_official_hierarchy(
    actual: np.ndarray,
    predicted: np.ndarray,
    orbit_class: str = "MEO",
    timestamps: Optional[pd.DatetimeIndex] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Computes the authoritative Official Competition Evaluation Hierarchy across X, Y, Z, Clock.
    
    Hierarchy:
    - Priority 1: Equal-weighted Shapiro-Wilk W, p-value, H0 decision (alpha=0.05).
    - Priority 2: Equal-weighted residual mean bias and standard deviation (tie-breaker 1).
    - Priority 3: Q-Q plot theoretical vs observed quantiles and outlier severity (tie-breaker 2).
    - Supplementary: MAE, RMSE, 3D Orbit Error, SISRE (isolated from selection decisions).
    """
    residuals = predicted - actual  # (N, 4)
    n_samples = len(residuals)

    # Validate residual array and sample count (Shapiro-Wilk requires finite values and N >= 3)
    if n_samples < 3:
        return {
            "eligible": False,
            "reason": f"insufficient residual samples for Shapiro-Wilk evaluation (n={n_samples} < 3)",
            "n_samples": n_samples,
        }

    target_keys = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]
    alias_keys = ["X", "Y", "Z", "Clock"]

    w_dict: Dict[str, float] = {}
    p_dict: Dict[str, float] = {}
    h_dict: Dict[str, int] = {}
    n_dict: Dict[str, int] = {}
    mean_dict: Dict[str, float] = {}
    std_dict: Dict[str, float] = {}
    qq_details: Dict[str, Dict[str, Any]] = {}
    per_target_flat: Dict[str, Dict[str, Any]] = {}

    total_outliers = 0
    max_discrepancies: List[float] = []

    for idx, (t_name, alias) in enumerate(zip(target_keys, alias_keys)):
        r = residuals[:, idx]
        valid_r = r[np.isfinite(r)]
        n_val = len(valid_r)
        n_dict[alias] = n_val

        if n_val < 3:
            return {
                "eligible": False,
                "reason": f"target {alias} has insufficient finite residual samples (n={n_val} < 3)",
                "n_samples": n_val,
            }

        # Priority 1: Shapiro-Wilk test
        # SciPy handles up to 5000 observations
        sample_sub = valid_r if n_val <= 5000 else np.random.default_rng(42).choice(valid_r, 5000, replace=False)
        w_res = stats.shapiro(sample_sub)
        w_val = float(w_res.statistic)
        p_val = float(w_res.pvalue)
        h_val = int(p_val < alpha)  # 0 = fail to reject H0 (normal), 1 = reject H0 (non-normal)

        w_dict[alias] = w_val
        p_dict[alias] = p_val
        h_dict[alias] = h_val

        # Priority 2: Residual mean and standard deviation
        mu = float(np.mean(valid_r))
        sigma = float(np.std(valid_r, ddof=1)) if n_val > 1 else 0.0
        mean_dict[alias] = mu
        std_dict[alias] = sigma

        # Priority 3: Q-Q Plot and Outlier Analysis
        y_sorted = np.sort(valid_r)
        std_safe = max(sigma, 1e-12)
        z_scores = (y_sorted - mu) / std_safe
        
        # Blom plotting position: p_i = (i - 0.375) / (N + 0.25)
        ranks = np.arange(1, n_val + 1)
        plot_positions = (ranks - 0.375) / (n_val + 0.25)
        q_theoretical = stats.norm.ppf(plot_positions)
        discrepancies = np.abs(z_scores - q_theoretical)
        
        # Documented outlier threshold: Standardized residual |z| > 3.0 or Q-Q discrepancy > 1.0 sigma
        outlier_mask = (np.abs(z_scores) > 3.0) | (discrepancies > 1.0)
        outlier_indices = [int(i) for i in np.where(outlier_mask)[0]]
        num_outliers = len(outlier_indices)
        total_outliers += num_outliers
        max_disc = float(np.max(discrepancies)) if len(discrepancies) > 0 else 0.0
        max_discrepancies.append(max_disc)

        qq_details[alias] = {
            "n": n_val,
            "theoretical_quantiles": q_theoretical.tolist(),
            "observed_residuals": y_sorted.tolist(),
            "standardized_residuals": z_scores.tolist(),
            "discrepancies": discrepancies.tolist(),
            "outlier_indices": outlier_indices,
            "outlier_count": num_outliers,
            "max_discrepancy": max_disc,
        }

        # Basic per-target metrics for reporting and backward compatibility
        mae = float(np.mean(np.abs(valid_r)))
        rmse = float(np.sqrt(np.mean(np.square(valid_r))))
        per_target_flat[t_name] = {
            "mae": mae,
            "rmse": rmse,
            "bias": mu,
            "std": sigma,
            "max_ae": float(np.max(np.abs(valid_r))),
            "shapiro_w": w_val,
            "shapiro_p": p_val,
            "p_value": p_val,
            "hypothesis_result": h_val,
            "is_normal": bool(h_val == 0),
            "qq_outlier_count": num_outliers,
        }

    # Equal weighting: exactly 25% X, 25% Y, 25% Z, 25% Clock
    w_avg = float((w_dict["X"] + w_dict["Y"] + w_dict["Z"] + w_dict["Clock"]) / 4.0)
    p_avg = float((p_dict["X"] + p_dict["Y"] + p_dict["Z"] + p_dict["Clock"]) / 4.0)
    total_rejected = int(sum(h_dict.values()))

    # Priority 2: Equal weighted aggregate mean (absolute bias) and standard deviation
    aggregate_mean = float((abs(mean_dict["X"]) + abs(mean_dict["Y"]) + abs(mean_dict["Z"]) + abs(mean_dict["Clock"])) / 4.0)
    aggregate_std = float((std_dict["X"] + std_dict["Y"] + std_dict["Z"] + std_dict["Clock"]) / 4.0)
    mean_dict["aggregate"] = aggregate_mean
    std_dict["aggregate"] = aggregate_std
    w_dict["average"] = w_avg
    p_dict["average"] = p_avg

    # Priority 3: Aggregate Q-Q metrics
    agg_max_disc = float(np.mean(max_discrepancies))

    # Supplementary / Diagnostic Metrics (NOT used for model selection)
    vector_3d_errors = np.sqrt(np.sum(residuals[:, :3] ** 2, axis=1))
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
        sisre_arr = compute_sisre(
            radial_error=residuals[:, 0] * 0.5,
            along_track_error=residuals[:, 1],
            cross_track_error=residuals[:, 2],
            clock_error=residuals[:, 3],
            orbit_class=orbit_class,
        )

    return {
        "eligible": True,
        "n_samples": n_samples,
        "priority_1": {
            "alpha": alpha,
            "W": w_dict,
            "p_value": p_dict,
            "hypothesis_result": h_dict,
            "sample_count": n_dict,
            "total_rejected_tests": total_rejected,
        },
        "priority_2": {
            "mean": mean_dict,
            "std": std_dict,
        },
        "priority_3": {
            "total_outliers": total_outliers,
            "aggregate_max_discrepancy": agg_max_disc,
            "qq_details": qq_details,
        },
        "supplementary": {
            "orbit_3d_vector_mae_m": float(np.mean(vector_3d_errors)),
            "orbit_3d_vector_rmse_m": float(np.sqrt(np.mean(np.square(vector_3d_errors)))),
            "clock_mae_m": per_target_flat["clock_error_m"]["mae"],
            "sisre_mean_m": float(np.mean(sisre_arr)),
            "sisre_rms_m": float(np.sqrt(np.mean(np.square(sisre_arr)))),
        },
        # Backward compatibility aliases
        "per_target": per_target_flat,
        "orbit_3d_vector_mae_m": float(np.mean(vector_3d_errors)),
        "orbit_3d_vector_rmse_m": float(np.sqrt(np.mean(np.square(vector_3d_errors)))),
        "clock_mae_m": per_target_flat["clock_error_m"]["mae"],
        "mean_shapiro_w": w_avg,
        "mean_shapiro_p": p_avg,
        "sisre_mean_m": float(np.mean(sisre_arr)),
        "sisre_rms_m": float(np.sqrt(np.mean(np.square(sisre_arr)))),
    }


def compute_metrics_for_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    orbit_class: str = "MEO",
    timestamps: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, Any]:
    """Backward-compatible wrapper for evaluate_residuals_official_hierarchy."""
    return evaluate_residuals_official_hierarchy(
        actual=actual,
        predicted=predicted,
        orbit_class=orbit_class,
        timestamps=timestamps,
        alpha=0.05,
    )


def compare_models_hierarchical(
    m_a: Dict[str, Any],
    m_b: Dict[str, Any],
    tie_tolerance: float = 1e-4,
) -> Tuple[int, str]:
    """Compares Model A vs Model B using the authoritative three-tier Official Competition Hierarchy.
    
    Returns:
        (comparison_code, reason_string)
        where comparison_code:
         1: Model A is strictly better than Model B
        -1: Model B is strictly better than Model A
         0: Truly tied across all 3 priorities
    """
    if not m_a.get("eligible", False) and not m_b.get("eligible", False):
        return 0, "Both models ineligible"
    if not m_a.get("eligible", False):
        return -1, "Model A is ineligible"
    if not m_b.get("eligible", False):
        return 1, "Model B is ineligible"

    # PRIORITY 1: Shapiro-Wilk W_avg across X, Y, Z, Clock (Higher is better)
    w_a = float(m_a["priority_1"]["W"]["average"])
    w_b = float(m_b["priority_1"]["W"]["average"])

    if abs(w_a - w_b) > tie_tolerance:
        if w_a > w_b:
            return 1, f"Priority 1: Higher Shapiro-Wilk W_avg ({w_a:.6f} vs {w_b:.6f})"
        else:
            return -1, f"Priority 1: Lower Shapiro-Wilk W_avg ({w_a:.6f} vs {w_b:.6f})"

    # PRIORITY 2: Tied on Priority 1! Compare residual bias |mean| and std (Lower is better)
    mean_a = float(m_a["priority_2"]["mean"]["aggregate"])
    mean_b = float(m_b["priority_2"]["mean"]["aggregate"])

    if abs(mean_a - mean_b) > tie_tolerance:
        if mean_a < mean_b:
            return 1, f"Priority 2: Lower aggregate residual bias ({mean_a:.6f} vs {mean_b:.6f}) after Priority 1 tie"
        else:
            return -1, f"Priority 2: Higher aggregate residual bias ({mean_a:.6f} vs {mean_b:.6f}) after Priority 1 tie"

    std_a = float(m_a["priority_2"]["std"]["aggregate"])
    std_b = float(m_b["priority_2"]["std"]["aggregate"])

    if abs(std_a - std_b) > tie_tolerance:
        if std_a < std_b:
            return 1, f"Priority 2: Lower aggregate residual std ({std_a:.6f} vs {std_b:.6f}) after Priority 1 and Mean tie"
        else:
            return -1, f"Priority 2: Higher aggregate residual std ({std_a:.6f} vs {std_b:.6f}) after Priority 1 and Mean tie"

    # PRIORITY 3: Tied on Priority 1 and 2! Compare Q-Q outlier count (Fewer is better)
    outliers_a = int(m_a["priority_3"]["total_outliers"])
    outliers_b = int(m_b["priority_3"]["total_outliers"])

    if outliers_a != outliers_b:
        if outliers_a < outliers_b:
            return 1, f"Priority 3: Fewer Q-Q outliers ({outliers_a} vs {outliers_b}) after Priority 1 and 2 tie"
        else:
            return -1, f"Priority 3: More Q-Q outliers ({outliers_a} vs {outliers_b}) after Priority 1 and 2 tie"

    disc_a = float(m_a["priority_3"]["aggregate_max_discrepancy"])
    disc_b = float(m_b["priority_3"]["aggregate_max_discrepancy"])

    if abs(disc_a - disc_b) > tie_tolerance:
        if disc_a < disc_b:
            return 1, f"Priority 3: Lower Q-Q discrepancy ({disc_a:.6f} vs {disc_b:.6f})"
        else:
            return -1, f"Priority 3: Higher Q-Q discrepancy ({disc_a:.6f} vs {disc_b:.6f})"

    return 0, "Identical performance across all three official priorities"


class CalibrationPipeline:
    """Orchestrates Phase A Model Calibration across all satellites using the Official Evaluation Hierarchy."""

    def __init__(
        self,
        registry: Optional[SatelliteModelRegistry] = None,
        artifacts_dir: Union[str, Path] = "models/registry/artifacts",
        reports_dir: Union[str, Path] = "reports/calibration",
        primary_metric: str = "shapiro_w_avg",
        selection_policy: str = "official_competition",
        candidate_models: Optional[List[str]] = None,
        alpha: float = 0.05,
        tie_tolerance: float = 1e-4,
    ):
        self.registry = registry or SatelliteModelRegistry()
        self.artifacts_dir = Path(artifacts_dir)
        self.reports_dir = Path(reports_dir)
        self.primary_metric = primary_metric
        self.selection_policy = selection_policy
        self.alpha = alpha
        self.tie_tolerance = tie_tolerance
        self.candidate_models = candidate_models or [
            "persistence",
            "harmonic_ridge",
            "random_forest",
            "random_forest_srp",
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
        """Executes Phase A calibration driven strictly by the Official Competition Hierarchy."""
        now = datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        run_id = run_id or f"calib_{timestamp_str}"
        report_dir = self.reports_dir / run_id
        qq_dir = report_dir / "qq_data"
        qq_dir.mkdir(parents=True, exist_ok=True)
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

        calibration_summary: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp": now.isoformat(),
            "selection_policy": self.selection_policy,
            "primary_metric": self.primary_metric,
            "alpha": self.alpha,
            "satellites_evaluated": common_sats,
            "candidate_models": self.candidate_models,
            "satellite_winners": {},
            "detailed_scores": {},
        }

        comparison_rows: List[Dict[str, Any]] = []
        detailed_rows: List[Dict[str, Any]] = []

        # 3. Train, predict, and evaluate eligible models per satellite
        for sat_id in common_sats:
            sat_train = train_series[sat_id]
            sat_test = test_series[sat_id]
            test_times = pd.to_datetime(sat_test["utc_time"]).reset_index(drop=True)
            test_actuals = sat_test[TARGET_COLS].to_numpy(dtype=np.float64)

            # Fingerprint of training data (strict train/test separation)
            train_bytes = sat_train[TARGET_COLS].to_numpy().tobytes()
            train_hash = hashlib.sha256(train_bytes).hexdigest()[:16]

            candidate_evaluations: Dict[str, Dict[str, Any]] = {}
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

                    # Evaluate using Official Competition Hierarchy
                    eval_res = evaluate_residuals_official_hierarchy(
                        actual=test_actuals,
                        predicted=preds,
                        orbit_class=sat_id,
                        timestamps=test_times,
                        alpha=self.alpha,
                    )

                    if not eval_res.get("eligible", True):
                        logger.warning(f"Candidate {model_name} on {sat_id} ineligible: {eval_res.get('reason')}")
                        continue

                    candidate_evaluations[model_name] = eval_res
                    trained_models[model_name] = model_inst

                    # Save machine-readable Q-Q data
                    qq_file = qq_dir / f"{sat_id}_{model_name}_qq.json"
                    qq_file.write_text(
                        json.dumps(eval_res["priority_3"]["qq_details"], indent=2),
                        encoding="utf-8",
                    )

                    # Detailed per-target rows for detailed_statistical_results.csv
                    p1 = eval_res["priority_1"]
                    p2 = eval_res["priority_2"]
                    supp = eval_res["supplementary"]
                    for t_name, alias in zip(TARGET_COLS, ["X", "Y", "Z", "Clock"]):
                        t_m = eval_res["per_target"][t_name]
                        detailed_rows.append({
                            "run_id": run_id,
                            "satellite_id": sat_id,
                            "model": model_name,
                            "target": t_name,
                            "n_samples": p1["sample_count"][alias],
                            "shapiro_w": p1["W"][alias],
                            "p_value": p1["p_value"][alias],
                            "hypothesis_result": p1["hypothesis_result"][alias],
                            "alpha": self.alpha,
                            "mean": p2["mean"][alias],
                            "std": p2["std"][alias],
                            "mae": t_m["mae"],
                            "rmse": t_m["rmse"],
                            "qq_outlier_count": t_m["qq_outlier_count"],
                        })

                except Exception as exc:
                    logger.warning(f"Failed evaluation for {model_name} on {sat_id}: {exc}")
                    continue

            if not candidate_evaluations:
                logger.error(f"No candidate model succeeded for satellite {sat_id}")
                continue

            # 4. Official Competition Hierarchical Model Selection per Satellite
            # Sort all candidate models using the official 3-tier comparator
            def candidate_sort_key(m_name: str) -> Any:
                m_info = candidate_evaluations[m_name]
                # Negative W_avg for descending sort (higher W is better)
                w_score = -float(m_info["priority_1"]["W"]["average"])
                # Positive mean/std/outliers for ascending sort (lower is better)
                mean_score = float(m_info["priority_2"]["mean"]["aggregate"])
                std_score = float(m_info["priority_2"]["std"]["aggregate"])
                outlier_score = int(m_info["priority_3"]["total_outliers"])
                disc_score = float(m_info["priority_3"]["aggregate_max_discrepancy"])
                return (w_score, mean_score, std_score, outlier_score, disc_score)

            sorted_candidates = sorted(candidate_evaluations.keys(), key=candidate_sort_key)
            winner_name = sorted_candidates[0]
            winner_eval = candidate_evaluations[winner_name]
            winner_model = trained_models[winner_name]

            # Determine explicit selection reason relative to runner up
            if len(sorted_candidates) > 1:
                runner_up = sorted_candidates[1]
                _, reason = compare_models_hierarchical(
                    winner_eval,
                    candidate_evaluations[runner_up],
                    tie_tolerance=self.tie_tolerance,
                )
                selection_reason = f"{winner_name} won vs {runner_up}: {reason}"
            else:
                selection_reason = f"Only eligible model: {winner_name}"

            # Log candidate comparison rows
            for rank_idx, m_name in enumerate(sorted_candidates, start=1):
                c_eval = candidate_evaluations[m_name]
                cp1 = c_eval["priority_1"]
                cp2 = c_eval["priority_2"]
                cp3 = c_eval["priority_3"]
                csupp = c_eval["supplementary"]
                comparison_rows.append({
                    "run_id": run_id,
                    "satellite_id": sat_id,
                    "model": m_name,
                    "rank": rank_idx,
                    "is_winner": bool(m_name == winner_name),
                    "selection_reason": selection_reason if m_name == winner_name else f"Rank {rank_idx}",
                    "W_avg": cp1["W"]["average"],
                    "W_X": cp1["W"]["X"],
                    "W_Y": cp1["W"]["Y"],
                    "W_Z": cp1["W"]["Z"],
                    "W_Clock": cp1["W"]["Clock"],
                    "p_avg": cp1["p_value"]["average"],
                    "p_X": cp1["p_value"]["X"],
                    "p_Y": cp1["p_value"]["Y"],
                    "p_Z": cp1["p_value"]["Z"],
                    "p_Clock": cp1["p_value"]["Clock"],
                    "H_X": cp1["hypothesis_result"]["X"],
                    "H_Y": cp1["hypothesis_result"]["Y"],
                    "H_Z": cp1["hypothesis_result"]["Z"],
                    "H_Clock": cp1["hypothesis_result"]["Clock"],
                    "total_rejected_tests": cp1["total_rejected_tests"],
                    "mean_aggregate": cp2["mean"]["aggregate"],
                    "mean_X": cp2["mean"]["X"],
                    "mean_Y": cp2["mean"]["Y"],
                    "mean_Z": cp2["mean"]["Z"],
                    "mean_Clock": cp2["mean"]["Clock"],
                    "std_aggregate": cp2["std"]["aggregate"],
                    "std_X": cp2["std"]["X"],
                    "std_Y": cp2["std"]["Y"],
                    "std_Z": cp2["std"]["Z"],
                    "std_Clock": cp2["std"]["Clock"],
                    "qq_outliers_total": cp3["total_outliers"],
                    "orbit_3d_vector_mae_m": csupp["orbit_3d_vector_mae_m"],
                    "clock_mae_m": csupp["clock_mae_m"],
                    "sisre_mean_m": csupp["sisre_mean_m"],
                })

            # 5. Persist winning model artifact
            artifact_filename = f"{sat_id}_{winner_name}.pt" if "moe" in winner_name or "nhits" in winner_name or "bilstm" in winner_name or "transformer" in winner_name else f"{sat_id}_{winner_name}.joblib"
            artifact_dest = self.artifacts_dir / artifact_filename
            winner_model.save(artifact_dest)

            # Also persist into nested satellite artifact layout
            sat_artifact_dir = get_satellite_artifact_dir(sat_id, self.artifacts_dir)
            ext = ".pt" if "moe" in winner_name or "nhits" in winner_name or "bilstm" in winner_name or "transformer" in winner_name else ".joblib"
            nested_dest = sat_artifact_dir / f"model{ext}"
            winner_model.save(nested_dest)

            # 6. Update Persistent Satellite Model Registry
            winning_score = float(winner_eval["priority_1"]["W"]["average"])
            candidate_w_scores = {
                m: float(candidate_evaluations[m]["priority_1"]["W"]["average"])
                for m in candidate_evaluations
            }

            winner_meta = winner_model.get_metadata()
            winner_phys_features = getattr(winner_meta, "physics_features", [])
            feature_manifest = getattr(winner_model, "feature_manifest", None)
            feat_dict = feature_manifest.to_dict() if feature_manifest else {"features": getattr(winner_meta, "features", [])}

            (sat_artifact_dir / "metadata.json").write_text(
                json.dumps(asdict(winner_meta), indent=2, default=str), encoding="utf-8"
            )
            (sat_artifact_dir / "feature_manifest.json").write_text(
                json.dumps(feat_dict, indent=2, default=str), encoding="utf-8"
            )
            (sat_artifact_dir / "evaluation.json").write_text(
                json.dumps(winner_eval, indent=2, default=str), encoding="utf-8"
            )

            selection = self.registry.register_calibration_winner(
                satellite_id=sat_id,
                winner_model=winner_name,
                score=winning_score,
                candidate_scores=candidate_w_scores,
                training_dataset_hash=train_hash,
                model_artifact=str(artifact_dest),
                primary_metric="shapiro_w_avg",
                selection_policy=self.selection_policy,
                winning_priority_1=winner_eval["priority_1"],
                winning_priority_2=winner_eval["priority_2"],
                winning_priority_3=winner_eval["priority_3"],
                candidate_results={
                    m: {
                        "priority_1": candidate_evaluations[m]["priority_1"],
                        "priority_2": candidate_evaluations[m]["priority_2"],
                        "priority_3": {
                            "total_outliers": candidate_evaluations[m]["priority_3"]["total_outliers"],
                            "aggregate_max_discrepancy": candidate_evaluations[m]["priority_3"]["aggregate_max_discrepancy"],
                        },
                        "supplementary": candidate_evaluations[m]["supplementary"],
                    }
                    for m in candidate_evaluations
                },
                supplementary_diagnostics=winner_eval["supplementary"],
                physics_features=winner_phys_features,
                orbit_state_source="nominal_approximation",
                orbit_type="GEO" if "GEO" in sat_id.upper() else "MEO",
                use_ric=bool(getattr(winner_model, "use_ric", False)),
                use_srp=bool(getattr(winner_model, "enable_srp", False) or getattr(winner_model, "use_srp", False)),
                cadence_minutes=15.0,
                feature_manifest=feat_dict,
            )

            calibration_summary["satellite_winners"][sat_id] = {
                "selected_model": selection.selected_model,
                "selection_mode": selection.selection_mode,
                "selection_policy": self.selection_policy,
                "selection_reason": selection_reason,
                "priority_1_W_avg": winning_score,
                "priority_1": winner_eval["priority_1"],
                "priority_2": winner_eval["priority_2"],
                "priority_3": {
                    "total_outliers": winner_eval["priority_3"]["total_outliers"],
                    "aggregate_max_discrepancy": winner_eval["priority_3"]["aggregate_max_discrepancy"],
                },
                "supplementary_diagnostics": winner_eval["supplementary"],
                "physics_features": winner_phys_features,
                "orbit_state_source": "nominal_approximation",
                "candidate_scores": candidate_w_scores,
                "artifact": str(artifact_dest),
            }
            calibration_summary["detailed_scores"][sat_id] = candidate_evaluations

        # 7. Write Machine-Readable Calibration Reports
        summary_file = report_dir / "summary.json"
        summary_file.write_text(json.dumps(calibration_summary, indent=2, default=str), encoding="utf-8")

        if comparison_rows:
            comp_df = pd.DataFrame(comparison_rows)
            comp_df.to_csv(report_dir / "model_comparison.csv", index=False)
            # Retain legacy filename as well for backward compatibility
            comp_df.to_csv(report_dir / "satellite_model_comparison.csv", index=False)

        if detailed_rows:
            det_df = pd.DataFrame(detailed_rows)
            det_df.to_csv(report_dir / "detailed_statistical_results.csv", index=False)
            # Retain legacy filename as well
            det_df.to_csv(report_dir / "detailed_metrics.csv", index=False)

        eligibility_data = {
            s: {m: e.to_dict() for m, e in m_dict.items()}
            for s, m_dict in eligibility_matrix.items()
        }
        (report_dir / "eligibility.json").write_text(json.dumps(eligibility_data, indent=2), encoding="utf-8")

        config_data = {
            "selection_policy": self.selection_policy,
            "primary_metric": self.primary_metric,
            "alpha": self.alpha,
            "tie_tolerance": self.tie_tolerance,
            "candidate_models": self.candidate_models,
            "artifacts_dir": str(self.artifacts_dir),
            "reports_dir": str(self.reports_dir),
        }
        (report_dir / "configuration.json").write_text(json.dumps(config_data, indent=2), encoding="utf-8")

        calibration_summary["status"] = "success"
        calibration_summary["report_dir"] = str(report_dir)
        return calibration_summary

    def train_single_satellite(
        self,
        dataset: SatelliteDataset,
        test_data: Union[pd.DataFrame, SatelliteDataset],
        use_ric: bool = False,
        use_srp: bool = False,
        target_cadence_minutes: Optional[float] = None,
        candidate_models: Optional[List[str]] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calibrates and registers models for a single uploaded satellite dataset.

        Evaluates candidate models strictly on test observations using the Official
        Competition Evaluation Hierarchy. Persists model artifacts to
        models/registry/artifacts/satellites/<sat_id>/ and atomically updates the registry.
        """
        sat_id = dataset.satellite_id
        orbit_type = dataset.orbit_type
        sat_train = dataset.dataframe.copy()

        if isinstance(test_data, SatelliteDataset):
            sat_test = test_data.dataframe.copy()
        elif isinstance(test_data, (str, Path)):
            sat_test = normalize_dataframe_columns(pd.read_csv(test_data))
        else:
            sat_test = normalize_dataframe_columns(test_data.copy())

        test_times = pd.to_datetime(sat_test["utc_time"]).reset_index(drop=True)
        test_actuals = sat_test[TARGET_COLS].to_numpy(dtype=np.float64)

        cadence = (
            target_cadence_minutes
            or (dataset.sampling.target_cadence_minutes if dataset.sampling else None)
            or 15.0
        )

        train_bytes = sat_train[TARGET_COLS].to_numpy().tobytes()
        train_hash = hashlib.sha256(train_bytes).hexdigest()[:16]

        candidate_evaluations: Dict[str, Dict[str, Any]] = {}
        trained_models: Dict[str, ForecastModel] = {}
        models_to_eval = list(candidate_models) if candidate_models else list(self.candidate_models)

        for model_name in models_to_eval:
            try:
                kwargs: Dict[str, Any] = {
                    "orbit_class": orbit_type,
                    "satellite_id": sat_id,
                    "cadence_minutes": cadence,
                }
                if "random_forest" in model_name:
                    kwargs["use_ric"] = use_ric or ("ric" in model_name)
                    kwargs["enable_srp"] = use_srp or ("srp" in model_name)
                elif "harmonic_ridge" in model_name:
                    kwargs["use_ric"] = use_ric or ("ric" in model_name)
                    kwargs["use_srp"] = use_srp or ("srp" in model_name)

                model_inst = create_model(model_name, **kwargs)
                model_inst.fit(sat_train)

                preds = model_inst.predict(sat_train, test_times)

                eval_res = evaluate_residuals_official_hierarchy(
                    actual=test_actuals,
                    predicted=preds,
                    orbit_class=orbit_type,
                    timestamps=test_times,
                    alpha=self.alpha,
                )

                if not eval_res.get("eligible", True):
                    continue

                candidate_evaluations[model_name] = eval_res
                trained_models[model_name] = model_inst
            except Exception as exc:
                logger.warning(f"Evaluation failed for candidate {model_name} on {sat_id}: {exc}")
                continue

        if not candidate_evaluations:
            raise ValueError(f"No candidate model succeeded calibration for satellite '{sat_id}'")

        def candidate_sort_key(m_name: str) -> Any:
            m_info = candidate_evaluations[m_name]
            w_score = -float(m_info["priority_1"]["W"]["average"])
            mean_score = float(m_info["priority_2"]["mean"]["aggregate"])
            std_score = float(m_info["priority_2"]["std"]["aggregate"])
            outlier_score = int(m_info["priority_3"]["total_outliers"])
            disc_score = float(m_info["priority_3"]["aggregate_max_discrepancy"])
            return (w_score, mean_score, std_score, outlier_score, disc_score)

        sorted_candidates = sorted(candidate_evaluations.keys(), key=candidate_sort_key)
        winner_name = sorted_candidates[0]
        winner_eval = candidate_evaluations[winner_name]
        winner_model = trained_models[winner_name]

        if len(sorted_candidates) > 1:
            runner_up = sorted_candidates[1]
            _, reason = compare_models_hierarchical(
                winner_eval,
                candidate_evaluations[runner_up],
                tie_tolerance=self.tie_tolerance,
            )
            selection_reason = f"{winner_name} won vs {runner_up}: {reason}"
        else:
            selection_reason = f"Only eligible model: {winner_name}"

        # Per-satellite dedicated artifact directory
        sat_artifact_dir = get_satellite_artifact_dir(sat_id, self.artifacts_dir)
        is_torch = any(k in winner_name for k in ("moe", "nhits", "bilstm", "transformer"))
        ext = ".pt" if is_torch else ".joblib"
        dedicated_artifact = sat_artifact_dir / f"model{ext}"
        flat_artifact = self.artifacts_dir / f"{sat_id}_{winner_name}{ext}"

        winner_model.save(dedicated_artifact)
        winner_model.save(flat_artifact)

        winner_meta = winner_model.get_metadata()
        winner_phys_features = getattr(winner_meta, "physics_features", [])
        feature_manifest = getattr(winner_model, "feature_manifest", None)
        feat_dict = feature_manifest.to_dict() if feature_manifest else {"features": getattr(winner_meta, "features", [])}

        (sat_artifact_dir / "metadata.json").write_text(
            json.dumps(asdict(winner_meta), indent=2, default=str), encoding="utf-8"
        )
        (sat_artifact_dir / "feature_manifest.json").write_text(
            json.dumps(feat_dict, indent=2, default=str), encoding="utf-8"
        )
        (sat_artifact_dir / "evaluation.json").write_text(
            json.dumps(winner_eval, indent=2, default=str), encoding="utf-8"
        )

        winning_score = float(winner_eval["priority_1"]["W"]["average"])
        candidate_w_scores = {
            m: float(candidate_evaluations[m]["priority_1"]["W"]["average"])
            for m in candidate_evaluations
        }

        selection = self.registry.register_calibration_winner(
            satellite_id=sat_id,
            winner_model=winner_name,
            score=winning_score,
            candidate_scores=candidate_w_scores,
            training_dataset_hash=train_hash,
            model_artifact=str(dedicated_artifact),
            primary_metric="shapiro_w_avg",
            selection_policy=self.selection_policy,
            winning_priority_1=winner_eval["priority_1"],
            winning_priority_2=winner_eval["priority_2"],
            winning_priority_3=winner_eval["priority_3"],
            candidate_results={
                m: {
                    "priority_1": candidate_evaluations[m]["priority_1"],
                    "priority_2": candidate_evaluations[m]["priority_2"],
                    "priority_3": {
                        "total_outliers": candidate_evaluations[m]["priority_3"]["total_outliers"],
                        "aggregate_max_discrepancy": candidate_evaluations[m]["priority_3"]["aggregate_max_discrepancy"],
                    },
                    "supplementary": candidate_evaluations[m]["supplementary"],
                }
                for m in candidate_evaluations
            },
            supplementary_diagnostics=winner_eval["supplementary"],
            physics_features=winner_phys_features,
            orbit_state_source="nominal_approximation",
            orbit_type=orbit_type,
            use_ric=use_ric or ("ric" in winner_name),
            use_srp=use_srp or ("srp" in winner_name),
            cadence_minutes=cadence,
            feature_manifest=feat_dict,
        )

        return {
            "satellite_id": sat_id,
            "orbit_type": orbit_type,
            "selected_model": selection.selected_model,
            "selection_mode": selection.selection_mode,
            "selection_reason": selection_reason,
            "winning_score": winning_score,
            "priority_1": winner_eval["priority_1"],
            "priority_2": winner_eval["priority_2"],
            "priority_3": winner_eval["priority_3"],
            "supplementary": winner_eval["supplementary"],
            "candidate_scores": candidate_w_scores,
            "model_artifact": str(dedicated_artifact),
            "flat_artifact": str(flat_artifact),
            "feature_manifest": feat_dict,
        }

