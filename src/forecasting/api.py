"""Public Backend API & Service Interface for NeuroNav.

Provides clean, decoupled, headless Python interfaces for the frontend developer.
Exposes data validation, model calibration, satellite model inspection, manual overrides,
independent inference, and model metadata without any GUI dependencies.
Supports both single-satellite workflows (independent upload/calibration) and
multi-satellite batch pipelines.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.forecasting.models import MODEL_REGISTRY, create_model, get_available_model_names
from src.forecasting.pipeline import (
    CalibrationPipeline,
    evaluate_residuals_official_hierarchy,
)
from src.forecasting.registry import (
    SatelliteModelRegistry,
    SatelliteSelection,
    get_satellite_artifact_dir,
)
from src.forecasting.router import PredictionRouter
from src.forecasting.validation import (
    SatelliteDataset,
    infer_orbit_type,
    load_telemetry_source,
    normalize_dataframe_columns,
    validate_dataset as run_validation,
    validate_satellite_dataset as run_satellite_validation,
)

# Global singletons / default instances
_default_registry = SatelliteModelRegistry()
_default_router = PredictionRouter(registry=_default_registry)
_default_pipeline = CalibrationPipeline(registry=_default_registry)


def validate_dataset(
    data: Union[str, Path, pd.DataFrame, Dict[str, pd.DataFrame]],
    min_history_rows: int = 8,
    is_test_dataset: bool = False,
) -> Dict[str, Any]:
    """Validates an incoming telemetry dataset and returns a structured validation report."""
    result = run_validation(data, min_history_rows=min_history_rows, is_test_dataset=is_test_dataset)
    return result.to_dict()


def calibrate_models(
    train_data: Union[str, Path, pd.DataFrame, Dict[str, pd.DataFrame]],
    test_data: Union[str, Path, pd.DataFrame, Dict[str, pd.DataFrame]],
    primary_metric: str = "shapiro_w_avg",
    selection_policy: str = "official_competition",
    candidate_models: Optional[List[str]] = None,
    alpha: float = 0.05,
    tie_tolerance: float = 1e-4,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Phase A Calibration:

    Trains/evaluates candidate models across all satellites using the Official Competition Hierarchy:
    Priority 1 (Shapiro-Wilk W_avg) -> Priority 2 (Residual Mean & Std) -> Priority 3 (Q-Q Outliers).
    Persists winners to the registry and emits comprehensive machine-readable reports.
    """
    pipeline = CalibrationPipeline(
        registry=_default_registry,
        primary_metric=primary_metric,
        selection_policy=selection_policy,
        candidate_models=candidate_models,
        alpha=alpha,
        tie_tolerance=tie_tolerance,
    )
    return pipeline.run_calibration(train_data, test_data, run_id=run_id)


def predict_with_satellite_models(
    data: Union[str, Path, pd.DataFrame, Dict[str, pd.DataFrame]],
    horizon_steps: int = 96,
    step_interval_minutes: int = 15,
    compute_ric: bool = True,
) -> pd.DataFrame:
    """Phase B Forecasting:

    Executes inference independently for each satellite present in data using its
    selected model from the registry, tagging each row with model provenance.
    """
    return _default_router.predict(
        data=data,
        horizon_steps=horizon_steps,
        step_interval_minutes=step_interval_minutes,
        compute_ric=compute_ric,
    )


# ---------------------------------------------------------------------------
# Single-Satellite Endpoints
# ---------------------------------------------------------------------------

def register_satellite(
    satellite_id: str,
    orbit_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Registers metadata for a new satellite in the system."""
    resolved_orbit = orbit_type.upper() if orbit_type else infer_orbit_type(satellite_id)
    sat_dir = get_satellite_artifact_dir(satellite_id)
    reg_meta = {
        "satellite_id": satellite_id,
        "orbit_type": resolved_orbit,
        "custom_metadata": metadata or {},
    }
    meta_path = sat_dir / "satellite_registration.json"
    meta_path.write_text(json.dumps(reg_meta, indent=2), encoding="utf-8")
    return reg_meta


def validate_satellite_dataset(
    source: Union[str, Path, pd.DataFrame],
    satellite_id: Optional[str] = None,
    orbit_type: Optional[str] = None,
    target_cadence_minutes: Optional[float] = None,
    min_history_rows: int = 8,
) -> Dict[str, Any]:
    """Validates an independently uploaded dataset for a single satellite."""
    dataset = run_satellite_validation(
        source=source,
        satellite_id=satellite_id,
        orbit_type=orbit_type,
        target_cadence_minutes=target_cadence_minutes,
        min_history_rows=min_history_rows,
    )
    return dataset.to_dict()


def train_satellite(
    dataset: Union[SatelliteDataset, str, Path, pd.DataFrame],
    test_dataset: Union[SatelliteDataset, str, Path, pd.DataFrame],
    satellite_id: Optional[str] = None,
    orbit_type: Optional[str] = None,
    use_ric: bool = False,
    use_srp: bool = False,
    target_cadence_minutes: Optional[float] = None,
    candidate_models: Optional[List[str]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Trains and selects the winning model for an independently uploaded single-satellite dataset.

    Evaluates candidate models using the Official Competition Hierarchy and registers the winner.
    """
    if not isinstance(dataset, SatelliteDataset):
        ds_train = run_satellite_validation(
            source=dataset,
            satellite_id=satellite_id,
            orbit_type=orbit_type,
            target_cadence_minutes=target_cadence_minutes,
        )
    else:
        ds_train = dataset

    if not isinstance(test_dataset, SatelliteDataset) and not isinstance(test_dataset, pd.DataFrame):
        ds_test = run_satellite_validation(
            source=test_dataset,
            satellite_id=ds_train.satellite_id,
            orbit_type=ds_train.orbit_type,
            target_cadence_minutes=target_cadence_minutes,
        )
    else:
        ds_test = test_dataset

    return _default_pipeline.train_single_satellite(
        dataset=ds_train,
        test_data=ds_test,
        use_ric=use_ric,
        use_srp=use_srp,
        target_cadence_minutes=target_cadence_minutes,
        candidate_models=candidate_models,
        run_id=run_id,
    )


def evaluate_satellite(
    satellite_id: str,
    test_dataset: Union[SatelliteDataset, str, Path, pd.DataFrame],
) -> Dict[str, Any]:
    """Evaluates the registered model for a specific satellite against test ground truth."""
    model, selection = _default_router.get_assigned_model(satellite_id)
    if isinstance(test_dataset, SatelliteDataset):
        test_df = test_dataset.dataframe
    elif isinstance(test_dataset, pd.DataFrame):
        test_df = normalize_dataframe_columns(test_dataset.copy())
    else:
        test_df = normalize_dataframe_columns(pd.read_csv(test_dataset))

    test_times = pd.to_datetime(test_df["utc_time"]).reset_index(drop=True)
    target_cols = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]
    test_actuals = test_df[target_cols].to_numpy(dtype=float)

    preds = model.predict(test_df, test_times)
    orbit_type = getattr(selection, "orbit_type", satellite_id)

    eval_res = evaluate_residuals_official_hierarchy(
        actual=test_actuals,
        predicted=preds,
        orbit_class=orbit_type,
        timestamps=test_times,
    )
    return {
        "satellite_id": satellite_id,
        "selected_model": selection.selected_model,
        "evaluation": eval_res,
    }


def predict_satellite(
    satellite_id: str,
    history_data: Union[SatelliteDataset, str, Path, pd.DataFrame],
    horizon_steps: int = 96,
    step_interval_minutes: Optional[int] = None,
    compute_ric: bool = True,
) -> pd.DataFrame:
    """Performs forecast inference for a single satellite given historical lookback observations."""
    if isinstance(history_data, SatelliteDataset):
        hist_df = history_data.dataframe
    elif isinstance(history_data, pd.DataFrame):
        hist_df = normalize_dataframe_columns(history_data.copy())
    else:
        hist_df = normalize_dataframe_columns(pd.read_csv(history_data))

    return _default_router.predict_single_satellite(
        satellite_id=satellite_id,
        history_df=hist_df,
        horizon_steps=horizon_steps,
        step_interval_minutes=step_interval_minutes,
        compute_ric=compute_ric,
    )


def get_satellite_model(satellite_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves the assigned model details and artifact path for a satellite."""
    selection = _default_registry.get_selection(satellite_id)
    if selection is None:
        return None
    return {
        "satellite_id": satellite_id,
        "selected_model": selection.selected_model,
        "selection_mode": selection.selection_mode,
        "model_version": selection.model_version,
        "model_artifact": selection.model_artifact,
        "selection_score": selection.selection_score,
        "orbit_type": getattr(selection, "orbit_type", "UNKNOWN"),
        "use_ric": getattr(selection, "use_ric", False),
        "use_srp": getattr(selection, "use_srp", False),
        "cadence_minutes": getattr(selection, "cadence_minutes", 15.0),
        "feature_manifest": getattr(selection, "feature_manifest", {}),
    }


def get_satellite_metadata(satellite_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves full technical metadata and provenance for a satellite's active model."""
    sat_dir = get_satellite_artifact_dir(satellite_id)
    meta_path = sat_dir / "metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    selection = _default_registry.get_selection(satellite_id)
    if selection:
        return selection.to_dict()
    return None


# ---------------------------------------------------------------------------
# Registry & Override Endpoints
# ---------------------------------------------------------------------------

def get_satellite_selection(satellite_id: str) -> Optional[Dict[str, Any]]:
    """Returns the current model selection metadata for a given satellite, or None if unselected."""
    selection = _default_registry.get_selection(satellite_id)
    return selection.to_dict() if selection else None


def get_all_satellite_selections() -> Dict[str, Dict[str, Any]]:
    """Returns the active model assignments for all registered satellites."""
    selections = _default_registry.get_all_selections()
    return {sat_id: sel.to_dict() for sat_id, sel in selections.items()}


def set_satellite_model(
    satellite_id: str,
    model_name: str,
    model_version: str = "1.0.0",
    model_artifact: Optional[str] = None,
    reason: str = "Operator manual override",
) -> Dict[str, Any]:
    """Manually assigns a forecasting model to a satellite.
    
    Sets selection_mode='manual'. This assignment will NOT be overwritten by future
    automatic calibration runs until reset_to_automatic() is explicitly called.
    """
    selection = _default_registry.set_manual_selection(
        satellite_id=satellite_id,
        model_name=model_name,
        model_version=model_version,
        model_artifact=model_artifact,
        reason=reason,
    )
    return selection.to_dict()


def reset_to_automatic(satellite_id: str) -> Dict[str, Any]:
    """Resets a satellite's selection mode back to 'automatic'."""
    success = _default_registry.reset_to_automatic(satellite_id)
    return {
        "satellite_id": satellite_id,
        "reset_success": success,
        "current_selection": get_satellite_selection(satellite_id),
    }


def get_model_metadata(model_name: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Returns technical metadata for a specified model, or all available candidate models."""
    if model_name:
        inst = create_model(model_name)
        return inst.get_metadata().to_dict()

    all_meta = []
    for name in get_available_model_names():
        try:
            inst = create_model(name)
            all_meta.append(inst.get_metadata().to_dict())
        except Exception:
            continue
    return all_meta


# ---------------------------------------------------------------------------
# Reporting & Calibration Summary Endpoints
# ---------------------------------------------------------------------------

def _resolve_target_dir(base: Path, report_id: Optional[str] = None) -> Optional[Path]:
    """Resolves target directory, picking the most recently modified if report_id is None."""
    if not base.exists():
        return None
    if report_id:
        target = base / report_id
        return target if target.exists() else None
    subdirs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    return subdirs[0] if subdirs else None


def get_calibration_report(
    report_id: Optional[str] = None,
    reports_base_dir: Union[str, Path] = "reports/calibration",
) -> Optional[Dict[str, Any]]:
    """Retrieves a previously generated calibration report summary."""
    return get_calibration_summary(report_id, reports_base_dir)


def get_calibration_summary(
    report_id: Optional[str] = None,
    reports_base_dir: Union[str, Path] = "reports/calibration",
) -> Optional[Dict[str, Any]]:
    """Loads calibration summary JSON."""
    target_dir = _resolve_target_dir(Path(reports_base_dir), report_id)
    if not target_dir:
        return None

    summary_file = target_dir / "summary.json"
    if not summary_file.exists():
        return None

    return json.loads(summary_file.read_text(encoding="utf-8"))


def get_model_comparison(
    report_id: Optional[str] = None,
    reports_base_dir: Union[str, Path] = "reports/calibration",
) -> Optional[pd.DataFrame]:
    """Loads satellite-model comparison metrics as a pandas DataFrame."""
    target_dir = _resolve_target_dir(Path(reports_base_dir), report_id)
    if not target_dir:
        return None

    for filename in ("model_comparison.csv", "satellite_model_comparison.csv"):
        comp_file = target_dir / filename
        if comp_file.exists():
            return pd.read_csv(comp_file)

    return None


def get_detailed_statistical_results(
    report_id: Optional[str] = None,
    reports_base_dir: Union[str, Path] = "reports/calibration",
) -> Optional[pd.DataFrame]:
    """Loads granular per-target statistical metrics (W, p-value, H0, mean, std) as a DataFrame."""
    target_dir = _resolve_target_dir(Path(reports_base_dir), report_id)
    if not target_dir:
        return None

    for filename in ("detailed_statistical_results.csv", "detailed_metrics.csv"):
        det_file = target_dir / filename
        if det_file.exists():
            return pd.read_csv(det_file)

    return None


def get_qq_data(
    satellite_id: str,
    model_name: str,
    report_id: Optional[str] = None,
    reports_base_dir: Union[str, Path] = "reports/calibration",
) -> Optional[Dict[str, Any]]:
    """Loads machine-readable Q-Q plot quantiles and outlier analysis for a satellite and model."""
    target_dir = _resolve_target_dir(Path(reports_base_dir), report_id)
    if not target_dir:
        return None

    qq_file = target_dir / "qq_data" / f"{satellite_id}_{model_name}_qq.json"
    if not qq_file.exists():
        return None

    return json.loads(qq_file.read_text(encoding="utf-8"))
