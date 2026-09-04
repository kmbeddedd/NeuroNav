"""Public Backend API & Service Interface for NeuroNav.

Provides clean, decoupled, headless Python interfaces for the frontend developer.
Exposes data validation, model calibration, satellite model inspection, manual overrides,
independent inference, and model metadata without any GUI dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.forecasting.models import MODEL_REGISTRY, create_model, get_available_model_names
from src.forecasting.pipeline import CalibrationPipeline
from src.forecasting.registry import SatelliteModelRegistry, SatelliteSelection
from src.forecasting.router import PredictionRouter
from src.forecasting.validation import validate_dataset as run_validation


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


def get_calibration_report(
    report_id: Optional[str] = None,
    reports_base_dir: Union[str, Path] = "reports/calibration",
) -> Optional[Dict[str, Any]]:
    """Retrieves a previously generated calibration report summary."""
def _resolve_target_dir(base: Path, report_id: Optional[str] = None) -> Optional[Path]:
    """Resolves target directory, picking the most recently modified if report_id is None."""
    if not base.exists():
        return None
    if report_id:
        target = base / report_id
        return target if target.exists() else None
    subdirs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    return subdirs[0] if subdirs else None


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

