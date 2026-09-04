"""NeuroNav Satellite-Specific Forecasting Package."""
from __future__ import annotations

from src.forecasting.api import (
    calibrate_models,
    get_all_satellite_selections,
    get_calibration_report,
    get_detailed_statistical_results,
    get_model_comparison,
    get_model_metadata,
    get_qq_data,
    get_satellite_selection,
    predict_with_satellite_models,
    reset_to_automatic,
    set_satellite_model,
    validate_dataset,
)
from src.forecasting.base import ForecastModel, ModelMetadata
from src.forecasting.eligibility import compute_eligibility_matrix
from src.forecasting.models import MODEL_REGISTRY, create_model, get_available_model_names
from src.forecasting.pipeline import (
    CalibrationPipeline,
    compare_models_hierarchical,
    compute_metrics_for_residuals,
    evaluate_residuals_official_hierarchy,
)
from src.forecasting.registry import (
    CorruptedRegistryError,
    RegistryError,
    SatelliteModelRegistry,
    SatelliteSelection,
)
from src.forecasting.router import (
    ModelArtifactError,
    NoModelSelectionError,
    PredictionRouter,
    RoutingError,
)
from src.forecasting.validation import ValidationIssue, ValidationResult

__all__ = [
    "ForecastModel",
    "ModelMetadata",
    "SatelliteSelection",
    "SatelliteModelRegistry",
    "PredictionRouter",
    "CalibrationPipeline",
    "ValidationResult",
    "ValidationIssue",
    "RoutingError",
    "NoModelSelectionError",
    "ModelArtifactError",
    "RegistryError",
    "CorruptedRegistryError",
    "MODEL_REGISTRY",
    "create_model",
    "get_available_model_names",
    "compute_metrics_for_residuals",
    "compute_eligibility_matrix",
    "validate_dataset",
    "calibrate_models",
    "predict_with_satellite_models",
    "get_satellite_selection",
    "get_all_satellite_selections",
    "set_satellite_model",
    "reset_to_automatic",
    "evaluate_residuals_official_hierarchy",
    "compare_models_hierarchical",
    "get_calibration_report",
    "get_model_comparison",
    "get_detailed_statistical_results",
    "get_qq_data",
    "get_model_metadata",
]
