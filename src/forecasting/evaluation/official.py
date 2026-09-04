"""Stable imports for the official P1/P2/P3 competition evaluation.

The implementation remains byte-for-byte within the calibration module during this
behavior-preserving refactor.  This facade establishes the evaluation boundary without
duplicating selection logic.
"""

from src.forecasting.training.calibration import (
    compare_models_hierarchical,
    compute_metrics_for_residuals,
    evaluate_residuals_official_hierarchy,
    rank_candidates_hierarchically,
)

__all__ = [
    "compare_models_hierarchical",
    "compute_metrics_for_residuals",
    "evaluate_residuals_official_hierarchy",
    "rank_candidates_hierarchically",
]
