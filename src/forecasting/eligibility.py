"""Eligibility matrix evaluator for satellite x model candidate pairs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class ModelCapabilities:
    """Explicit declaration of model data/physics capabilities and requirements."""
    supports_irregular_timestamps: bool
    requires_regular_cadence: bool
    min_required_rows: int
    supports_feature_manifest: bool = True
    supports_nominal_physics: bool = True
    supports_provided_state: bool = True


MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    "persistence": ModelCapabilities(
        supports_irregular_timestamps=True,
        requires_regular_cadence=False,
        min_required_rows=4,
        supports_feature_manifest=False,
        supports_nominal_physics=False,
        supports_provided_state=False,
    ),
    "harmonic_ridge": ModelCapabilities(
        supports_irregular_timestamps=True,
        requires_regular_cadence=False,
        min_required_rows=12,
        supports_feature_manifest=True,
        supports_nominal_physics=True,
        supports_provided_state=True,
    ),
    "random_forest": ModelCapabilities(
        supports_irregular_timestamps=True,
        requires_regular_cadence=False,
        min_required_rows=12,
        supports_feature_manifest=True,
        supports_nominal_physics=True,
        supports_provided_state=True,
    ),
    "gaussian_process": ModelCapabilities(
        supports_irregular_timestamps=True,
        requires_regular_cadence=False,
        min_required_rows=8,
        supports_feature_manifest=False,
        supports_nominal_physics=False,
        supports_provided_state=False,
    ),
    "decoupled_clock": ModelCapabilities(
        supports_irregular_timestamps=True,
        requires_regular_cadence=False,
        min_required_rows=12,
        supports_feature_manifest=False,
        supports_nominal_physics=False,
        supports_provided_state=False,
    ),
    "bilstm_gru": ModelCapabilities(
        supports_irregular_timestamps=False,
        requires_regular_cadence=True,
        min_required_rows=16,
        supports_feature_manifest=False,
        supports_nominal_physics=False,
        supports_provided_state=False,
    ),
    "transformer": ModelCapabilities(
        supports_irregular_timestamps=False,
        requires_regular_cadence=True,
        min_required_rows=16,
        supports_feature_manifest=False,
        supports_nominal_physics=False,
        supports_provided_state=False,
    ),
    "geo_moe": ModelCapabilities(
        supports_irregular_timestamps=False,
        requires_regular_cadence=True,
        min_required_rows=16,
        supports_feature_manifest=False,
        supports_nominal_physics=False,
        supports_provided_state=False,
    ),
    "nhits": ModelCapabilities(
        supports_irregular_timestamps=False,
        requires_regular_cadence=True,
        min_required_rows=24,
        supports_feature_manifest=False,
        supports_nominal_physics=False,
        supports_provided_state=False,
    ),
}


@dataclass
class ModelEligibility:
    satellite: str
    model: str
    eligible: bool
    reason: Optional[str] = None
    min_required_rows: int = 12
    available_rows: int = 0
    supports_irregular_timestamps: bool = True
    requires_regular_cadence: bool = False
    supports_nominal_physics: bool = True
    supports_provided_state: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Minimum historical observations required by model type
MODEL_ROW_REQUIREMENTS = {k: v.min_required_rows for k, v in MODEL_CAPABILITIES.items()}


def check_model_eligibility(
    satellite_id: str,
    model_name: str,
    df: pd.DataFrame,
    enforce_cadence: bool = False,
    is_irregular: Optional[bool] = None,
) -> ModelEligibility:
    """Evaluates if a candidate model is eligible to be trained/evaluated on a satellite dataset."""
    model_key = model_name.lower().replace("-", "_").replace(" ", "_")
    caps = MODEL_CAPABILITIES.get(
        model_key,
        ModelCapabilities(
            supports_irregular_timestamps=True,
            requires_regular_cadence=False,
            min_required_rows=12,
        ),
    )
    required_rows = caps.min_required_rows
    available_rows = len(df)

    # 1. Sample count check
    if available_rows < required_rows:
        return ModelEligibility(
            satellite=satellite_id,
            model=model_name,
            eligible=False,
            reason=(
                f"Insufficient history: found {available_rows} rows, "
                f"model '{model_name}' requires at least {required_rows}"
            ),
            min_required_rows=required_rows,
            available_rows=available_rows,
            supports_irregular_timestamps=caps.supports_irregular_timestamps,
            requires_regular_cadence=caps.requires_regular_cadence,
            supports_nominal_physics=caps.supports_nominal_physics,
            supports_provided_state=caps.supports_provided_state,
        )

    # 2. Check cadence regularity requirement if enforced
    if enforce_cadence and is_irregular and caps.requires_regular_cadence:
        return ModelEligibility(
            satellite=satellite_id,
            model=model_name,
            eligible=False,
            reason=(
                f"Model '{model_name}' requires regular cadence, but dataset '{satellite_id}' "
                f"has irregular timestamps. Resample or choose a timestamp-tolerant model."
            ),
            min_required_rows=required_rows,
            available_rows=available_rows,
            supports_irregular_timestamps=caps.supports_irregular_timestamps,
            requires_regular_cadence=caps.requires_regular_cadence,
            supports_nominal_physics=caps.supports_nominal_physics,
            supports_provided_state=caps.supports_provided_state,
        )

    # 3. Check variance in targets (constant target cannot fit GP or neural models)
    target_cols = [c for c in ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"] if c in df.columns]
    if target_cols:
        stds = df[target_cols].std()
        if (stds == 0).all() and model_key in ("gaussian_process", "bilstm_gru", "transformer", "nhits"):
            return ModelEligibility(
                satellite=satellite_id,
                model=model_name,
                eligible=False,
                reason="Target columns exhibit zero variance across observations",
                min_required_rows=required_rows,
                available_rows=available_rows,
                supports_irregular_timestamps=caps.supports_irregular_timestamps,
                requires_regular_cadence=caps.requires_regular_cadence,
                supports_nominal_physics=caps.supports_nominal_physics,
                supports_provided_state=caps.supports_provided_state,
            )

    return ModelEligibility(
        satellite=satellite_id,
        model=model_name,
        eligible=True,
        min_required_rows=required_rows,
        available_rows=available_rows,
        supports_irregular_timestamps=caps.supports_irregular_timestamps,
        requires_regular_cadence=caps.requires_regular_cadence,
        supports_nominal_physics=caps.supports_nominal_physics,
        supports_provided_state=caps.supports_provided_state,
    )


def compute_eligibility_matrix(
    satellite_data: Dict[str, pd.DataFrame],
    candidate_models: List[str],
    enforce_cadence: bool = False,
) -> Dict[str, Dict[str, ModelEligibility]]:
    """Evaluates eligibility for all (satellite, model) combinations.
    
    Returns:
        Dict[satellite_id -> Dict[model_name -> ModelEligibility]]
    """
    matrix: Dict[str, Dict[str, ModelEligibility]] = {}
    for sat_id, df in satellite_data.items():
        matrix[sat_id] = {}
        for model in candidate_models:
            matrix[sat_id][model] = check_model_eligibility(
                sat_id, model, df, enforce_cadence=enforce_cadence
            )
    return matrix
