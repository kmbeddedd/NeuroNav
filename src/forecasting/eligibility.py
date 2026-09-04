"""Eligibility matrix evaluator for satellite x model candidate pairs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class ModelEligibility:
    satellite: str
    model: str
    eligible: bool
    reason: Optional[str] = None
    min_required_rows: int = 12
    available_rows: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Minimum historical observations required by model type
MODEL_ROW_REQUIREMENTS = {
    "persistence": 4,
    "harmonic_ridge": 12,
    "random_forest": 12,
    "gaussian_process": 8,
    "bilstm_gru": 16,
    "transformer": 16,
    "geo_moe": 16,
    "decoupled_clock": 12,
    "nhits": 24,
    "patchtst": 24,
}


def check_model_eligibility(
    satellite_id: str,
    model_name: str,
    df: pd.DataFrame,
) -> ModelEligibility:
    """Evaluates if a candidate model is eligible to be trained/evaluated on a satellite dataset."""
    model_key = model_name.lower().replace("-", "_").replace(" ", "_")
    required_rows = MODEL_ROW_REQUIREMENTS.get(model_key, 12)
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
        )

    # 2. Check variance in targets (constant target cannot fit GP or neural models)
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
            )

    return ModelEligibility(
        satellite=satellite_id,
        model=model_name,
        eligible=True,
        min_required_rows=required_rows,
        available_rows=available_rows,
    )


def compute_eligibility_matrix(
    satellite_data: Dict[str, pd.DataFrame],
    candidate_models: List[str],
) -> Dict[str, Dict[str, ModelEligibility]]:
    """Evaluates eligibility for all (satellite, model) combinations.
    
    Returns:
        Dict[satellite_id -> Dict[model_name -> ModelEligibility]]
    """
    matrix: Dict[str, Dict[str, ModelEligibility]] = {}
    for sat_id, df in satellite_data.items():
        matrix[sat_id] = {}
        for model in candidate_models:
            matrix[sat_id][model] = check_model_eligibility(sat_id, model, df)
    return matrix
