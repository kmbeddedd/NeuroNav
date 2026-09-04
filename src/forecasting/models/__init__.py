"""Forecasting model registry and factory."""
from __future__ import annotations

from typing import Dict, List, Type

from src.forecasting.base import ForecastModel
from src.forecasting.models.bilstm_gru import BiLSTMGRUModel
from src.forecasting.models.decoupled_clock import DecoupledClockModel
from src.forecasting.models.gaussian_process import GaussianProcessModel
from src.forecasting.models.geo_moe import GEOGatedMoEModel
from src.forecasting.models.harmonic_ridge import HarmonicRidgeModel
from src.forecasting.models.nhits import NHiTSModel
from src.forecasting.models.persistence import PersistenceModel
from src.forecasting.models.random_forest import RandomForestModel
from src.forecasting.models.transformer import TransformerModel

# Canonical Model Registry
MODEL_REGISTRY: Dict[str, Type[ForecastModel]] = {
    "persistence": PersistenceModel,
    "harmonic_ridge": HarmonicRidgeModel,
    "random_forest": RandomForestModel,
    "gaussian_process": GaussianProcessModel,
    "geo_moe": GEOGatedMoEModel,
    "bilstm_gru": BiLSTMGRUModel,
    "transformer": TransformerModel,
    "decoupled_clock": DecoupledClockModel,
    "nhits": NHiTSModel,
}

# Aliases for flexibility
MODEL_ALIASES: Dict[str, str] = {
    "persistence": "persistence",
    "harmonic_ridge": "harmonic_ridge",
    "harmonic": "harmonic_ridge",
    "ridge": "harmonic_ridge",
    "random_forest": "random_forest",
    "rf": "random_forest",
    "gaussian_process": "gaussian_process",
    "gp": "gaussian_process",
    "geo_moe": "geo_moe",
    "geo_gated_moe": "geo_moe",
    "moe": "geo_moe",
    "bilstm_gru": "bilstm_gru",
    "bilstm": "bilstm_gru",
    "transformer": "transformer",
    "hybrid_transformer": "transformer",
    "decoupled_clock": "decoupled_clock",
    "decoupled": "decoupled_clock",
    "nhits": "nhits",
    "n-hits": "nhits",
}


def get_available_model_names() -> List[str]:
    """Returns the list of canonical registered model identifiers."""
    return sorted(MODEL_REGISTRY.keys())


def create_model(model_name: str, **kwargs) -> ForecastModel:
    """Factory creating an instance of a registered forecasting model."""
    key = MODEL_ALIASES.get(model_name.lower().replace("-", "_").replace(" ", "_"))
    if key is None or key not in MODEL_REGISTRY:
        available = get_available_model_names()
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
    return MODEL_REGISTRY[key](**kwargs)
