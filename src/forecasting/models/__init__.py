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

class RandomForestSRPModel(RandomForestModel):
    """Random Forest variant with explicit SRP solar radiation physics features."""
    def __init__(self, name: str = "Random Forest + SRP", **kwargs):
        kwargs.setdefault("enable_srp", True)
        super().__init__(name=name, **kwargs)


class RandomForestRICModel(RandomForestModel):
    """Random Forest variant with explicit RIC coordinate features."""
    def __init__(self, name: str = "Random Forest + RIC", **kwargs):
        kwargs.setdefault("use_ric", True)
        super().__init__(name=name, **kwargs)


class HarmonicRidgeSRPModel(HarmonicRidgeModel):
    """Harmonic Ridge variant with explicit SRP physics features."""
    def __init__(self, name: str = "Harmonic Ridge + SRP", **kwargs):
        kwargs.setdefault("use_srp", True)
        super().__init__(name=name, **kwargs)


class HarmonicRidgeRICModel(HarmonicRidgeModel):
    """Harmonic Ridge variant with explicit RIC coordinate features."""
    def __init__(self, name: str = "Harmonic Ridge + RIC", **kwargs):
        kwargs.setdefault("use_ric", True)
        super().__init__(name=name, **kwargs)


# Canonical Model Registry
MODEL_REGISTRY: Dict[str, Type[ForecastModel]] = {
    "persistence": PersistenceModel,
    "harmonic_ridge": HarmonicRidgeModel,
    "harmonic_ridge_srp": HarmonicRidgeSRPModel,
    "harmonic_ridge_ric": HarmonicRidgeRICModel,
    "random_forest": RandomForestModel,
    "random_forest_srp": RandomForestSRPModel,
    "random_forest_ric": RandomForestRICModel,
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
    "harmonic_ridge_srp": "harmonic_ridge_srp",
    "ridge_srp": "harmonic_ridge_srp",
    "harmonic_ridge_ric": "harmonic_ridge_ric",
    "ridge_ric": "harmonic_ridge_ric",
    "random_forest": "random_forest",
    "rf": "random_forest",
    "random_forest_srp": "random_forest_srp",
    "rf_srp": "random_forest_srp",
    "random_forest_ric": "random_forest_ric",
    "rf_ric": "random_forest_ric",
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


import inspect


def create_model(model_name: str, **kwargs) -> ForecastModel:
    """Factory creating an instance of a registered forecasting model."""
    key = MODEL_ALIASES.get(model_name.lower().replace("-", "_").replace(" ", "_"))
    if key is None or key not in MODEL_REGISTRY:
        available = get_available_model_names()
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    model_cls = MODEL_REGISTRY[key]
    sig = inspect.signature(model_cls.__init__)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    if has_var_keyword:
        inst = model_cls(**kwargs)
    else:
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        inst = model_cls(**valid_kwargs)
        for k, v in kwargs.items():
            if k not in valid_kwargs and not hasattr(inst, k):
                try:
                    setattr(inst, k, v)
                except Exception:
                    pass

    return inst
