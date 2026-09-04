"""Compatibility layer re-exporting forecasting model registry from src.forecasting.models."""
from __future__ import annotations

from src.forecasting.models import (
    MODEL_REGISTRY,
    create_model,
    get_available_model_names,
)

# Aliases for GUI backwards compatibility
MODEL_ADAPTER_CLASSES = MODEL_REGISTRY
get_available_model_adapters = get_available_model_names

__all__ = [
    "MODEL_REGISTRY",
    "MODEL_ADAPTER_CLASSES",
    "create_model",
    "get_available_model_names",
    "get_available_model_adapters",
]
