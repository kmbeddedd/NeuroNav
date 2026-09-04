"""Compatibility layer re-exporting validation helpers from src.forecasting.validation."""
from __future__ import annotations

from src.forecasting.validation import (
    SATELLITE_COLS,
    TIME_COLS,
    detect_satellite_col,
    detect_time_col,
)

__all__ = [
    "SATELLITE_COLS",
    "TIME_COLS",
    "detect_satellite_col",
    "detect_time_col",
]
