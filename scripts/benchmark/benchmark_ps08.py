"""Compatibility entry point for the historical PS-08 benchmark."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ps08.benchmark import *  # noqa: F401,F403
from research.ps08.benchmark import (
    _compute_regime_probability,
    _fit_causal_baseline,
    _fit_regime_detector,
    _physical_history_tensor,
    _physical_query_features,
    _rolling_backtest_geo,
    main,
)


if __name__ == "__main__":
    main()
