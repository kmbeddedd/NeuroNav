"""Command-line entry point for satellite-specific NeuroNav inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting import (  # noqa: E402
    get_satellite_model,
    predict_satellite,
    validate_satellite_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate telemetry and forecast with a registered NeuroNav model."
    )
    parser.add_argument("--satellite", required=True, help="Registered satellite ID, such as GEO")
    parser.add_argument("--history", required=True, help="Historical telemetry CSV")
    parser.add_argument("--orbit-type", default=None, help="Optional GEO, MEO, LEO, or UNKNOWN override")
    parser.add_argument("--horizon-steps", type=int, default=96, help="Number of forecast rows")
    parser.add_argument(
        "--step-interval-minutes",
        type=int,
        default=None,
        help="Override the cadence stored with the selected model",
    )
    parser.add_argument("--no-ric", action="store_true", help="Do not derive R/I/C predictions")
    parser.add_argument("--output", default=None, help="Optional destination CSV; defaults to stdout")
    return parser


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.horizon_steps < 1:
        parser.error("--horizon-steps must be at least 1")
    if args.step_interval_minutes is not None and args.step_interval_minutes < 1:
        parser.error("--step-interval-minutes must be at least 1")

    validation = validate_satellite_dataset(
        args.history,
        satellite_id=args.satellite,
        orbit_type=args.orbit_type,
    )
    selection = get_satellite_model(args.satellite)
    if selection is None:
        parser.error(
            f"satellite '{args.satellite}' has no active model; calibrate or register a selection first"
        )

    forecast = predict_satellite(
        satellite_id=args.satellite,
        history_data=args.history,
        horizon_steps=args.horizon_steps,
        step_interval_minutes=args.step_interval_minutes,
        compute_ric=not args.no_ric,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(output_path, index=False)
        print(
            f"Wrote {len(forecast)} forecasts for {validation['satellite_id']} "
            f"using {selection['selected_model']} to {output_path}"
        )
    else:
        forecast.to_csv(sys.stdout, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

