"""Data validation engine for GNSS satellite telemetry.

Performs deterministic, non-crashing schema, cadence, and integrity validation
and returns structured machine-readable reports.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Canonical column names used internally
TARGET_COLS_INTERNAL = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]

COLUMN_ALIASES = {
    "utc_time": "utc_time",
    "timestamp": "utc_time",
    "time": "utc_time",
    "x_error (m)": "x_error_m",
    "error_x": "x_error_m",
    "x_error": "x_error_m",
    "y_error (m)": "y_error_m",
    "y_error": "y_error_m",
    "error_y": "y_error_m",
    "z_error (m)": "z_error_m",
    "z_error": "z_error_m",
    "error_z": "z_error_m",
    "satclockerror (m)": "clock_error_m",
    "satclockerror": "clock_error_m",
    "clock_error (m)": "clock_error_m",
    "clock_error": "clock_error_m",
    "error_clock": "clock_error_m",
    "satellite_id": "satellite_id",
    "satellite": "satellite_id",
    "sat_id": "satellite_id",
    "prn": "satellite_id",
}


@dataclass
class ValidationIssue:
    status: str = "invalid"
    stage: str = "data_validation"
    satellite: str = "general"
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SatelliteDataReport:
    satellite_id: str
    row_count: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duplicate_epochs: int = 0
    missing_values: Dict[str, int] = field(default_factory=dict)
    is_monotonic: bool = True
    estimated_cadence_minutes: Optional[float] = None
    irregular_cadence_count: int = 0
    valid: bool = True
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    is_valid: bool
    status: str  # "valid" or "invalid"
    stage: str = "data_validation"
    satellites_found: List[str] = field(default_factory=list)
    satellite_reports: Dict[str, SatelliteDataReport] = field(default_factory=dict)
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    normalized_data: Optional[Dict[str, pd.DataFrame]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "stage": self.stage,
            "satellites_found": self.satellites_found,
            "satellite_reports": {
                k: v.to_dict() for k, v in self.satellite_reports.items()
            },
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes DataFrame column names using standard aliases."""
    normalized = {}
    for col in df.columns:
        cleaned = " ".join(str(col).strip().lower().split())
        normalized[col] = COLUMN_ALIASES.get(cleaned, col)
    renamed = df.rename(columns=normalized)
    if "utc_time" in renamed.columns:
        renamed["utc_time"] = pd.to_datetime(renamed["utc_time"])
    return renamed


def load_telemetry_source(
    source: Union[str, Path, pd.DataFrame, Dict[str, pd.DataFrame]],
    default_satellite_id: str = "UNKNOWN",
    is_test_dataset: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Loads telemetry from a DataFrame, CSV path, or folder of series CSVs.
    
    Returns:
        Dict mapping satellite_id -> DataFrame of telemetry rows.
    """
    if isinstance(source, dict):
        return {
            sat_id: normalize_dataframe_columns(df.copy())
            for sat_id, df in source.items()
        }

    if isinstance(source, pd.DataFrame):
        df = normalize_dataframe_columns(source.copy())
        if "satellite_id" in df.columns:
            return {
                str(sat_id): sat_df.sort_values("utc_time").reset_index(drop=True)
                for sat_id, sat_df in df.groupby("satellite_id")
            }
        return {default_satellite_id: df}

    path = Path(source)
    if path.is_file():
        df = normalize_dataframe_columns(pd.read_csv(path))
        if "satellite_id" in df.columns:
            return {
                str(sat_id): sat_df.sort_values("utc_time").reset_index(drop=True)
                for sat_id, sat_df in df.groupby("satellite_id")
            }
        # Infer satellite from filename if possible (e.g. DATA_GEO_Train.csv -> GEO)
        stem = path.stem.upper()
        sat_id = default_satellite_id
        for candidate in ("GEO", "MEO-1", "MEO-2", "MEO2", "MEO1", "MEO"):
            if candidate in stem:
                sat_id = candidate.replace("MEO2", "MEO-2").replace("MEO1", "MEO-1")
                break
        return {sat_id: df}

    if path.is_dir():
        sat_data: Dict[str, pd.DataFrame] = {}
        all_csvs = sorted(path.glob("*.csv"))
        # Check if directory has train/test split files
        has_train_test = any("TRAIN" in f.stem.upper() or "TEST" in f.stem.upper() for f in all_csvs)

        for csv_file in all_csvs:
            stem = csv_file.stem.upper()
            if has_train_test:
                is_train_file = "TRAIN" in stem
                is_test_file = "TEST" in stem
                if is_test_dataset and not is_test_file:
                    continue
                if not is_test_dataset and not is_train_file and has_train_test:
                    continue

                sat_id = default_satellite_id
                if "GEO" in stem:
                    sat_id = "GEO"
                elif "MEO" in stem:
                    sat_id = "MEO-2" if ("2" in stem or "MEO2" in stem) else "MEO-1"
                else:
                    sat_id = stem.split("_")[0]
                df = normalize_dataframe_columns(pd.read_csv(csv_file))
                sat_data[sat_id] = df
            else:
                df = normalize_dataframe_columns(pd.read_csv(csv_file))
                if "satellite_id" in df.columns:
                    for s_id, s_df in df.groupby("satellite_id"):
                        sat_data[str(s_id)] = s_df.sort_values("utc_time").reset_index(drop=True)
                else:
                    sat_data[csv_file.stem] = df
        return sat_data

    raise ValueError(f"Unsupported telemetry source: {source}")


def validate_dataset(
    source: Union[str, Path, pd.DataFrame, Dict[str, pd.DataFrame]],
    min_history_rows: int = 8,
    require_all_targets: bool = True,
    is_test_dataset: bool = False,
) -> ValidationResult:
    """Performs rigorous data validation returning a structured ValidationResult.
    
    Never raises an uncaught exception; returns structured error objects for frontend.
    """
    errors: List[ValidationIssue] = []
    warnings: List[ValidationIssue] = []
    satellite_reports: Dict[str, SatelliteDataReport] = {}
    normalized_data: Dict[str, pd.DataFrame] = {}

    try:
        raw_series = load_telemetry_source(source, is_test_dataset=is_test_dataset)
    except Exception as exc:
        errors.append(ValidationIssue(
            status="invalid",
            stage="data_validation",
            satellite="all",
            reason=f"Failed to read data source: {exc}",
        ))
        return ValidationResult(
            is_valid=False,
            status="invalid",
            errors=errors,
            warnings=warnings,
        )

    if not raw_series:
        errors.append(ValidationIssue(
            status="invalid",
            stage="data_validation",
            satellite="all",
            reason="Data source is empty; no satellite series found",
        ))
        return ValidationResult(
            is_valid=False,
            status="invalid",
            errors=errors,
            warnings=warnings,
        )

    satellites_found = sorted(raw_series.keys())

    for sat_id, df in raw_series.items():
        sat_issues: List[str] = []
        sat_errors: List[ValidationIssue] = []
        row_count = len(df)

        # 1. Required Timestamp column
        if "utc_time" not in df.columns:
            msg = "Missing required timestamp column ('utc_time' or 'Timestamp')"
            sat_issues.append(msg)
            sat_errors.append(ValidationIssue(
                status="invalid",
                stage="data_validation",
                satellite=sat_id,
                reason=msg,
                details={"available_columns": list(df.columns)},
            ))
            satellite_reports[sat_id] = SatelliteDataReport(
                satellite_id=sat_id,
                row_count=row_count,
                valid=False,
                issues=sat_issues,
            )
            errors.extend(sat_errors)
            continue

        # Convert timestamps
        timestamps = pd.to_datetime(df["utc_time"], errors="coerce")
        null_ts_count = int(timestamps.isna().sum())
        if null_ts_count > 0:
            msg = f"Timestamp column contains {null_ts_count} unparsable or null values"
            sat_issues.append(msg)
            sat_errors.append(ValidationIssue(
                status="invalid",
                stage="data_validation",
                satellite=sat_id,
                reason=msg,
            ))

        # Check required target error columns
        missing_targets = [col for col in TARGET_COLS_INTERNAL if col not in df.columns]
        if missing_targets and require_all_targets:
            msg = f"Missing required target error columns: {missing_targets}"
            sat_issues.append(msg)
            sat_errors.append(ValidationIssue(
                status="invalid",
                stage="data_validation",
                satellite=sat_id,
                reason=msg,
                details={"missing": missing_targets, "available": list(df.columns)},
            ))

        # Minimum observations
        if row_count < min_history_rows:
            msg = (
                f"Insufficient historical observations: found {row_count} rows, "
                f"minimum required is {min_history_rows}"
            )
            sat_issues.append(msg)
            sat_errors.append(ValidationIssue(
                status="invalid",
                stage="data_validation",
                satellite=sat_id,
                reason=msg,
                details={"row_count": row_count, "min_required": min_history_rows},
            ))

        # Clean copy for numeric & cadence inspection
        clean_df = df.copy()
        clean_df["utc_time"] = timestamps
        clean_df = clean_df.dropna(subset=["utc_time"]).sort_values("utc_time")

        # Duplicates
        duplicate_count = int(clean_df.duplicated(subset=["utc_time"]).sum())
        if duplicate_count > 0:
            clean_df = clean_df.drop_duplicates(subset=["utc_time"], keep="first")
            msg = f"Found {duplicate_count} duplicate timestamp rows (deduplicated)"
            warnings.append(ValidationIssue(
                status="warning",
                stage="data_validation",
                satellite=sat_id,
                reason=msg,
                details={"duplicate_count": duplicate_count},
            ))

        # Check monotonic ordering
        is_monotonic = bool(clean_df["utc_time"].is_monotonic_increasing)
        if not is_monotonic:
            msg = "Timestamps are not monotonically increasing"
            sat_issues.append(msg)
            sat_errors.append(ValidationIssue(
                status="invalid",
                stage="data_validation",
                satellite=sat_id,
                reason=msg,
            ))

        # Missing values in target columns
        missing_counts = {}
        for col in TARGET_COLS_INTERNAL:
            if col in clean_df.columns:
                n_miss = int(clean_df[col].isna().sum())
                missing_counts[col] = n_miss
                if n_miss > 0:
                    msg = f"Column {col} has {n_miss} non-finite/null values"
                    sat_issues.append(msg)
                    sat_errors.append(ValidationIssue(
                        status="invalid",
                        stage="data_validation",
                        satellite=sat_id,
                        reason=msg,
                    ))

        # Cadence check
        time_diffs = clean_df["utc_time"].diff().dropna()
        est_cadence = None
        irregular_count = 0
        if len(time_diffs) > 0:
            median_interval = time_diffs.median()
            est_cadence = float(median_interval.total_seconds() / 60.0)
            irregular_count = int((time_diffs != median_interval).sum())

        sat_valid = len(sat_errors) == 0
        satellite_reports[sat_id] = SatelliteDataReport(
            satellite_id=sat_id,
            row_count=row_count,
            start_time=clean_df["utc_time"].min().isoformat() if len(clean_df) else None,
            end_time=clean_df["utc_time"].max().isoformat() if len(clean_df) else None,
            duplicate_epochs=duplicate_count,
            missing_values=missing_counts,
            is_monotonic=is_monotonic,
            estimated_cadence_minutes=est_cadence,
            irregular_cadence_count=irregular_count,
            valid=sat_valid,
            issues=sat_issues,
        )

        errors.extend(sat_errors)
        if sat_valid:
            clean_df["satellite_id"] = sat_id
            normalized_data[sat_id] = clean_df.reset_index(drop=True)

    is_overall_valid = len(errors) == 0 and len(normalized_data) > 0

    return ValidationResult(
        is_valid=is_overall_valid,
        status="valid" if is_overall_valid else "invalid",
        satellites_found=satellites_found,
        satellite_reports=satellite_reports,
        errors=errors,
        warnings=warnings,
        normalized_data=normalized_data if is_overall_valid else None,
    )
