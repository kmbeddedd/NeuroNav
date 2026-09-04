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
    "epoch": "utc_time",
    "x_error (m)": "x_error_m",
    "error_x": "x_error_m",
    "x_error": "x_error_m",
    "x": "x_error_m",
    "y_error (m)": "y_error_m",
    "y_error": "y_error_m",
    "error_y": "y_error_m",
    "y": "y_error_m",
    "z_error (m)": "z_error_m",
    "z_error": "z_error_m",
    "error_z": "z_error_m",
    "z": "z_error_m",
    "satclockerror (m)": "clock_error_m",
    "satclockerror": "clock_error_m",
    "clock_error (m)": "clock_error_m",
    "clock_error": "clock_error_m",
    "error_clock": "clock_error_m",
    "clock": "clock_error_m",
    "satellite_id": "satellite_id",
    "satellite": "satellite_id",
    "sat_id": "satellite_id",
    "prn": "satellite_id",
}

SATELLITE_COLS = ("Satellite_ID", "satellite_id", "sat_id", "PRN", "prn", "Satellite")
TIME_COLS = ("utc_time", "Timestamp", "timestamp", "time", "Date_Time", "datetime", "Epoch", "epoch")


def detect_satellite_col(df: pd.DataFrame) -> Optional[str]:
    """Identify the satellite identifier column in dataframe."""
    for col in SATELLITE_COLS:
        if col in df.columns:
            return col
    return None


def detect_time_col(df: pd.DataFrame) -> Optional[str]:
    """Identify the timestamp column in dataframe."""
    for col in TIME_COLS:
        if col in df.columns:
            return col
    return None



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
class SamplingMetadata:
    original_cadence_minutes: Optional[float]  # detected median interval
    target_cadence_minutes: float              # configured/auto target
    is_irregular: bool
    observed_epochs: int
    interpolated_epochs: int
    interpolation_fraction: float
    max_gap_minutes: float
    mean_cadence_minutes: Optional[float] = None
    cadence_std_minutes: Optional[float] = None
    cadence_classification: str = "regular"  # "regular" | "mildly_irregular" | "strongly_irregular"
    duration_hours: float = 0.0
    cadence_warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SatelliteDataset:
    satellite_id: str
    orbit_type: str  # "GEO" | "MEO" | "LEO" | "UNKNOWN"
    dataframe: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    sampling: Optional[SamplingMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "satellite_id": self.satellite_id,
            "orbit_type": self.orbit_type,
            "row_count": len(self.dataframe),
            "columns": list(self.dataframe.columns),
            "metadata": self.metadata,
            "sampling": self.sampling.to_dict() if self.sampling else None,
        }


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
    satellite_id: Optional[str] = None,
    orbit_type: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Loads telemetry from a DataFrame, CSV path, or folder of series CSVs.
    
    Returns:
        Dict mapping satellite_id -> DataFrame of telemetry rows.
    """
    target_sat_id = satellite_id or default_satellite_id

    if isinstance(source, dict):
        return {
            sat_id: normalize_dataframe_columns(df.copy())
            for sat_id, df in source.items()
        }

    if isinstance(source, pd.DataFrame):
        df = normalize_dataframe_columns(source.copy())
        if "utc_time" in df.columns:
            df = df.sort_values("utc_time").reset_index(drop=True)
        if satellite_id:
            df["satellite_id"] = satellite_id
            return {satellite_id: df}
        if "satellite_id" in df.columns:
            return {
                str(sat_id): sat_df.reset_index(drop=True)
                for sat_id, sat_df in df.groupby("satellite_id")
            }
        return {target_sat_id: df}

    path = Path(source)
    if path.is_file():
        df = normalize_dataframe_columns(pd.read_csv(path))
        if "utc_time" in df.columns:
            df = df.sort_values("utc_time").reset_index(drop=True)
        if satellite_id:
            df["satellite_id"] = satellite_id
            return {satellite_id: df}
        if "satellite_id" in df.columns:
            return {
                str(sat_id): sat_df.reset_index(drop=True)
                for sat_id, sat_df in df.groupby("satellite_id")
            }
        # Infer satellite from filename if possible (e.g. DATA_GEO_Train.csv -> GEO)
        stem = path.stem.upper()
        sat_id = target_sat_id
        if "GEO" in stem:
            sat_id = "GEO"
        elif "MEO" in stem:
            sat_id = "MEO-2" if ("2" in stem or "MEO2" in stem) else "MEO-1"
        else:
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


def infer_orbit_type(satellite_id: str) -> str:
    """Infers orbit regime ("GEO", "MEO", "LEO", "UNKNOWN") from satellite identifier."""
    s = str(satellite_id).strip().upper()
    if "GEO" in s:
        return "GEO"
    if "MEO" in s:
        return "MEO"
    if "LEO" in s:
        return "LEO"
    return "UNKNOWN"


def regularize_cadence(
    df: pd.DataFrame,
    target_cadence_minutes: Optional[float] = None,
    max_interpolation_fraction: float = 0.3,
    resample_if_irregular: bool = False,
) -> Tuple[pd.DataFrame, SamplingMetadata]:
    """Inspects and optionally regularizes the temporal cadence of a telemetry DataFrame.
    
    Args:
        df: DataFrame containing 'utc_time' column.
        target_cadence_minutes: Desired cadence in minutes. If None, uses detected median difference.
        max_interpolation_fraction: Maximum allowable fraction of interpolated epochs (default 0.3).
            Raises ValueError if exceeded when resample_if_irregular=True.
        resample_if_irregular: If True, interpolates onto regular time grid. If False, retains
            raw observations without artificial interpolation and tags _is_interpolated=False.
            
    Returns:
        Tuple of (DataFrame with '_is_interpolated' column, SamplingMetadata).
    """
    if "utc_time" not in df.columns:
        raise ValueError("DataFrame must contain 'utc_time' column.")

    clean_df = df.copy()
    clean_df["utc_time"] = pd.to_datetime(clean_df["utc_time"])
    clean_df = clean_df.dropna(subset=["utc_time"]).sort_values("utc_time").drop_duplicates(subset=["utc_time"])

    n_obs = len(clean_df)
    if n_obs < 2:
        meta = SamplingMetadata(
            original_cadence_minutes=None,
            target_cadence_minutes=float(target_cadence_minutes) if target_cadence_minutes else 15.0,
            is_irregular=False,
            observed_epochs=n_obs,
            interpolated_epochs=0,
            interpolation_fraction=0.0,
            max_gap_minutes=0.0,
            cadence_warning="Insufficient epochs (< 2) to evaluate cadence.",
        )
        clean_df["_is_interpolated"] = False
        return clean_df.reset_index(drop=True), meta

    diffs = clean_df["utc_time"].diff().dropna()
    diffs_sec = diffs.dt.total_seconds()
    median_minutes = float(diffs_sec.median() / 60.0)
    mean_minutes = float(diffs_sec.mean() / 60.0)
    std_minutes = float(diffs_sec.std() / 60.0) if len(diffs_sec) > 1 else 0.0
    max_gap_minutes = float(diffs_sec.max() / 60.0)
    duration_hours = float((clean_df["utc_time"].max() - clean_df["utc_time"].min()).total_seconds() / 3600.0)

    target_cadence = float(target_cadence_minutes) if target_cadence_minutes is not None else median_minutes
    target_td = pd.Timedelta(minutes=target_cadence)

    # Allow a small 5-second jitter tolerance for timestamp recording drift
    tolerated_irregular = (diffs - target_td).abs() > pd.Timedelta(seconds=5)
    is_irregular = bool(tolerated_irregular.any())

    if not is_irregular or std_minutes <= 0.15:
        cadence_class = "regular"
    elif max_gap_minutes <= 3.0 * median_minutes and std_minutes <= 0.5 * median_minutes:
        cadence_class = "mildly_irregular"
    else:
        cadence_class = "strongly_irregular"

    cadence_warning = None
    if is_irregular:
        cadence_warning = (
            f"Observed cadence varies from target {target_cadence:.1f}m "
            f"(median={median_minutes:.1f}m, mean={mean_minutes:.1f}m, "
            f"std={std_minutes:.1f}m, max_gap={max_gap_minutes:.1f}m, class={cadence_class})."
        )

    if not resample_if_irregular or not is_irregular:
        clean_df["_is_interpolated"] = False
        meta = SamplingMetadata(
            original_cadence_minutes=median_minutes,
            target_cadence_minutes=target_cadence,
            is_irregular=is_irregular,
            observed_epochs=n_obs,
            interpolated_epochs=0,
            interpolation_fraction=0.0,
            max_gap_minutes=max_gap_minutes,
            mean_cadence_minutes=mean_minutes,
            cadence_std_minutes=std_minutes,
            cadence_classification=cadence_class,
            duration_hours=duration_hours,
            cadence_warning=cadence_warning,
        )
        return clean_df.reset_index(drop=True), meta

    # Resample onto a regular grid
    start_time = clean_df["utc_time"].min()
    end_time = clean_df["utc_time"].max()
    grid = pd.date_range(start=start_time, end=end_time, freq=target_td)

    indexed_df = clean_df.set_index("utc_time")
    numeric_cols = list(indexed_df.select_dtypes(include=[np.number]).columns)
    non_numeric_cols = [c for c in indexed_df.columns if c not in numeric_cols]

    reindexed = indexed_df.reindex(indexed_df.index.union(grid))
    if numeric_cols:
        reindexed[numeric_cols] = reindexed[numeric_cols].interpolate(method="time")
    for col in non_numeric_cols:
        reindexed[col] = reindexed[col].ffill().bfill()

    final_df = reindexed.loc[grid].copy().reset_index().rename(columns={"index": "utc_time"})

    total_epochs = len(final_df)
    existing_set = set(clean_df["utc_time"])
    final_df["_is_interpolated"] = ~final_df["utc_time"].isin(existing_set)
    interpolated_epochs = int(final_df["_is_interpolated"].sum())
    interp_fraction = float(interpolated_epochs / max(1, total_epochs))

    if interp_fraction > max_interpolation_fraction:
        msg = f"Interpolation fraction {interp_fraction:.1%} exceeds maximum allowable {max_interpolation_fraction:.1%}."
        cadence_warning = f"{cadence_warning} {msg}" if cadence_warning else msg
        raise ValueError(msg)

    meta = SamplingMetadata(
        original_cadence_minutes=median_minutes,
        target_cadence_minutes=target_cadence,
        is_irregular=True,
        observed_epochs=n_obs,
        interpolated_epochs=interpolated_epochs,
        interpolation_fraction=interp_fraction,
        max_gap_minutes=max_gap_minutes,
        mean_cadence_minutes=mean_minutes,
        cadence_std_minutes=std_minutes,
        cadence_classification="regular",
        duration_hours=duration_hours,
        cadence_warning=cadence_warning,
    )
    return final_df.reset_index(drop=True), meta


def validate_satellite_dataset(
    source: Union[str, Path, pd.DataFrame],
    satellite_id: Optional[str] = None,
    orbit_type: Optional[str] = None,
    target_cadence_minutes: Optional[float] = None,
    min_history_rows: int = 8,
    resample_if_irregular: bool = False,
    max_interpolation_fraction: float = 0.3,
) -> SatelliteDataset:
    """Validates and packages a single satellite upload into a validated SatelliteDataset.
    
    Accepts CSVs with or without a 'satellite_id' column.
    
    Args:
        source: File path, string path, or pandas DataFrame.
        satellite_id: Explicit satellite ID (e.g. 'GEO', 'MEO-1'). If omitted, inferred from source.
        orbit_type: 'GEO', 'MEO', 'LEO', etc. If omitted, inferred from satellite ID.
        target_cadence_minutes: Optional target cadence in minutes.
        min_history_rows: Minimum required epochs (default 8).
        resample_if_irregular: Whether to linearly interpolate irregular epochs.
        max_interpolation_fraction: Max allowed interpolated fraction (default 0.3).
        
    Returns:
        SatelliteDataset containing normalized DataFrame and SamplingMetadata.
        
    Raises:
        ValueError: If validation fails or data does not meet schema requirements.
    """
    raw_dict = load_telemetry_source(source, satellite_id=satellite_id, orbit_type=orbit_type)
    if not raw_dict:
        raise ValueError(f"No telemetry data could be loaded from {source}")

    chosen_id = satellite_id or next(iter(raw_dict.keys()))
    df = raw_dict[chosen_id]

    val_res = validate_dataset(df, min_history_rows=min_history_rows)
    if not val_res.is_valid:
        err_msgs = [e.reason for e in val_res.errors]
        raise ValueError(f"Dataset validation failed for {chosen_id}: {'; '.join(err_msgs)}")

    validated_df = val_res.normalized_data.get(chosen_id, next(iter(val_res.normalized_data.values())))

    reg_df, sampling_meta = regularize_cadence(
        validated_df,
        target_cadence_minutes=target_cadence_minutes,
        max_interpolation_fraction=max_interpolation_fraction,
        resample_if_irregular=resample_if_irregular,
    )

    resolved_orbit_type = (orbit_type or infer_orbit_type(chosen_id)).upper()
    reg_df["satellite_id"] = chosen_id

    return SatelliteDataset(
        satellite_id=chosen_id,
        orbit_type=resolved_orbit_type,
        dataframe=reg_df,
        metadata={
            "source": str(source) if isinstance(source, (str, Path)) else "in_memory_dataframe",
            "start_time": reg_df["utc_time"].min().isoformat() if len(reg_df) else None,
            "end_time": reg_df["utc_time"].max().isoformat() if len(reg_df) else None,
        },
        sampling=sampling_meta,
    )
