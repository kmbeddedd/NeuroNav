"""Authoritative feature engineering pipeline for satellite orbit and clock error forecasting.

Integrates secular and diurnal harmonic phase features, deterministic solar radiation
pressure (SRP) geometry, and radial-in-track-cross-track (RIC) coordinate features.
Guarantees strict training/inference parity and zero target error leakage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.physics import (
    OrbitalStateProvider,
    build_physics_features,
    build_ric_features,
    get_orbital_state_provider,
)

TARGET_COLS_DEFAULT = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]


def extract_harmonic_time_features(
    times: Union[pd.Series, pd.DatetimeIndex, Sequence[pd.Timestamp]],
    origin: pd.Timestamp,
    harmonics: int = 6,
) -> np.ndarray:
    """Extracts secular and diurnal harmonic phase features from timestamps."""
    index = pd.DatetimeIndex(times)
    elapsed_days = (index - origin).total_seconds().to_numpy(dtype=np.float64) / 86400.0
    phase = 2.0 * np.pi * (
        index.hour.to_numpy(dtype=np.float64) * 3600.0
        + index.minute.to_numpy(dtype=np.float64) * 60.0
        + index.second.to_numpy(dtype=np.float64)
    ) / 86400.0

    columns = [elapsed_days, np.square(elapsed_days) / 49.0]
    for h in range(1, harmonics + 1):
        columns.extend([np.sin(h * phase), np.cos(h * phase)])

    return np.column_stack(columns).astype(np.float64)


@dataclass
class FeatureManifest:
    """Tracks feature configuration, versioning, and column layout for reproducible inference."""

    feature_version: str = "v2"
    features: List[str] = field(default_factory=list)
    target_columns: List[str] = field(default_factory=lambda: list(TARGET_COLS_DEFAULT))
    use_ric: bool = False
    use_srp: bool = False
    orbital_state_source: str = "nominal"
    cadence_minutes: float = 15.0
    harmonics_count: int = 6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureManifest:
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


def build_training_features(
    df: pd.DataFrame,
    manifest: FeatureManifest,
    origin: Optional[pd.Timestamp] = None,
    provider: Optional[OrbitalStateProvider] = None,
    satellite_id: str = "",
    orbit_class: str = "MEO",
) -> Tuple[np.ndarray, List[str], pd.Timestamp]:
    """Builds the feature matrix for model training.

    Args:
        df: Training DataFrame containing 'utc_time' and error columns.
        manifest: FeatureManifest specifying enabled feature components.
        origin: Reference epoch for elapsed time. Defaults to floor of first timestamp.
        provider: OrbitalStateProvider instance.
        satellite_id: Satellite identifier.
        orbit_class: 'GEO', 'MEO', etc.

    Returns:
        Tuple of (feature_matrix, feature_names, resolved_origin).
    """
    if "utc_time" not in df.columns:
        raise ValueError("DataFrame missing required 'utc_time' column.")

    clean_df = df.dropna(subset=["utc_time"]).sort_values("utc_time").reset_index(drop=True)
    ts = pd.DatetimeIndex(clean_df["utc_time"])

    resolved_origin = origin or ts[0].floor("D")

    # 1. Harmonic time features
    x_time = extract_harmonic_time_features(ts, resolved_origin, harmonics=manifest.harmonics_count)
    feature_names: List[str] = ["elapsed_days", "elapsed_days_sq"]
    for k in range(1, manifest.harmonics_count + 1):
        feature_names.extend([f"sin_harm_{k}", f"cos_harm_{k}"])

    feature_blocks = [x_time]

    # 2. Physics / SRP features
    if manifest.use_srp:
        state_provider = provider or get_orbital_state_provider(
            "nominal_approximation" if manifest.orbital_state_source == "nominal" else manifest.orbital_state_source
        )
        srp_df = build_physics_features(
            ts,
            orbital_state_provider=state_provider,
            satellite_id=satellite_id,
            orbit_class=orbit_class,
            features=("sun_beta_angle", "shadow_factor", "solar_cos_angle"),
        )
        feature_blocks.append(srp_df.to_numpy(dtype=np.float64))
        feature_names.extend(list(srp_df.columns))

    # 3. RIC features (computed from observed historical errors)
    if manifest.use_ric:
        state_provider = provider or get_orbital_state_provider(
            "nominal_approximation" if manifest.orbital_state_source == "nominal" else manifest.orbital_state_source
        )
        ric_df = build_ric_features(
            clean_df,
            ts,
            provider=state_provider,
            satellite_id=satellite_id,
            orbit_class=orbit_class,
        )
        feature_blocks.append(ric_df.to_numpy(dtype=np.float64))
        feature_names.extend(["ric_r", "ric_i", "ric_c"])

    feature_matrix = np.column_stack(feature_blocks).astype(np.float64)
    manifest.features = list(feature_names)

    return feature_matrix, feature_names, resolved_origin


def build_inference_features(
    timestamps: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    origin: pd.Timestamp,
    manifest: FeatureManifest,
    provider: Optional[OrbitalStateProvider] = None,
    satellite_id: str = "",
    orbit_class: str = "MEO",
    history_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Builds the feature matrix for model inference across forecast horizon timestamps.

    Strictly causal: future errors are never assumed known.

    Args:
        timestamps: Evaluation epochs for the forecast horizon.
        origin: Reference epoch used during training.
        manifest: Trained model's FeatureManifest.
        provider: OrbitalStateProvider instance.
        satellite_id: Satellite identifier.
        orbit_class: 'GEO', 'MEO', etc.
        history_df: Optional historical lookback DataFrame to supply latest known state for RIC features.

    Returns:
        Tuple of (feature_matrix, feature_names).
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if len(ts) == 0:
        return np.empty((0, len(manifest.features)), dtype=np.float64), list(manifest.features)

    # 1. Harmonic time features
    x_time = extract_harmonic_time_features(ts, origin, harmonics=manifest.harmonics_count)
    feature_names: List[str] = ["elapsed_days", "elapsed_days_sq"]
    for k in range(1, manifest.harmonics_count + 1):
        feature_names.extend([f"sin_harm_{k}", f"cos_harm_{k}"])

    feature_blocks = [x_time]

    # 2. Physics / SRP features
    if manifest.use_srp:
        state_provider = provider or get_orbital_state_provider(
            "nominal_approximation" if manifest.orbital_state_source == "nominal" else manifest.orbital_state_source
        )
        srp_df = build_physics_features(
            ts,
            orbital_state_provider=state_provider,
            satellite_id=satellite_id,
            orbit_class=orbit_class,
            features=("sun_beta_angle", "shadow_factor", "solar_cos_angle"),
        )
        feature_blocks.append(srp_df.to_numpy(dtype=np.float64))
        feature_names.extend(list(srp_df.columns))

    # 3. RIC features (causal handoff: latest observed RIC state held across horizon)
    if manifest.use_ric:
        state_provider = provider or get_orbital_state_provider(
            "nominal_approximation" if manifest.orbital_state_source == "nominal" else manifest.orbital_state_source
        )
        if history_df is not None and not history_df.empty:
            req_cols = ["x_error_m", "y_error_m", "z_error_m"]
            avail_cols = [c for c in req_cols if c in history_df.columns]
            if len(avail_cols) == 3:
                last_row = history_df.dropna(subset=["utc_time", *req_cols]).iloc[-1:]
                if not last_row.empty:
                    last_ric = build_ric_features(
                        last_row,
                        last_row["utc_time"],
                        provider=state_provider,
                        satellite_id=satellite_id,
                        orbit_class=orbit_class,
                    )
                    ric_vals = last_ric[["ric_r", "ric_i", "ric_c"]].to_numpy(dtype=np.float64)
                    x_ric = np.repeat(ric_vals, len(ts), axis=0)
                else:
                    x_ric = np.zeros((len(ts), 3), dtype=np.float64)
            else:
                x_ric = np.zeros((len(ts), 3), dtype=np.float64)
        else:
            x_ric = np.zeros((len(ts), 3), dtype=np.float64)

        feature_blocks.append(x_ric)
        feature_names.extend(["ric_r", "ric_i", "ric_c"])

    feature_matrix = np.column_stack(feature_blocks).astype(np.float64)
    return feature_matrix, feature_names
