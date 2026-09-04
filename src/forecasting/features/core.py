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

from src.forecasting.physics import (
    OrbitalStateProvider,
    build_physics_features,
    build_ric_features,
    get_orbital_state_provider,
    ric_basis,
)

TARGET_COLS_DEFAULT = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]


def extract_harmonic_time_features(
    times: Union[pd.Series, pd.DatetimeIndex, Sequence[pd.Timestamp]],
    origin: pd.Timestamp,
    harmonics: int = 6,
    include_dt: bool = False,
    last_known_time: Optional[pd.Timestamp] = None,
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
    if include_dt:
        if len(index) > 1:
            diffs = np.empty(len(index), dtype=np.float64)
            diffs[1:] = (index[1:] - index[:-1]).total_seconds()
            if last_known_time is not None:
                diffs[0] = (index[0] - pd.to_datetime(last_known_time)).total_seconds()
            else:
                diffs[0] = diffs[1]
        elif len(index) == 1:
            if last_known_time is not None:
                diffs = np.array([(index[0] - pd.to_datetime(last_known_time)).total_seconds()], dtype=np.float64)
            else:
                diffs = np.array([900.0], dtype=np.float64)
        else:
            diffs = np.empty(0, dtype=np.float64)
        columns.append(diffs)

    for h in range(1, harmonics + 1):
        columns.extend([np.sin(h * phase), np.cos(h * phase)])

    return np.column_stack(columns).astype(np.float64)


def build_ric_geometry_features(
    timestamps: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    provider: Optional[OrbitalStateProvider] = None,
    satellite_id: str = "",
    orbit_class: str = "MEO",
) -> np.ndarray:
    """Computes time-varying RIC unit basis orientation features from orbital state.

    Extracts the Z-axis projections of radial, in-track, and cross-track basis vectors
    in ECEF. Guarantees identical causal semantics during training and forecasting.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    state_provider = provider or get_orbital_state_provider("nominal_approximation")
    pos, vel = state_provider.get_state(ts, satellite_id=satellite_id, orbit_class=orbit_class)
    basis = ric_basis(pos, vel)
    ric_r = basis[:, 2, 0]
    ric_i = basis[:, 2, 1]
    ric_c = basis[:, 2, 2]
    return np.column_stack([ric_r, ric_i, ric_c]).astype(np.float64)


@dataclass
class FeatureManifest:
    """Tracks feature configuration, versioning, and column layout for reproducible inference."""

    feature_version: str = "v2"
    features: List[str] = field(default_factory=list)
    target_columns: List[str] = field(default_factory=lambda: list(TARGET_COLS_DEFAULT))
    physics_mode: str = "nominal"  # "none" | "nominal" | "provided"
    use_ric: bool = False
    use_srp: bool = False
    orbital_state_source: str = "nominal"
    cadence_minutes: float = 15.0
    harmonics_count: int = 6
    include_dt: bool = False
    state_artifact: Optional[str] = None

    def __post_init__(self):
        if self.physics_mode == "none":
            self.use_ric = False
            self.use_srp = False
            self.orbital_state_source = "none"
        elif self.physics_mode == "provided":
            self.orbital_state_source = "provided"
        elif self.physics_mode == "nominal":
            if self.orbital_state_source == "nominal":
                self.orbital_state_source = "nominal_approximation"

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
    x_time = extract_harmonic_time_features(
        ts,
        resolved_origin,
        harmonics=manifest.harmonics_count,
        include_dt=manifest.include_dt,
    )
    feature_names: List[str] = ["elapsed_days", "elapsed_days_sq"]
    if manifest.include_dt:
        feature_names.append("dt_seconds")
    for k in range(1, manifest.harmonics_count + 1):
        feature_names.extend([f"sin_harm_{k}", f"cos_harm_{k}"])

    feature_blocks = [x_time]

    effective_use_srp = manifest.use_srp and manifest.physics_mode != "none"
    effective_use_ric = manifest.use_ric and manifest.physics_mode != "none"

    # 2. Physics / SRP features
    if effective_use_srp:
        state_provider = provider or get_orbital_state_provider(
            "nominal_approximation" if manifest.orbital_state_source in ("nominal", "nominal_approximation") else manifest.orbital_state_source
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

    # 3. RIC features (time-varying instantaneous basis orientation)
    if effective_use_ric:
        state_provider = provider or get_orbital_state_provider(
            "nominal_approximation" if manifest.orbital_state_source in ("nominal", "nominal_approximation") else manifest.orbital_state_source
        )
        ric_geom = build_ric_geometry_features(
            ts,
            provider=state_provider,
            satellite_id=satellite_id,
            orbit_class=orbit_class,
        )
        feature_blocks.append(ric_geom)
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

    Strictly causal and symmetrical with build_training_features: future errors are never assumed known.

    Args:
        timestamps: Evaluation epochs for the forecast horizon.
        origin: Reference epoch used during training.
        manifest: Trained model's FeatureManifest.
        provider: OrbitalStateProvider instance.
        satellite_id: Satellite identifier.
        orbit_class: 'GEO', 'MEO', etc.
        history_df: Optional historical lookback DataFrame to supply latest known timestamp.

    Returns:
        Tuple of (feature_matrix, feature_names).
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if len(ts) == 0:
        return np.empty((0, len(manifest.features)), dtype=np.float64), list(manifest.features)

    last_known_time = None
    if history_df is not None and not history_df.empty and "utc_time" in history_df.columns:
        last_known_time = pd.to_datetime(history_df["utc_time"]).dropna().iloc[-1]

    # 1. Harmonic time features
    x_time = extract_harmonic_time_features(
        ts,
        origin,
        harmonics=manifest.harmonics_count,
        include_dt=manifest.include_dt,
        last_known_time=last_known_time,
    )
    feature_names: List[str] = ["elapsed_days", "elapsed_days_sq"]
    if manifest.include_dt:
        feature_names.append("dt_seconds")
    for k in range(1, manifest.harmonics_count + 1):
        feature_names.extend([f"sin_harm_{k}", f"cos_harm_{k}"])

    feature_blocks = [x_time]

    effective_use_srp = manifest.use_srp and manifest.physics_mode != "none"
    effective_use_ric = manifest.use_ric and manifest.physics_mode != "none"

    # 2. Physics / SRP features
    if effective_use_srp:
        state_provider = provider or get_orbital_state_provider(
            "nominal_approximation" if manifest.orbital_state_source in ("nominal", "nominal_approximation") else manifest.orbital_state_source
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

    # 3. RIC features (instantaneous basis orientation at each forecast epoch)
    if effective_use_ric:
        state_provider = provider or get_orbital_state_provider(
            "nominal_approximation" if manifest.orbital_state_source in ("nominal", "nominal_approximation") else manifest.orbital_state_source
        )
        ric_geom = build_ric_geometry_features(
            ts,
            provider=state_provider,
            satellite_id=satellite_id,
            orbit_class=orbit_class,
        )
        feature_blocks.append(ric_geom)
        feature_names.extend(["ric_r", "ric_i", "ric_c"])

    feature_matrix = np.column_stack(feature_blocks).astype(np.float64)
    return feature_matrix, feature_names
