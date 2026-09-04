from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

SPEED_OF_LIGHT_M_S = 299792458.0

def _unit(vector: np.ndarray, eps: float=1e-12) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm <= eps):
        raise ValueError('Cannot construct an orbital frame from a zero-length vector')
    return vector / norm

def ric_basis(position_ecef: np.ndarray, velocity_ecef: np.ndarray) -> np.ndarray:
    position = np.asarray(position_ecef, dtype=np.float64)
    velocity = np.asarray(velocity_ecef, dtype=np.float64)
    if position.shape != velocity.shape or position.shape[-1] != 3:
        raise ValueError('position_ecef and velocity_ecef must share shape (..., 3)')
    radial = _unit(position)
    cross_track = _unit(np.cross(position, velocity))
    in_track = _unit(np.cross(cross_track, radial))
    return np.stack([radial, in_track, cross_track], axis=-1)

def ecef_error_to_ric(error_ecef: np.ndarray, position_ecef: np.ndarray, velocity_ecef: np.ndarray) -> np.ndarray:
    error = np.asarray(error_ecef, dtype=np.float64)
    basis = ric_basis(position_ecef, velocity_ecef)
    if error.shape != basis.shape[:-1]:
        raise ValueError('error_ecef must have the same (..., 3) shape as position')
    return np.einsum('...ji,...j->...i', basis, error)

def ric_error_to_ecef(error_ric: np.ndarray, position_ecef: np.ndarray, velocity_ecef: np.ndarray) -> np.ndarray:
    error = np.asarray(error_ric, dtype=np.float64)
    basis = ric_basis(position_ecef, velocity_ecef)
    if error.shape != basis.shape[:-1]:
        raise ValueError('error_ric must have the same (..., 3) shape as position')
    return np.einsum('...ij,...j->...i', basis, error)

def clock_seconds_to_metres(clock_seconds: np.ndarray) -> np.ndarray:
    return np.asarray(clock_seconds, dtype=np.float64) * SPEED_OF_LIGHT_M_S

def approximate_sun_unit_ecef(timestamps: np.ndarray) -> np.ndarray:
    """Computes approximate unit vector pointing toward the Sun in ECEF coordinates.
    
    Uses standard solar astronomical equations (declination and Greenwich Hour Angle)
    to provide a low-overhead proxy for solar radiation pressure and eclipse geometry.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    # Day of year (1-366) and hour of day in UTC
    doy = ts.dayofyear.to_numpy(dtype=np.float64)
    utc_hours = ts.hour.to_numpy(dtype=np.float64) + ts.minute.to_numpy(dtype=np.float64) / 60.0 + ts.second.to_numpy(dtype=np.float64) / 3600.0
    
    # Solar declination angle
    delta_rad = np.radians(-23.44 * np.cos(np.radians(360.0 / 365.25 * (doy + 10.0))))
    # Greenwich Hour Angle of the Sun (approx noon = 0 deg)
    gha_rad = np.radians(15.0 * (utc_hours - 12.0))
    
    sun_x = np.cos(delta_rad) * np.cos(gha_rad)
    sun_y = -np.cos(delta_rad) * np.sin(gha_rad)
    sun_z = np.sin(delta_rad)
    
    sun_vec = np.stack([sun_x, sun_y, sun_z], axis=-1)
    return _unit(sun_vec)

def compute_solar_cos_angle(position_ecef: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Computes cosine of angle between satellite position vector and the Sun."""
    sat_unit = _unit(np.asarray(position_ecef, dtype=np.float64))
    sun_unit = approximate_sun_unit_ecef(timestamps)
    return np.einsum('...i,...i->...', sat_unit, sun_unit)

def fit_harmonic_orbit_baseline(
    t_in: np.ndarray,
    y_in: np.ndarray,
    t_out: np.ndarray,
    periods_hours: tuple[float, ...] = (12.0, 24.0),
    ridge_lambda: float = 1e-4,
) -> np.ndarray:
    """Fits an analytical harmonic + polynomial baseline over lookback t_in and extrapolates over t_out.
    
    Model: y(t) = c0 + c1*t + sum_k [A_k * sin(w_k*t) + B_k * cos(w_k*t)]
    Uses L2-regularized least squares (Ridge) for unconditional numerical stability.
    """
    t_in = np.asarray(t_in, dtype=np.float64)
    y_in = np.asarray(y_in, dtype=np.float64)
    t_out = np.asarray(t_out, dtype=np.float64)
    
    # Center and scale time to hours relative to t_in[0]
    t0 = t_in[0]
    t_in_hrs = (t_in - t0) / 3600.0
    t_out_hrs = (t_out - t0) / 3600.0
    
    def _design_matrix(t: np.ndarray) -> np.ndarray:
        cols = [np.ones_like(t), t]
        for p in periods_hours:
            omega = 2.0 * np.pi / p
            cols.append(np.sin(omega * t))
            cols.append(np.cos(omega * t))
        return np.column_stack(cols)
    
    A_in = _design_matrix(t_in_hrs)
    A_out = _design_matrix(t_out_hrs)
    
    # Solve Ridge: (A^T A + lambda I) w = A^T y
    AtA = A_in.T @ A_in
    reg = ridge_lambda * np.eye(AtA.shape[0])
    reg[0, 0] = 0.0  # Do not regularize intercept
    At_y = A_in.T @ y_in
    
    w = np.linalg.solve(AtA + reg, At_y)
    return A_out @ w


class OrbitalStateProvider(ABC):
    """Abstract provider interface for satellite orbital position and velocity state."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Identifier describing the origin and physical fidelity of the orbital state."""
        pass

    @abstractmethod
    def get_state(
        self,
        timestamps: Sequence[pd.Timestamp] | pd.DatetimeIndex,
        satellite_id: str = "",
        orbit_class: str = "MEO",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (position_ecef, velocity_ecef) in metres and metres/second.

        Both arrays share shape (N, 3).
        """
        pass


class NominalStateProvider(OrbitalStateProvider):
    """Generates nominal analytical orbital state vectors for geometric transformations.

    NOTE: This is an explicitly approximate geometry fallback for datasets that contain
    only error residuals (X, Y, Z, Clock) and lack actual broadcast or precise ephemeris.
    Physical fidelity is nominal/approximate and does not represent measured ephemeris.
    """

    def __init__(self, geo_longitude_deg: float = 0.0):
        self.geo_longitude_deg = geo_longitude_deg

    @property
    def source_name(self) -> str:
        return "nominal_approximation"

    def get_state(
        self,
        timestamps: Sequence[pd.Timestamp] | pd.DatetimeIndex,
        satellite_id: str = "",
        orbit_class: str = "MEO",
    ) -> Tuple[np.ndarray, np.ndarray]:
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
        if len(ts) == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

        elapsed_sec = (ts - ts[0]).total_seconds().to_numpy(dtype=np.float64)
        n_epochs = len(ts)
        orbit_type = str(orbit_class).upper()

        if "GEO" in orbit_type or "GEO" in satellite_id.upper():
            # Geostationary Earth Orbit (circular, equatorial):
            # In an Earth-Centered Earth-Fixed (ECEF) frame, an idealized geostationary satellite
            # is stationary relative to Earth's rotating surface at its nominal subsatellite
            # longitude lambda_0 on the equatorial plane (z = 0).
            # Nominal geostationary orbital radius: R ~ 42,164,140 m.
            # Its ECEF position is stationary: [R * cos(lambda_0), R * sin(lambda_0), 0].
            # To construct the physical orbital frame (RIC basis), the in-track (along-track)
            # direction is defined by the satellite's eastward prograde orbital motion in space.
            # Angular velocity magnitude: omega_Earth = 7.292115e-5 rad/s (WGS84).
            # Inertial velocity projected in ECEF frame gives: [-v_orb * sin(lambda_0), v_orb * cos(lambda_0), 0].
            r_mag = 42164140.0  # metres
            omega_earth = 7.292115e-5  # rad/s
            v_orb = omega_earth * r_mag  # ~ 3074.6 m/s
            lam_rad = np.radians(self.geo_longitude_deg)

            pos_x = np.full(n_epochs, r_mag * np.cos(lam_rad), dtype=np.float64)
            pos_y = np.full(n_epochs, r_mag * np.sin(lam_rad), dtype=np.float64)
            pos_z = np.zeros(n_epochs, dtype=np.float64)

            vel_x = np.full(n_epochs, -v_orb * np.sin(lam_rad), dtype=np.float64)
            vel_y = np.full(n_epochs, v_orb * np.cos(lam_rad), dtype=np.float64)
            vel_z = np.zeros(n_epochs, dtype=np.float64)

        else:
            # Medium Earth Orbit (GPS/Galileo nominal Keplerian model):
            # Semi-major axis a ~ 26,560 km, inclination i ~ 55 deg, period T ~ 43,082 s (11h 58m).
            # Documented nominal approximation with elapsed phase from initial epoch.
            a_mag = 26560000.0  # metres
            period_sec = 43082.0
            omega = 2.0 * np.pi / period_sec
            inc_rad = np.radians(55.0)
            phase = omega * elapsed_sec

            # Orbital plane coordinates
            x_orb = a_mag * np.cos(phase)
            y_orb = a_mag * np.sin(phase)
            vx_orb = -a_mag * omega * np.sin(phase)
            vy_orb = a_mag * omega * np.cos(phase)

            # Rotate by inclination around X-axis into equatorial plane
            pos_x = x_orb
            pos_y = y_orb * np.cos(inc_rad)
            pos_z = y_orb * np.sin(inc_rad)

            vel_x = vx_orb
            vel_y = vy_orb * np.cos(inc_rad)
            vel_z = vy_orb * np.sin(inc_rad)

        position_ecef = np.column_stack([pos_x, pos_y, pos_z])
        velocity_ecef = np.column_stack([vel_x, vel_y, vel_z])
        return position_ecef, velocity_ecef


class ProvidedStateProvider(OrbitalStateProvider):
    """Provides orbital state vectors from a supplied ephemeris or orbit-state table.

    Accepts a DataFrame containing position and velocity columns:
      - Timestamp index or 'utc_time' column
      - position_x, position_y, position_z (or x, y, z)
      - velocity_x, velocity_y, velocity_z (or vx, vy, vz)
    Interpolates linearly to requested timestamps.
    """

    def __init__(self, state_df: pd.DataFrame):
        df = state_df.copy()
        if "utc_time" in df.columns:
            df["utc_time"] = pd.to_datetime(df["utc_time"])
            df = df.set_index("utc_time")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        self.state_df = df.sort_index()

    @property
    def source_name(self) -> str:
        return "provided"

    def get_state(
        self,
        timestamps: Sequence[pd.Timestamp] | pd.DatetimeIndex,
        satellite_id: str = "",
        orbit_class: str = "MEO",
    ) -> Tuple[np.ndarray, np.ndarray]:
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
        if len(ts) == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

        cols = {c.lower(): c for c in self.state_df.columns}
        pos_cols = [
            cols.get("position_x", cols.get("x", None)),
            cols.get("position_y", cols.get("y", None)),
            cols.get("position_z", cols.get("z", None)),
        ]
        vel_cols = [
            cols.get("velocity_x", cols.get("vx", None)),
            cols.get("velocity_y", cols.get("vy", None)),
            cols.get("velocity_z", cols.get("vz", None)),
        ]

        if any(c is None for c in pos_cols):
            raise ValueError(f"state_df missing required position columns: {list(self.state_df.columns)}")

        src_times = (self.state_df.index - self.state_df.index[0]).total_seconds().to_numpy(dtype=np.float64)
        target_times = (ts - self.state_df.index[0]).total_seconds().to_numpy(dtype=np.float64)

        pos_res = np.column_stack([
            np.interp(target_times, src_times, self.state_df[col].to_numpy(dtype=np.float64))
            for col in pos_cols
        ])

        if all(c is not None for c in vel_cols):
            vel_res = np.column_stack([
                np.interp(target_times, src_times, self.state_df[col].to_numpy(dtype=np.float64))
                for col in vel_cols
            ])
        else:
            if len(target_times) > 1:
                vel_res = np.gradient(pos_res, target_times, axis=0)
            else:
                vel_res = np.zeros_like(pos_res)

        return pos_res, vel_res


def approximate_nominal_orbit(
    timestamps: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    orbit_class: str = "MEO",
    satellite_id: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates nominal ECEF position and velocity vectors for orbit frame reconstruction.

    NOTE: This is an explicitly approximate geometry fallback. Never represents measured ephemeris.
    """
    provider = NominalStateProvider()
    return provider.get_state(timestamps, satellite_id=satellite_id, orbit_class=orbit_class)


# Backwards compatibility alias
nominal_satellite_orbit = approximate_nominal_orbit


def get_orbital_state_provider(source: str = "nominal_approximation", **kwargs) -> OrbitalStateProvider:
    """Factory resolving configured OrbitalStateProvider instance."""
    if source in ("nominal", "nominal_approximation"):
        return NominalStateProvider(**kwargs)
    if source == "provided":
        state_df = kwargs.get("state_df")
        if state_df is None:
            raise ValueError("ProvidedStateProvider requires 'state_df' parameter.")
        return ProvidedStateProvider(state_df=state_df)
    raise ValueError(f"Unknown orbital state provider source: '{source}'")


def build_ric_features(
    error_ecef_df: pd.DataFrame,
    timestamps: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    provider: Optional[OrbitalStateProvider] = None,
    satellite_id: str = "",
    orbit_class: str = "MEO",
) -> pd.DataFrame:
    """Computes Radial, In-track, Cross-track (RIC) error features from ECEF errors.

    Args:
        error_ecef_df: DataFrame containing 'x_error_m', 'y_error_m', 'z_error_m'.
        timestamps: Associated observation timestamps.
        provider: OrbitalStateProvider instance (defaults to NominalStateProvider).
        satellite_id: Satellite identifier.
        orbit_class: 'GEO', 'MEO', etc.

    Returns:
        DataFrame with columns ['ric_r', 'ric_i', 'ric_c'].
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    state_provider = provider or get_orbital_state_provider("nominal_approximation")
    pos, vel = state_provider.get_state(ts, satellite_id=satellite_id, orbit_class=orbit_class)

    req_cols = ["x_error_m", "y_error_m", "z_error_m"]
    for c in req_cols:
        if c not in error_ecef_df.columns:
            raise ValueError(f"Missing required ECEF error column: '{c}'")

    err_ecef = error_ecef_df[req_cols].to_numpy(dtype=np.float64)
    ric_err = ecef_error_to_ric(err_ecef, pos, vel)

    return pd.DataFrame(
        ric_err,
        columns=["ric_r", "ric_i", "ric_c"],
        index=error_ecef_df.index,
    )


def build_physics_features(
    timestamps: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    orbital_state_provider: Optional[OrbitalStateProvider] = None,
    satellite_id: str = "",
    orbit_class: str = "MEO",
    features: Tuple[str, ...] = ("sun_beta_angle", "shadow_factor", "solar_cos_angle"),
) -> pd.DataFrame:
    """Constructs deterministic physics and solar geometry features for model ingestion.

    Guarantees strict training/inference parity and zero target error leakage.

    Features:
      - sun_beta_angle (deg): Elevation angle of Sun above orbital plane [-90.0, +90.0].
      - shadow_factor (fraction): Cylindrical/conical Earth eclipse factor [0.0, 1.0].
      - solar_cos_angle (cosine): Cosine of angle between geocentric satellite position and Sun [-1.0, 1.0].
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    feature_list = list(features)
    if len(ts) == 0:
        return pd.DataFrame(columns=feature_list)
    if ts.isna().any():
        raise ValueError("timestamps contains null or unparsable values")

    provider = orbital_state_provider or get_orbital_state_provider("nominal_approximation")
    pos, vel = provider.get_state(ts, satellite_id=satellite_id, orbit_class=orbit_class)

    if len(pos) != len(ts) or pos.shape[-1] != 3:
        raise ValueError(f"Invalid position shape: {pos.shape}, expected ({len(ts)}, 3)")

    # Validate non-zero vectors
    pos_norm = np.linalg.norm(pos, axis=-1)
    if np.any(pos_norm < 1.0):
        raise ValueError("Orbital position vector contains near-zero lengths (< 1.0 m)")

    data: Dict[str, np.ndarray] = {}
    for f in feature_list:
        if f == "sun_beta_angle":
            beta = compute_sun_beta_angle(pos, vel, ts)
            data["sun_beta_angle"] = np.clip(beta, -90.0, 90.0)
        elif f == "shadow_factor":
            shadow = compute_shadow_factor(pos, ts)
            data["shadow_factor"] = np.clip(shadow, 0.0, 1.0)
        elif f == "solar_cos_angle":
            cos_ang = compute_solar_cos_angle(pos, ts)
            data["solar_cos_angle"] = np.clip(cos_ang, -1.0, 1.0)
        else:
            raise ValueError(f"Unsupported physics feature: '{f}'")

    df_phys = pd.DataFrame(data, index=ts)
    if not np.all(np.isfinite(df_phys.to_numpy())):
        raise ValueError("Generated physics features contain non-finite values")
    return df_phys



def compute_sun_beta_angle(
    position_ecef: np.ndarray,
    velocity_ecef: np.ndarray,
    timestamps: np.ndarray | pd.DatetimeIndex,
) -> np.ndarray:
    """Computes Sun elevation angle (beta angle) above the satellite orbital plane in degrees.
    
    Beta angle is a primary driver of solar radiation pressure (SRP) variations and eclipse seasons.
    """
    pos = np.asarray(position_ecef, dtype=np.float64)
    vel = np.asarray(velocity_ecef, dtype=np.float64)
    sun_unit = approximate_sun_unit_ecef(timestamps)
    
    # Orbit normal unit vector: h = r x v / ||r x v||
    orbit_h = np.cross(pos, vel)
    norm_h = np.linalg.norm(orbit_h, axis=-1, keepdims=True)
    norm_h = np.where(norm_h <= 1e-12, 1.0, norm_h)
    h_unit = orbit_h / norm_h
    
    # sin(beta) = s . h
    sin_beta = np.einsum('...i,...i->...', sun_unit, h_unit)
    sin_beta = np.clip(sin_beta, -1.0, 1.0)
    return np.degrees(np.arcsin(sin_beta))


def compute_shadow_factor(
    position_ecef: np.ndarray,
    timestamps: np.ndarray | pd.DatetimeIndex,
    r_earth_m: float = 6378137.0,
    penumbra_margin_m: float = 50000.0,
) -> np.ndarray:
    """Computes Earth shadow factor (0.0 = fully illuminated, 1.0 = full umbra shadow).
    
    Uses cylindrical/conical shadow geometry based on satellite position and Sun direction.
    """
    pos = np.asarray(position_ecef, dtype=np.float64)
    sun_unit = approximate_sun_unit_ecef(timestamps)
    
    # Projection along Sun direction
    # Positive means satellite is on Sun-facing side of Earth (illuminated)
    proj_sun = np.einsum('...i,...i->...', pos, sun_unit)
    
    shadow = np.zeros(len(pos), dtype=np.float64)
    night_mask = proj_sun < 0
    
    if np.any(night_mask):
        # Perpendicular distance to Earth-Sun line
        perp_vec = pos[night_mask] - proj_sun[night_mask, None] * sun_unit[night_mask]
        dist_perp = np.linalg.norm(perp_vec, axis=-1)
        
        # Umbra and penumbra boundaries
        r_umbra = r_earth_m
        r_penumbra = r_earth_m + penumbra_margin_m
        
        full_shadow = dist_perp <= r_umbra
        penumbra = (dist_perp > r_umbra) & (dist_perp < r_penumbra)
        
        shadow_night = np.zeros(np.sum(night_mask), dtype=np.float64)
        shadow_night[full_shadow] = 1.0
        if np.any(penumbra):
            fraction = (r_penumbra - dist_perp[penumbra]) / penumbra_margin_m
            shadow_night[penumbra] = np.clip(fraction, 0.0, 1.0)
            
        shadow[night_mask] = shadow_night
        
    return shadow


def compute_sisre(
    radial_error: np.ndarray,
    along_track_error: np.ndarray,
    cross_track_error: np.ndarray,
    clock_error: np.ndarray,
    orbit_class: str = 'MEO',
    clock_in_seconds: bool = False,
) -> np.ndarray:
    """Computes Signal-in-Space Ranging Error (SISRE) according to IGS/DLR/Galileo standards.
    
    Formula:
        SISRE = sqrt( (w_R * Delta_R - Delta_t)^2 + w_A^2 * Delta_A^2 + w_C^2 * Delta_C^2 )
        
    Weights:
        - MEO (GPS, Galileo, BDS MEO): w_R = 0.98, w_A = 0.143 (w_A^2 = 1/49), w_C = 0.143
        - GEO (BDS GEO, QZSS GEO): w_R = 0.99, w_A = 0.088 (w_A^2 = 1/127), w_C = 0.088
    """
    dr = np.asarray(radial_error, dtype=np.float64)
    da = np.asarray(along_track_error, dtype=np.float64)
    dc = np.asarray(cross_track_error, dtype=np.float64)
    dt = np.asarray(clock_error, dtype=np.float64)
    
    if clock_in_seconds:
        dt = dt * SPEED_OF_LIGHT_M_S
        
    orbit_type = str(orbit_class).upper()
    if 'GEO' in orbit_type:
        w_r = 0.99
        w_a_sq = 1.0 / 127.0
        w_c_sq = 1.0 / 127.0
    else:  # MEO default
        w_r = 0.98
        w_a_sq = 1.0 / 49.0
        w_c_sq = 1.0 / 49.0
        
    radial_clock_term = np.square(w_r * dr - dt)
    along_term = w_a_sq * np.square(da)
    cross_term = w_c_sq * np.square(dc)
    
    return np.sqrt(radial_clock_term + along_term + cross_term)


