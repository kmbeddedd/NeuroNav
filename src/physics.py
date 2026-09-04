from __future__ import annotations
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
    ts = pd.to_datetime(timestamps)
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

