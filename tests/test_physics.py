import numpy as np
import pandas as pd
from src.physics import (
    ecef_error_to_ric,
    ric_basis,
    ric_error_to_ecef,
    approximate_sun_unit_ecef,
    compute_solar_cos_angle,
    fit_harmonic_orbit_baseline,
)

def test_ric_basis_is_orthonormal_and_round_trips():
    position = np.array([[26000000.0, 1000000.0, 4000000.0]])
    velocity = np.array([[-200.0, 3000.0, 600.0]])
    error = np.array([[2.0, -3.0, 5.0]])
    basis = ric_basis(position, velocity)
    identity = np.einsum('...ji,...jk->...ik', basis, basis)
    np.testing.assert_allclose(identity, np.eye(3)[None], atol=1e-12)
    ric = ecef_error_to_ric(error, position, velocity)
    np.testing.assert_allclose(ric_error_to_ecef(ric, position, velocity), error, atol=1e-12)

def test_approximate_sun_unit_and_solar_cos_angle():
    ts = pd.date_range('2026-01-01 00:00:00', periods=96, freq='15min')
    sun_vec = approximate_sun_unit_ecef(ts)
    assert sun_vec.shape == (96, 3)
    # Norm of unit vectors must be 1.0
    norms = np.linalg.norm(sun_vec, axis=-1)
    np.testing.assert_allclose(norms, np.ones(96), atol=1e-6)

    sat_pos = np.repeat([[26000000.0, 0.0, 0.0]], 96, axis=0)
    cos_ang = compute_solar_cos_angle(sat_pos, ts)
    assert cos_ang.shape == (96,)
    assert np.all(cos_ang >= -1.0) and np.all(cos_ang <= 1.0)

def test_fit_harmonic_orbit_baseline_extrapolation():
    # Construct a known 12-hour harmonic signal with linear drift
    t_in = np.arange(96) * 900.0  # 24 hours
    t_out = np.arange(96, 192) * 900.0  # next 24 hours
    omega = 2.0 * np.pi / (12.0 * 3600.0)
    
    # Pure sinusoidal signal
    y_true_in = np.sin(omega * t_in)[:, None]
    y_true_out = np.sin(omega * t_out)[:, None]
    
    y_pred_out = fit_harmonic_orbit_baseline(t_in, y_true_in, t_out, periods_hours=(12.0, 24.0))
    assert y_pred_out.shape == (96, 1)
    # The extrapolation should match the harmonic function with high precision
    np.testing.assert_allclose(y_pred_out, y_true_out, atol=1e-2)

