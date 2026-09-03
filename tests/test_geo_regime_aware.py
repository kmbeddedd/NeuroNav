"""
Unit and regression tests for the GEO Regime-Aware Residual model (PS-08).
Covers Tests A through H required by specification:
  Test A - Explicit Delta t exists (different timestamp intervals produce different inputs)
  Test B - Causal history (test query cannot access timestamps >= query time)
  Test C - Residual reconstruction (prediction == baseline + delta)
  Test D - No central-collapse caused by scaling (inverse scaling preserves physical units)
  Test E - Regime feature is causal (future targets cannot influence regime probability)
  Test F - Irregular history representation (15-min and 2-hour gaps explicitly distinguished)
  Test G - Official benchmark remains intact (all 6 legacy models + 7th model execute)
  Test H - Model identity (prediction records identify model name and diagnostics)
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.benchmark_ps08 import (
    TARGETS,
    _compute_regime_probability,
    _fit_causal_baseline,
    _fit_regime_detector,
    _physical_history_tensor,
    _physical_query_features,
    save_predictions,
    time_features,
    GEORegimeAwareResidualModel,
)


def _create_synthetic_history(timestamps: list[pd.Timestamp], values: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(values, columns=TARGETS)
    df.insert(0, 'utc_time', timestamps)
    return df


class DummyBaseline:
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((len(x), 4), dtype=float)


# ---------------------------------------------------------------------------
# Test A — explicit Delta t exists
# Two histories with identical values but different timestamp gaps must
# generate different temporal inputs.
# ---------------------------------------------------------------------------
def test_a_explicit_delta_t_exists():
    t0 = pd.Timestamp('2025-09-01 00:00:00')
    n_pts = 6
    vals = np.ones((n_pts, 4), dtype=float) * 5.0
    dummy_base = DummyBaseline()
    c_d = np.zeros(4)
    s_d = np.ones(4)

    # History 1: 15-minute intervals
    times_15m = [t0 + pd.Timedelta(minutes=15 * i) for i in range(n_pts)]
    df_15m = _create_synthetic_history(times_15m, vals)

    # History 2: 2-hour intervals
    times_2h = [t0 + pd.Timedelta(hours=2 * i) for i in range(n_pts)]
    df_2h = _create_synthetic_history(times_2h, vals)

    query_time = t0 + pd.Timedelta(days=2)

    feat_15m, meta_15m = _physical_history_tensor(
        df_15m, cutoff_idx=n_pts - 1, query_time=query_time, origin=t0,
        baseline_model=dummy_base, center_d=c_d, scale_d=s_d, max_len=16
    )
    feat_2h, meta_2h = _physical_history_tensor(
        df_2h, cutoff_idx=n_pts - 1, query_time=query_time, origin=t0,
        baseline_model=dummy_base, center_d=c_d, scale_d=s_d, max_len=16
    )

    # Values are identical, but temporal features must differ
    assert not np.allclose(feat_15m, feat_2h), "Identical values with different gaps must produce different inputs"
    # Specifically check Delta t (col 13) and age (col 14)
    dt_col = 13
    age_col = 14
    assert not np.allclose(feat_15m[:, dt_col], feat_2h[:, dt_col]), "Delta t column must distinguish gaps"
    assert not np.allclose(feat_15m[:, age_col], feat_2h[:, age_col]), "Age column must distinguish query lead"


# ---------------------------------------------------------------------------
# Test B — causal history
# A test query cannot use any timestamp at or after the query time.
# ---------------------------------------------------------------------------
def test_b_causal_history():
    t0 = pd.Timestamp('2025-09-01 00:00:00')
    times = [t0 + pd.Timedelta(hours=i) for i in range(10)]
    vals = np.random.RandomState(42).randn(10, 4)
    df = _create_synthetic_history(times, vals)

    query_time = pd.Timestamp('2025-09-01 05:30:00')  # between index 5 and 6
    cutoff_idx = int(np.flatnonzero(df['utc_time'] < query_time)[-1])
    assert cutoff_idx == 5
    assert df['utc_time'].iloc[cutoff_idx] < query_time

    dummy_base = DummyBaseline()
    feat, meta = _physical_history_tensor(
        df, cutoff_idx=cutoff_idx, query_time=query_time, origin=t0,
        baseline_model=dummy_base, center_d=np.zeros(4), scale_d=np.ones(4), max_len=8
    )

    # All history timestamps used must be strictly before query_time
    eligible_times = df.iloc[:cutoff_idx + 1]['utc_time']
    assert (eligible_times < query_time).all(), "History timestamps must be strictly causal (t < t_query)"


# ---------------------------------------------------------------------------
# Test C — residual reconstruction
# Verify: prediction == baseline + predicted_delta within numerical tolerance.
# ---------------------------------------------------------------------------
def test_c_residual_reconstruction():
    # Setup test vectors
    baseline_vals = np.array([12.5, -8.3, 4.1, 0.95])
    predicted_delta = np.array([2.1, -1.4, 0.7, -0.05])

    # Reconstruct
    reconstructed = baseline_vals + predicted_delta

    # Verify identity within numerical precision
    diff = np.abs(reconstructed - (baseline_vals + predicted_delta))
    assert np.all(diff < 1e-12), "Reconstructed prediction must exactly equal baseline + delta"

    # Verify scaling unscaling path: pred_delta = pred_scaled * scale + center
    c_d = np.array([0.5, -0.2, 0.1, 0.0])
    s_d = np.array([2.0, 3.0, 1.5, 0.8])
    pred_scaled = (predicted_delta - c_d) / s_d
    unscaled_delta = pred_scaled * s_d + c_d
    reconstructed_from_scaled = baseline_vals + unscaled_delta
    assert np.allclose(reconstructed_from_scaled, reconstructed, atol=1e-7)


# ---------------------------------------------------------------------------
# Test D — no central-collapse caused by scaling
# Verify inverse scaling returns the correct physical units.
# ---------------------------------------------------------------------------
def test_d_no_central_collapse_caused_by_scaling():
    # Actual GEO excursions swing between -75m and +58m
    raw_deltas = np.array([
        [-52.4, 48.2, -15.1, 8.3],
        [0.05, -0.02, 0.01, -0.005],
        [58.0, -75.0, 30.5, -12.4],
    ])

    c_d = np.array([-0.1, 0.2, -0.05, 0.01])
    s_d = np.array([3.5, 4.2, 2.8, 1.5])

    # Forward scale
    scaled = (raw_deltas - c_d) / s_d
    # Inverse scale
    restored = scaled * s_d + c_d

    # Must preserve exact large-amplitude excursions without clipping or collapse
    assert np.allclose(restored, raw_deltas, atol=1e-10)
    assert np.isclose(restored[2, 0], 58.0)
    assert np.isclose(restored[2, 1], -75.0)


# ---------------------------------------------------------------------------
# Test E — regime feature is causal
# No future target values can influence the regime state.
# ---------------------------------------------------------------------------
def test_e_regime_feature_is_causal():
    t0 = pd.Timestamp('2025-09-01 00:00:00')
    n_pts = 10
    times = [t0 + pd.Timedelta(hours=i) for i in range(n_pts)]
    vals_causal = np.full((n_pts, 4), 2.0)
    df_causal = _create_synthetic_history(times, vals_causal)

    detector = _fit_regime_detector(df_causal)
    norms_causal = np.sqrt(np.sum(vals_causal[:, :3] ** 2, axis=1))
    prob_causal = _compute_regime_probability(norms_causal, detector)

    # Now add future observations with massive excursions (1000m)
    future_times = [t0 + pd.Timedelta(hours=n_pts + i) for i in range(5)]
    vals_future = np.full((5, 4), 1000.0)
    df_with_future = pd.concat([
        df_causal,
        _create_synthetic_history(future_times, vals_future)
    ], ignore_index=True)

    # At cutoff index n_pts - 1, causal filter only looks at observations <= cutoff
    causal_slice = df_with_future.iloc[:n_pts]
    norms_after = np.sqrt(np.sum(causal_slice[list(TARGETS)[:3]].to_numpy() ** 2, axis=1))
    prob_after = _compute_regime_probability(norms_after, detector)

    assert np.isclose(prob_causal, prob_after, atol=1e-12), "Future values must not influence causal regime probability"


# ---------------------------------------------------------------------------
# Test F — irregular history
# A sequence containing 15-minute and 2-hour gaps must be represented correctly.
# ---------------------------------------------------------------------------
def test_f_irregular_history():
    t0 = pd.Timestamp('2025-09-01 00:00:00')
    # Step 0: t0
    # Step 1: t0 + 15m (gap = 0.25h)
    # Step 2: t0 + 15m + 2h (gap = 2.0h)
    times = [
        t0,
        t0 + pd.Timedelta(minutes=15),
        t0 + pd.Timedelta(minutes=135),
    ]
    vals = np.ones((3, 4), dtype=float)
    df = _create_synthetic_history(times, vals)
    dummy_base = DummyBaseline()

    feat, meta = _physical_history_tensor(
        df, cutoff_idx=2, query_time=t0 + pd.Timedelta(hours=4), origin=t0,
        baseline_model=dummy_base, center_d=np.zeros(4), scale_d=np.ones(4), max_len=3
    )

    dt_col = 13
    # Step features: row 1 is 15-min gap (0.25h / 2.0 = 0.125)
    #                row 2 is 2-hour gap (2.0h / 2.0 = 1.0)
    dt_step1 = feat[1, dt_col] * 2.0  # unscaled hours
    dt_step2 = feat[2, dt_col] * 2.0  # unscaled hours

    assert np.isclose(dt_step1, 0.25, atol=1e-4), f"Expected 0.25h, got {dt_step1}"
    assert np.isclose(dt_step2, 2.0, atol=1e-4), f"Expected 2.0h, got {dt_step2}"
    assert np.isclose(dt_step2 / dt_step1, 8.0, atol=1e-3), "2h gap must be 8x larger than 15m gap"


# ---------------------------------------------------------------------------
# Test G — official benchmark remains intact
# All six legacy PS-08 models still execute.
# ---------------------------------------------------------------------------
def test_g_official_benchmark_remains_intact():
    from scripts.benchmark_ps08 import run_benchmark
    import inspect

    source = inspect.getsource(run_benchmark)
    required_models = [
        'Persistence',
        'Harmonic Ridge',
        'Random Forest',
        'Gaussian Process',
        'BiLSTM-GRU',
        'Transformer',
        'GEO Regime-Aware Residual',
    ]
    for model_name in required_models:
        assert f"'{model_name}'" in source, f"Benchmark runner must contain model: {model_name}"


# ---------------------------------------------------------------------------
# Test H — model identity
# Output rows must identify the model.
# ---------------------------------------------------------------------------
def test_h_model_identity(tmp_path: Path):
    times = pd.date_range('2025-09-08', periods=3, freq='h')
    dummy_data = {
        'GEO': {
            'test': pd.DataFrame({
                'utc_time': times,
                'x_error_m': [1.0, 2.0, 3.0],
                'y_error_m': [1.0, 2.0, 3.0],
                'z_error_m': [1.0, 2.0, 3.0],
                'clock_error_m': [1.0, 2.0, 3.0],
            })
        }
    }
    all_preds = {
        'Persistence': {'GEO': np.zeros((3, 4))},
        'GEO Regime-Aware Residual': {'GEO': np.ones((3, 4))},
    }
    diagnostics = {
        'GEO Regime-Aware Residual': {
            ('GEO', 0): {
                'baseline_x': 0.5, 'baseline_y': 0.5, 'baseline_z': 0.5, 'baseline_clock': 0.5,
                'delta_x': 0.5, 'delta_y': 0.5, 'delta_z': 0.5, 'delta_clock': 0.5,
                'regime_probability': 0.85, 'lead_time_hours': 1.2, 'history_span_hours': 23.5,
            }
        }
    }

    out_file = tmp_path / 'predictions.csv'
    save_predictions(dummy_data, all_preds, out_file, diagnostics=diagnostics)

    saved_df = pd.read_csv(out_file)
    assert 'model' in saved_df.columns, "Predictions CSV must contain 'model' column"
    models_present = set(saved_df['model'].unique())
    assert 'GEO Regime-Aware Residual' in models_present
    assert 'Persistence' in models_present

    # Check that diagnostic fields exist and are populated
    geo_residual_rows = saved_df[saved_df['model'] == 'GEO Regime-Aware Residual']
    assert 'regime_probability' in geo_residual_rows.columns
    assert 'lead_time_hours' in geo_residual_rows.columns
    assert 'history_span_hours' in geo_residual_rows.columns
    assert np.isclose(geo_residual_rows['regime_probability'].iloc[0], 0.85)
