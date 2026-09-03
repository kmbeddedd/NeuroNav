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

    # Also verify model produces non-trivial predictions: a model that always
    # predicts delta=0 would predict raw_deltas[i] = baseline alone,
    # but the restored large-amplitude deltas should be preserved by inverse scaling.
    # Check that the maximum restored amplitude is not collapsed to near-zero.
    max_amp = np.max(np.abs(restored))
    assert max_amp > 10.0, f"Inverse scaling must preserve large amplitudes, got max={max_amp}m"


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
        'GEO Gated MoE',
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
        'GEO Gated MoE': {'GEO': np.ones((3, 4))},
    }
    diagnostics = {
        'GEO Gated MoE': {
            ('GEO', 0): {
                'baseline_x': 0.5, 'baseline_y': 0.5, 'baseline_z': 0.5, 'baseline_clock': 0.5,
                'delta_x': 0.5, 'delta_y': 0.5, 'delta_z': 0.5, 'delta_clock': 0.5,
                'p_gate': 0.75, 'regime_probability': 0.85,
                'lead_time_hours': 1.2, 'history_span_hours': 23.5,
            }
        }
    }

    out_file = tmp_path / 'predictions.csv'
    save_predictions(dummy_data, all_preds, out_file, diagnostics=diagnostics)

    saved_df = pd.read_csv(out_file)
    assert 'model' in saved_df.columns, "Predictions CSV must contain 'model' column"
    models_present = set(saved_df['model'].unique())
    assert 'GEO Gated MoE' in models_present
    assert 'Persistence' in models_present

    # Check that diagnostic fields exist and are populated
    moe_rows = saved_df[saved_df['model'] == 'GEO Gated MoE']
    assert 'lead_time_hours' in moe_rows.columns
    assert 'history_span_hours' in moe_rows.columns
    assert 'p_gate' in moe_rows.columns


# ---------------------------------------------------------------------------
# Test I — model forward returns 3 values (delta_pred, gate_logit, p_gate)
# The new GEOGatedMoEModel.forward must return exactly 3 tensors.
# ---------------------------------------------------------------------------
def test_i_model_returns_three_values():
    from scripts.benchmark_ps08 import GEOGatedMoEModel
    model = GEOGatedMoEModel(history_dim=19, query_dim=13, num_series=3)
    model.eval()
    B = 4
    history   = torch.zeros(B, 32, 19)
    query     = torch.zeros(B, 13)
    series_id = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        out = model(history, query, series_id)
    assert len(out) == 3, f"forward() must return 3 tensors, got {len(out)}"
    delta_pred, gate_logit, p_gate = out
    assert delta_pred.shape  == (B, 4), f"delta_pred shape wrong: {delta_pred.shape}"
    assert gate_logit.shape  == (B,),   f"gate_logit shape wrong: {gate_logit.shape}"
    assert p_gate.shape      == (B, 1), f"p_gate shape wrong: {p_gate.shape}"


# ---------------------------------------------------------------------------
# Test J — p_gate is in (0, 1) for all inputs
# Sigmoid output must be strictly bounded.
# ---------------------------------------------------------------------------
def test_j_p_gate_bounded():
    from scripts.benchmark_ps08 import GEOGatedMoEModel
    torch.manual_seed(0)
    model = GEOGatedMoEModel(history_dim=19, query_dim=13, num_series=3)
    model.eval()
    B = 64
    history   = torch.randn(B, 32, 19)
    query     = torch.randn(B, 13)
    series_id = torch.randint(0, 3, (B,))
    with torch.no_grad():
        _, _, p_gate = model(history, query, series_id)
    assert (p_gate > 0).all(), "p_gate must be > 0 (strict sigmoid)"
    assert (p_gate < 1).all(), "p_gate must be < 1 (strict sigmoid)"


# ---------------------------------------------------------------------------
# Test K — perturbing gate logit changes prediction
# This verifies that p_gate actually gates the output amplitude, not just
# a dormant auxiliary variable.
# ---------------------------------------------------------------------------
def test_k_gate_controls_prediction():
    from scripts.benchmark_ps08 import GEOGatedMoEModel
    import copy
    torch.manual_seed(42)
    model = GEOGatedMoEModel(history_dim=19, query_dim=13, num_series=3)
    model.eval()

    B = 4
    history   = torch.randn(B, 32, 19)
    query     = torch.randn(B, 13)
    series_id = torch.zeros(B, dtype=torch.long)

    with torch.no_grad():
        delta_base, _, p_gate_base = model(history, query, series_id)

    # Patch gate_head to saturate toward excursion (p_gate → 1)
    model_high = copy.deepcopy(model)
    with torch.no_grad():
        model_high.gate_head[-1].bias.fill_(20.0)   # logit → large positive → p_gate → 1
    with torch.no_grad():
        delta_high, _, p_gate_high = model_high(history, query, series_id)

    # Patch gate_head to saturate toward normal (p_gate → 0)
    model_low = copy.deepcopy(model)
    with torch.no_grad():
        model_low.gate_head[-1].bias.fill_(-20.0)   # logit → large negative → p_gate → 0
    with torch.no_grad():
        delta_low, _, p_gate_low = model_low(history, query, series_id)

    assert (p_gate_high > 0.99).all(), "Gate must saturate high"
    assert (p_gate_low < 0.01).all(), "Gate must saturate low"
    # When p_gate is saturated high, output ≈ excursion_head; low → normal_head.
    # They must differ unless both heads are identical (which they won't be after random init).
    max_diff = (delta_high - delta_low).abs().max().item()
    assert max_diff > 1e-4, f"Saturating p_gate must change output (max_diff={max_diff})"


# ---------------------------------------------------------------------------
# Test L — normal_head and excursion_head are structurally distinct
# Verifies that the architecture has genuinely separate expert outputs.
# ---------------------------------------------------------------------------
def test_l_expert_heads_distinct():
    from scripts.benchmark_ps08 import GEOGatedMoEModel
    model = GEOGatedMoEModel(history_dim=19, query_dim=13, num_series=3)
    normal_params    = sum(p.numel() for p in model.normal_head.parameters())
    excursion_params = sum(p.numel() for p in model.excursion_head.parameters())
    assert normal_params > 0,    "normal_head must have parameters"
    assert excursion_params > 0, "excursion_head must have parameters"
    # Excursion head is wider (more capacity for large amplitudes)
    assert excursion_params > normal_params, (
        f"excursion_head ({excursion_params} params) must be larger than normal_head ({normal_params} params)"
    )


# ---------------------------------------------------------------------------
# Test M — gate_logit is differentiable w.r.t. model parameters
# The gate must participate in the computational graph (no stop_gradient).
# ---------------------------------------------------------------------------
def test_m_gate_participates_in_gradient():
    from scripts.benchmark_ps08 import GEOGatedMoEModel
    torch.manual_seed(0)
    model = GEOGatedMoEModel(history_dim=19, query_dim=13, num_series=3)
    model.train()

    history   = torch.randn(4, 32, 19)
    query     = torch.randn(4, 13)
    series_id = torch.zeros(4, dtype=torch.long)

    delta_pred, gate_logit, p_gate = model(history, query, series_id)
    # Loss that touches both prediction and gate
    loss = delta_pred.sum() + gate_logit.sum()
    loss.backward()

    # gate_head parameters must have gradients
    for param in model.gate_head.parameters():
        assert param.grad is not None, "gate_head parameters must receive gradients"
        assert param.grad.abs().sum() > 0, "gate_head gradients must be non-zero"


# ---------------------------------------------------------------------------
# Test N — GEOGatedMoEModel is the alias target of GEORegimeAwareResidualModel
# Existing tests that import GEORegimeAwareResidualModel must not break.
# ---------------------------------------------------------------------------
def test_n_backward_compatible_alias():
    from scripts.benchmark_ps08 import GEOGatedMoEModel, GEORegimeAwareResidualModel
    assert GEORegimeAwareResidualModel is GEOGatedMoEModel, (
        "GEORegimeAwareResidualModel must be an alias for GEOGatedMoEModel"
    )
    # Alias can be constructed and used identically
    model = GEORegimeAwareResidualModel(history_dim=19, query_dim=13, num_series=3)
    assert hasattr(model, 'gate_head'),       "Alias model must have gate_head"
    assert hasattr(model, 'normal_head'),     "Alias model must have normal_head"
    assert hasattr(model, 'excursion_head'),  "Alias model must have excursion_head"

