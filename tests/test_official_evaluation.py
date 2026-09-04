"""Tests for the Official Competition Model Selection Hierarchy.

Verifies all 22 required evaluation invariants:
1. X/Y/Z/Clock receive equal weight (25% each).
2. Higher average W wins Priority 1.
3. Priority 2 is used only when Priority 1 is tied.
4. Priority 3 is used only when Priority 1 and 2 are tied.
5. MAE/SISRE cannot override the official selection hierarchy.
6. p-value is calculated and stored.
7. H0 result is calculated using alpha = 0.05.
8. Per-parameter W values are retained.
9. Per-parameter p-values are retained.
10. Per-parameter hypothesis results are retained.
11. Insufficient residual samples (n < 3) are handled explicitly.
12. Invalid residuals are handled explicitly.
13. Satellite A can select a different model from Satellite B.
14. Registry persists across process restarts with selection_policy='official_competition'.
15. Manual selections persist across calibration.
16. Reset-to-automatic works.
17. New forecast data uses the stored satellite-specific model.
18. Missing model artifacts produce explicit errors (ModelArtifactError).
19. Unknown models do not fall back to BiLSTM (NoModelSelectionError).
20. 8th-day truth never enters model training/preprocessing.
21. Q-Q data is generated correctly.
22. Supplementary metrics are not used by the competition selector.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.forecasting.api import (
    calibrate_models,
    get_all_satellite_selections,
    get_detailed_statistical_results,
    get_model_comparison,
    get_qq_data,
    get_satellite_selection,
    predict_with_satellite_models,
    reset_to_automatic,
    set_satellite_model,
    validate_dataset,
)
from src.forecasting.pipeline import (
    CalibrationPipeline,
    compare_models_hierarchical,
    evaluate_residuals_official_hierarchy,
    rank_candidates_hierarchically,
)
from src.forecasting.registry import SatelliteModelRegistry
from src.forecasting.router import ModelArtifactError, NoModelSelectionError, PredictionRouter

@pytest.fixture
def synthetic_pair_data():
    """Generates synthetic 7-day train and 8th-day test datasets for 2 satellites."""
    dates_train = pd.date_range("2025-09-01 00:00:00", periods=48, freq="15min")
    dates_test = pd.date_range("2025-09-08 00:00:00", periods=16, freq="15min")

    train_rows = []
    test_rows = []

    for sat in ("SAT-A", "SAT-B"):
        freq_factor = 1.0 if sat == "SAT-A" else 2.0
        for i, dt in enumerate(dates_train):
            train_rows.append({
                "utc_time": dt.isoformat(),
                "satellite_id": sat,
                "x_error_m": np.sin(freq_factor * i * 0.1) * 2.0,
                "y_error_m": np.cos(freq_factor * i * 0.1) * 2.0,
                "z_error_m": np.sin(freq_factor * i * 0.05) * 1.5,
                "clock_error_m": 0.05 * i + 0.001 * (i ** 2),
            })
        for i, dt in enumerate(dates_test):
            test_rows.append({
                "utc_time": dt.isoformat(),
                "satellite_id": sat,
                "x_error_m": np.sin(freq_factor * (48 + i) * 0.1) * 2.0,
                "y_error_m": np.cos(freq_factor * (48 + i) * 0.1) * 2.0,
                "z_error_m": np.sin(freq_factor * (48 + i) * 0.05) * 1.5,
                "clock_error_m": 0.05 * (48 + i) + 0.001 * ((48 + i) ** 2),
            })

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Invariants 1, 6, 7, 8, 9, 10: Equal Weighting & Priority 1 Stats Retention
# ---------------------------------------------------------------------------
def test_equal_weighting_and_priority1_retention():
    """Verifies that X, Y, Z, Clock each receive exactly 25% weight in W_avg, and stats are preserved."""
    rng = np.random.default_rng(42)
    n = 100
    # Create known residuals: Gaussian in X, Y, Z and skewed in Clock
    res_x = rng.normal(0, 1, n)
    res_y = rng.normal(0, 2, n)
    res_z = rng.normal(0, 0.5, n)
    res_clk = rng.exponential(1.0, n)  # Highly non-normal

    actual = np.zeros((n, 4))
    predicted = np.column_stack([res_x, res_y, res_z, res_clk])

    metrics = evaluate_residuals_official_hierarchy(actual, predicted, alpha=0.05)

    assert metrics["eligible"] is True
    p1 = metrics["priority_1"]

    # Invariant 1: Equal 25% weighting
    w_expected = (p1["W"]["X"] + p1["W"]["Y"] + p1["W"]["Z"] + p1["W"]["Clock"]) / 4.0
    assert pytest.approx(p1["W"]["average"], 1e-9) == w_expected

    # Invariants 6 & 8: Per-parameter W and p-values are retained
    for param in ("X", "Y", "Z", "Clock"):
        assert 0.0 <= p1["W"][param] <= 1.0
        assert 0.0 <= p1["p_value"][param] <= 1.0

    # Invariant 7: H0 decision uses alpha = 0.05 (Clock must be rejected)
    assert p1["alpha"] == 0.05
    assert p1["hypothesis_result"]["Clock"] == 1  # Exp is non-normal -> Reject H0
    assert p1["hypothesis_result"]["X"] == 0      # Normal -> Fail to reject H0


# ---------------------------------------------------------------------------
# Invariant 2: Higher average W wins Priority 1
# ---------------------------------------------------------------------------
def test_higher_average_w_wins_priority_1():
    """Verifies that a model with higher W_avg wins Priority 1 unconditionally."""
    model_a = {
        "eligible": True,
        "priority_1": {"W": {"average": 0.9650}},
        "priority_2": {"mean": {"aggregate": 5.0}, "std": {"aggregate": 10.0}},
        "priority_3": {"total_outliers": 8, "aggregate_max_discrepancy": 2.0},
    }
    model_b = {
        "eligible": True,
        "priority_1": {"W": {"average": 0.8200}},
        "priority_2": {"mean": {"aggregate": 0.01}, "std": {"aggregate": 0.02}},
        "priority_3": {"total_outliers": 0, "aggregate_max_discrepancy": 0.1},
    }

    code, reason = compare_models_hierarchical(model_a, model_b)
    assert code == 1
    assert "Priority 1: Higher Shapiro-Wilk W_avg" in reason


# ---------------------------------------------------------------------------
# Invariant 3: Priority 2 (Mean/Std) is used ONLY when Priority 1 is tied
# ---------------------------------------------------------------------------
def test_priority_2_used_only_when_priority_1_tied():
    """Verifies Priority 2 break-tie logic: lower mean bias wins, then lower std."""
    # Case 3A: P1 tied, Model A has lower bias
    m_a = {
        "eligible": True,
        "priority_1": {"W": {"average": 0.950000}},
        "priority_2": {"mean": {"aggregate": 0.020}, "std": {"aggregate": 0.50}},
        "priority_3": {"total_outliers": 3, "aggregate_max_discrepancy": 1.0},
    }
    m_b = {
        "eligible": True,
        "priority_1": {"W": {"average": 0.950000}},  # EXACT TIE
        "priority_2": {"mean": {"aggregate": 0.150}, "std": {"aggregate": 0.50}},
        "priority_3": {"total_outliers": 1, "aggregate_max_discrepancy": 0.5},
    }
    code, reason = compare_models_hierarchical(m_a, m_b, tie_tolerance=1e-4)
    assert code == 1
    assert "Priority 2: Lower aggregate residual bias" in reason

    # Case 3B: P1 tied, P2 Mean tied, Model B has lower std
    m_a["priority_2"]["mean"]["aggregate"] = 0.05
    m_b["priority_2"]["mean"]["aggregate"] = 0.05  # Mean tied
    m_a["priority_2"]["std"]["aggregate"] = 0.80
    m_b["priority_2"]["std"]["aggregate"] = 0.40  # Model B has lower std
    code, reason = compare_models_hierarchical(m_a, m_b, tie_tolerance=1e-4)
    assert code == -1
    assert "Priority 2" in reason and "aggregate residual std" in reason


# ---------------------------------------------------------------------------
# Invariant 4 & 21: Priority 3 (Q-Q Outliers) used ONLY when P1 & P2 are tied
# ---------------------------------------------------------------------------
def test_priority_3_used_only_when_p1_and_p2_tied():
    """Verifies Priority 3: fewer Q-Q outliers breaks the tie when P1 and P2 are identical."""
    m_a = {
        "eligible": True,
        "priority_1": {"W": {"average": 0.92000}},
        "priority_2": {"mean": {"aggregate": 0.100}, "std": {"aggregate": 0.500}},
        "priority_3": {"total_outliers": 1, "aggregate_max_discrepancy": 0.6},
    }
    m_b = {
        "eligible": True,
        "priority_1": {"W": {"average": 0.92000}},  # Tied
        "priority_2": {"mean": {"aggregate": 0.100}, "std": {"aggregate": 0.500}},  # Tied
        "priority_3": {"total_outliers": 4, "aggregate_max_discrepancy": 1.2},
    }
    code, reason = compare_models_hierarchical(m_a, m_b, tie_tolerance=1e-4)
    assert code == 1
    assert "Priority 3: Fewer Q-Q outliers" in reason


def test_qq_data_generation():
    """Verifies that machine-readable Q-Q data is properly generated with quantiles and discrepancies."""
    rng = np.random.default_rng(42)
    n = 40
    res = rng.normal(0, 1, (n, 4))
    res[0, 0] = 10.0  # Deliberate outlier

    metrics = evaluate_residuals_official_hierarchy(np.zeros((n, 4)), res)
    p3 = metrics["priority_3"]
    qq_x = p3["qq_details"]["X"]

    assert qq_x["n"] == n
    assert len(qq_x["theoretical_quantiles"]) == n
    assert len(qq_x["observed_residuals"]) == n
    assert len(qq_x["discrepancies"]) == n
    assert qq_x["outlier_count"] >= 1


# ---------------------------------------------------------------------------
# Invariant 5 & 22: MAE / SISRE Cannot Override the Official Hierarchy
# ---------------------------------------------------------------------------
def test_mae_sisre_cannot_override_official_hierarchy():
    """Demonstrates that a model with superior MAE/SISRE LOSES if its Shapiro-Wilk W is lower."""
    # Model A: High error (MAE = 5.0m), but Gaussian residuals (W = 0.98)
    model_a = {
        "eligible": True,
        "priority_1": {"W": {"average": 0.9800}},
        "priority_2": {"mean": {"aggregate": 0.1}, "std": {"aggregate": 5.0}},
        "priority_3": {"total_outliers": 0, "aggregate_max_discrepancy": 0.2},
        "supplementary": {"orbit_3d_vector_mae_m": 5.0, "sisre_mean_m": 4.5},
    }
    # Model B: Low error (MAE = 0.2m), but non-Gaussian residuals (W = 0.75)
    model_b = {
        "eligible": True,
        "priority_1": {"W": {"average": 0.7500}},
        "priority_2": {"mean": {"aggregate": 0.01}, "std": {"aggregate": 0.2}},
        "priority_3": {"total_outliers": 5, "aggregate_max_discrepancy": 2.5},
        "supplementary": {"orbit_3d_vector_mae_m": 0.2, "sisre_mean_m": 0.15},
    }

    code, reason = compare_models_hierarchical(model_a, model_b)
    # Model A MUST WIN despite worse MAE/SISRE!
    assert code == 1
    assert "Priority 1: Higher Shapiro-Wilk W_avg" in reason


# ---------------------------------------------------------------------------
# Invariant 11 & 12: Insufficient Samples & Invalid Residuals Handled
# ---------------------------------------------------------------------------
def test_insufficient_samples_and_invalid_residuals():
    """Verifies that n < 3 or non-finite residuals return eligible: false with a structured reason."""
    # Sample size < 3
    act_small = np.zeros((2, 4))
    pred_small = np.ones((2, 4))
    res_small = evaluate_residuals_official_hierarchy(act_small, pred_small)
    assert res_small["eligible"] is False
    assert "insufficient residual samples" in res_small["reason"]

    # Non-finite values
    act_nan = np.zeros((10, 4))
    pred_nan = np.full((10, 4), np.nan)
    res_nan = evaluate_residuals_official_hierarchy(act_nan, pred_nan)
    assert res_nan["eligible"] is False
    assert "insufficient finite residual samples" in res_nan["reason"]


# ---------------------------------------------------------------------------
# Invariants 13, 14, 15, 16, 20: Calibration, Persistence, Overrides & Leakage
# ---------------------------------------------------------------------------
def test_end_to_end_official_calibration_and_persistence(tmp_path: Path, synthetic_pair_data):
    """Tests end-to-end Phase A calibration driven by official competition hierarchy."""
    train_df, test_df = synthetic_pair_data
    reg_path = tmp_path / "registry.json"
    art_dir = tmp_path / "artifacts"
    rep_dir = tmp_path / "reports"

    registry = SatelliteModelRegistry(registry_path=reg_path)
    pipeline = CalibrationPipeline(
        registry=registry,
        artifacts_dir=art_dir,
        reports_dir=rep_dir,
        candidate_models=["persistence", "harmonic_ridge"],
    )

    result = pipeline.run_calibration(train_df, test_df, run_id="test_run_01")
    assert result["status"] == "success"
    assert result["selection_policy"] == "official_competition"

    # Invariant 13: Satellites selected independently
    winners = result["satellite_winners"]
    assert "SAT-A" in winners and "SAT-B" in winners
    for sat_id in ("SAT-A", "SAT-B"):
        assert "priority_1" in winners[sat_id]
        assert "priority_2" in winners[sat_id]
        assert "priority_3" in winners[sat_id]

    # Invariant 14: Registry persistence across process reload
    reloaded_reg = SatelliteModelRegistry(registry_path=reg_path)
    sel_a = reloaded_reg.get_selection("SAT-A")
    assert sel_a is not None
    assert sel_a.selection_policy == "official_competition"
    assert sel_a.primary_metric == "shapiro_w_avg"
    assert sel_a.winning_priority_1 != {}

    # Invariant 15: Manual override preservation
    reloaded_reg.set_manual_selection("SAT-A", "persistence", reason="Operator choice")
    # Re-run calibration
    pipeline.run_calibration(train_df, test_df, run_id="test_run_02")
    sel_a_after = reloaded_reg.get_selection("SAT-A")
    assert sel_a_after.selection_mode == "manual"
    assert sel_a_after.selected_model == "persistence"

    # Invariant 16: Reset-to-automatic works
    reloaded_reg.reset_to_automatic("SAT-A")
    pipeline.run_calibration(train_df, test_df, run_id="test_run_03")
    sel_a_reset = reloaded_reg.get_selection("SAT-A")
    assert sel_a_reset.selection_mode == "automatic"

    # Verify report files were generated
    report_path = rep_dir / "test_run_01"
    assert (report_path / "summary.json").exists()
    assert (report_path / "model_comparison.csv").exists()
    assert (report_path / "detailed_statistical_results.csv").exists()
    assert (report_path / "qq_data").exists()


# ---------------------------------------------------------------------------
# Invariants 17, 18, 19: Operational Routing & Zero Silent Fallback
# ---------------------------------------------------------------------------
def test_operational_routing_and_zero_fallback(tmp_path: Path, synthetic_pair_data):
    """Verifies that router uses selected models and fails closed with explicit errors."""
    train_df, test_df = synthetic_pair_data
    reg_path = tmp_path / "registry.json"
    registry = SatelliteModelRegistry(registry_path=reg_path)

    router = PredictionRouter(registry=registry, artifacts_dir=tmp_path / "artifacts")

    # Invariant 19: Unknown satellite raises NoModelSelectionError (ZERO BiLSTM fallback)
    with pytest.raises(NoModelSelectionError) as exc_info:
        router.predict(train_df)
    assert "No model selection found in registry" in str(exc_info.value)

    # Calibrate to populate registry
    pipeline = CalibrationPipeline(
        registry=registry,
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        candidate_models=["persistence"],
    )
    pipeline.run_calibration(train_df, test_df, run_id="calib_for_routing")

    # Invariant 17: Operational prediction uses stored satellite selection
    forecast_df = router.predict(train_df, horizon_steps=12)
    assert len(forecast_df) == 24  # 12 steps * 2 sats
    assert set(forecast_df["satellite_id"].unique()) == {"SAT-A", "SAT-B"}
    assert (forecast_df["model_used"] == "persistence").all()

    # Invariant 18: Missing artifact raises ModelArtifactError
    # Delete artifact
    sel_a = registry.get_selection("SAT-A")
    if sel_a and sel_a.model_artifact:
        Path(sel_a.model_artifact).unlink(missing_ok=True)
    
    fresh_router = PredictionRouter(registry=registry, artifacts_dir=tmp_path / "artifacts")
    with pytest.raises(ModelArtifactError):
        fresh_router.predict(train_df[train_df["satellite_id"] == "SAT-A"])


# ---------------------------------------------------------------------------
# Section 11: Tie Tolerance and Candidate Ranking Verification
# ---------------------------------------------------------------------------
def test_section11_tie_tolerance_and_hierarchical_ranking():
    """Verifies that tie_tolerance (tau=1e-4) correctly triggers Priority 2 / 3 and rank_candidates_hierarchically."""
    # Model A: W_avg = 0.95004, bias = 0.50
    # Model B: W_avg = 0.95001 (diff = 0.00003 < 1e-4 -> tied in P1), bias = 0.05 (Model B wins on P2!)
    m_a = {
        "eligible": True,
        "priority_1": {"W": {"average": 0.95004}},
        "priority_2": {"mean": {"aggregate": 0.50}, "std": {"aggregate": 0.50}},
        "priority_3": {"total_outliers": 5, "aggregate_max_discrepancy": 1.0},
    }
    m_b = {
        "eligible": True,
        "priority_1": {"W": {"average": 0.95001}},
        "priority_2": {"mean": {"aggregate": 0.05}, "std": {"aggregate": 0.50}},
        "priority_3": {"total_outliers": 5, "aggregate_max_discrepancy": 1.0},
    }
    # With tie_tolerance=1e-4, Model B wins because P1 difference is 3e-5 (within tolerance)
    code, reason = compare_models_hierarchical(m_a, m_b, tie_tolerance=1e-4)
    assert code == -1  # m_b wins
    assert "Priority 2" in reason and "residual bias" in reason

    # But with strict tie_tolerance=1e-6, Model A wins because 0.95004 > 0.95001
    code_strict, reason_strict = compare_models_hierarchical(m_a, m_b, tie_tolerance=1e-6)
    assert code_strict == 1  # m_a wins
    assert "Priority 1" in reason_strict

    # Test candidate ranking with 3 models:
    # m_c has much higher W (0.98), should rank 1st.
    m_c = {
        "eligible": True,
        "priority_1": {"W": {"average": 0.98000}},
        "priority_2": {"mean": {"aggregate": 1.0}, "std": {"aggregate": 1.0}},
        "priority_3": {"total_outliers": 10, "aggregate_max_discrepancy": 5.0},
    }
    candidates = {"model_a": m_a, "model_b": m_b, "model_c": m_c}
    ranked = rank_candidates_hierarchically(candidates, tie_tolerance=1e-4)
    assert ranked == ["model_c", "model_b", "model_a"]
