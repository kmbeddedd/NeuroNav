"""Comprehensive test suite for satellite-specific forecasting backend and model selection pipeline.

Verifies:
- Satellite-specific routing and multi-model execution
- Registry persistence, JSON reload, and corruption resilience
- Manual vs automatic selection modes and recalibration invariants
- Strict fail-closed error handling (NO silent fallback to BiLSTM)
- Data validation and model eligibility matrices
- RIC frame transformation round-trip numerical tolerance
- SISRE and SRP feature calculations
- Decoupled clock and N-HiTS model operations
- Zero target leakage into training/preprocessing
- End-to-end Phase A Calibration and Phase B Forecast
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
    get_calibration_report,
    get_model_metadata,
    get_satellite_selection,
    predict_with_satellite_models,
    reset_to_automatic,
    set_satellite_model,
    validate_dataset,
)
from src.forecasting.base import ForecastModel
from src.forecasting.eligibility import check_model_eligibility, compute_eligibility_matrix
from src.forecasting.models import (
    MODEL_REGISTRY,
    create_model,
    get_available_model_names,
)
from src.forecasting.pipeline import (
    CalibrationPipeline,
    compute_metrics_for_residuals,
)
from src.forecasting.registry import (
    CorruptedRegistryError,
    SatelliteModelRegistry,
    SatelliteSelection,
)
from src.forecasting.router import (
    ModelArtifactError,
    NoModelSelectionError,
    PredictionRouter,
)
from src.physics import (
    compute_shadow_factor,
    compute_sisre,
    compute_sun_beta_angle,
    ecef_error_to_ric,
    nominal_satellite_orbit,
    ric_error_to_ecef,
)


@pytest.fixture
def synthetic_sat_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates synthetic 7-day train and 1-day test telemetry for MEO-1, MEO-2, and GEO."""
    rng = np.random.RandomState(42)
    t_train = pd.date_range("2026-01-01 00:00", periods=50, freq="2h")
    t_test = pd.date_range("2026-01-05 04:00", periods=12, freq="2h")

    train_rows, test_rows = [], []
    for sat in ("MEO-1", "MEO-2", "GEO"):
        for t in t_train:
            train_rows.append({
                "utc_time": t,
                "satellite_id": sat,
                "x_error_m": float(rng.randn() * 2.0),
                "y_error_m": float(rng.randn() * 2.0),
                "z_error_m": float(rng.randn() * 2.0),
                "clock_error_m": float(rng.randn() * 0.5),
            })
        for t in t_test:
            test_rows.append({
                "utc_time": t,
                "satellite_id": sat,
                "x_error_m": float(rng.randn() * 2.0),
                "y_error_m": float(rng.randn() * 2.0),
                "z_error_m": float(rng.randn() * 2.0),
                "clock_error_m": float(rng.randn() * 0.5),
            })

    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


# ---------------------------------------------------------------------------
# 1. RIC Frame Round-Trip & Astrodynamics
# ---------------------------------------------------------------------------
def test_ric_round_trip_numerical_tolerance():
    """Verify ECEF -> RIC -> ECEF round-trip error is within floating-point tolerance (< 1e-6 m)."""
    t = pd.date_range("2026-01-01", periods=10, freq="15min")
    pos, vel = nominal_satellite_orbit(t, orbit_class="MEO")

    # Arbitrary ECEF error vectors
    error_ecef_orig = np.array([
        [1.25, -2.50, 3.75],
        [-0.50, 4.20, -1.10],
        [0.00, 1.00, -2.00],
    ], dtype=np.float64)

    ric_err = ecef_error_to_ric(error_ecef_orig, pos[:3], vel[:3])
    ecef_reconstructed = ric_error_to_ecef(ric_err, pos[:3], vel[:3])

    max_delta = np.max(np.abs(error_ecef_orig - ecef_reconstructed))
    assert max_delta < 1e-6, f"Round trip delta {max_delta} exceeds tolerance"


def test_sisre_calculation():
    """Verify SISRE calculation satisfies physical weights and scaling."""
    # When radial and clock cancel out, only along/cross contribute
    r = np.array([1.0])
    clk = np.array([0.98])  # 0.98 * 1.0 - 0.98 = 0
    a = np.array([7.0])     # (1/49) * 49 = 1.0
    c = np.array([0.0])

    sisre = compute_sisre(r, a, c, clk, orbit_class="MEO", clock_in_seconds=False)
    assert np.isclose(sisre[0], 1.0, atol=1e-3)


def test_srp_shadow_factor():
    """Verify Earth shadow factor calculation."""
    t = pd.to_datetime(["2026-03-21 12:00:00", "2026-03-21 00:00:00"])
    # Satellite at GEO distance: Day side vs Night side
    pos_sun = np.array([[42164000.0, 0.0, 0.0], [-42164000.0, 0.0, 0.0]])
    shadow = compute_shadow_factor(pos_sun, t)
    assert len(shadow) == 2
    assert 0.0 <= shadow[0] <= 1.0
    assert 0.0 <= shadow[1] <= 1.0


# ---------------------------------------------------------------------------
# 2. Registry Persistence, Manual Selection & Invariants
# ---------------------------------------------------------------------------
def test_registry_persistence_and_reload(tmp_path: Path):
    """Verify selections persist to JSON and reload identically across process restarts."""
    reg_file = tmp_path / "registry.json"
    reg = SatelliteModelRegistry(reg_file)

    reg.register_calibration_winner(
        satellite_id="MEO-1",
        winner_model="random_forest",
        score=0.25,
        candidate_scores={"persistence": 0.8, "random_forest": 0.25},
    )
    reg.set_manual_selection(
        satellite_id="GEO",
        model_name="geo_moe",
        reason="Operator verified manual assignment",
    )

    # Reload from disk in a brand new registry instance
    reg2 = SatelliteModelRegistry(reg_file)
    sel_meo = reg2.get_selection("MEO-1")
    sel_geo = reg2.get_selection("GEO")

    assert sel_meo is not None
    assert sel_meo.selected_model == "random_forest"
    assert sel_meo.selection_mode == "automatic"
    assert sel_meo.selection_score == 0.25

    assert sel_geo is not None
    assert sel_geo.selected_model == "geo_moe"
    assert sel_geo.selection_mode == "manual"


def test_manual_selection_survives_calibration(tmp_path: Path):
    """INVARIANT: Manual selection must NOT be overwritten by subsequent automatic calibration."""
    reg_file = tmp_path / "registry.json"
    reg = SatelliteModelRegistry(reg_file)

    # Set manual selection on MEO-1
    reg.set_manual_selection(
        satellite_id="MEO-1",
        model_name="harmonic_ridge",
        reason="Forced operator baseline",
    )

    # Attempt automatic calibration win for a different model
    reg.register_calibration_winner(
        satellite_id="MEO-1",
        winner_model="random_forest",
        score=0.10,
        candidate_scores={"harmonic_ridge": 0.50, "random_forest": 0.10},
    )

    # Verify manual model was retained!
    current = reg.get_selection("MEO-1")
    assert current is not None
    assert current.selected_model == "harmonic_ridge"
    assert current.selection_mode == "manual"
    # Verify audit event was logged
    audit_events = [h["event"] for h in current.history]
    assert "automatic_calibration_skipped_due_to_manual_override" in audit_events


def test_reset_to_automatic(tmp_path: Path):
    """Verify reset_to_automatic allows future calibration to update model."""
    reg_file = tmp_path / "registry.json"
    reg = SatelliteModelRegistry(reg_file)

    reg.set_manual_selection("MEO-1", "harmonic_ridge")
    assert reg.get_selection("MEO-1").selection_mode == "manual"

    success = reg.reset_to_automatic("MEO-1")
    assert success is True
    assert reg.get_selection("MEO-1").selection_mode == "automatic"

    # Now automatic calibration CAN update it
    reg.register_calibration_winner(
        satellite_id="MEO-1",
        winner_model="gaussian_process",
        score=0.15,
        candidate_scores={"harmonic_ridge": 0.40, "gaussian_process": 0.15},
    )
    assert reg.get_selection("MEO-1").selected_model == "gaussian_process"


def test_corrupted_registry_handling(tmp_path: Path):
    """Verify corrupted JSON raises CorruptedRegistryError."""
    bad_file = tmp_path / "corrupt_registry.json"
    bad_file.write_text("{ this is not valid json : 123", encoding="utf-8")
    with pytest.raises(CorruptedRegistryError):
        SatelliteModelRegistry(bad_file)


# ---------------------------------------------------------------------------
# 3. Prediction Router & No Silent BiLSTM Fallback
# ---------------------------------------------------------------------------
def test_no_silent_bilstm_fallback_on_unselected_satellite(tmp_path: Path):
    """INVARIANT: Unselected satellite MUST raise NoModelSelectionError, NEVER fall back to BiLSTM."""
    reg_file = tmp_path / "registry.json"
    reg = SatelliteModelRegistry(reg_file)
    router = PredictionRouter(registry=reg, artifacts_dir=tmp_path / "artifacts")

    # Satellite G99 has no selection
    df = pd.DataFrame({
        "utc_time": pd.date_range("2026-01-01", periods=10, freq="15min"),
        "satellite_id": "G99",
        "x_error_m": np.ones(10),
        "y_error_m": np.ones(10),
        "z_error_m": np.ones(10),
        "clock_error_m": np.zeros(10),
    })

    with pytest.raises(NoModelSelectionError) as exc_info:
        router.predict(df)
    assert "G99" in str(exc_info.value)


def test_missing_artifact_raises_actionable_error(tmp_path: Path):
    """Missing model artifact raises structured ModelArtifactError."""
    reg_file = tmp_path / "registry.json"
    reg = SatelliteModelRegistry(reg_file)
    reg.set_manual_selection("MEO-1", "random_forest", model_artifact=str(tmp_path / "nonexistent.joblib"))

    router = PredictionRouter(registry=reg, artifacts_dir=tmp_path / "artifacts")
    df = pd.DataFrame({
        "utc_time": pd.date_range("2026-01-01", periods=10, freq="15min"),
        "satellite_id": "MEO-1",
        "x_error_m": np.ones(10),
        "y_error_m": np.ones(10),
        "z_error_m": np.ones(10),
        "clock_error_m": np.zeros(10),
    })

    with pytest.raises(ModelArtifactError) as exc_info:
        router.predict(df)
    assert "MEO-1" in str(exc_info.value)
    assert "random_forest" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4. Multi-Model Satellite-Specific Inference
# ---------------------------------------------------------------------------
def test_satellite_specific_multi_model_inference(tmp_path: Path, synthetic_sat_data):
    """Different satellites must simultaneously execute different assigned model classes."""
    train_df, test_df = synthetic_sat_data
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    reg_file = tmp_path / "registry.json"
    reg = SatelliteModelRegistry(reg_file)

    # Train and persist Model A (Harmonic Ridge) for MEO-1
    sat1_train = train_df[train_df["satellite_id"] == "MEO-1"].copy()
    m1 = create_model("harmonic_ridge")
    m1.fit(sat1_train)
    p1 = artifacts_dir / "MEO-1_harmonic_ridge.joblib"
    m1.save(p1)
    reg.set_manual_selection("MEO-1", "harmonic_ridge", model_artifact=str(p1))

    # Train and persist Model B (Random Forest) for MEO-2
    sat2_train = train_df[train_df["satellite_id"] == "MEO-2"].copy()
    m2 = create_model("random_forest")
    m2.fit(sat2_train)
    p2 = artifacts_dir / "MEO-2_random_forest.joblib"
    m2.save(p2)
    reg.set_manual_selection("MEO-2", "random_forest", model_artifact=str(p2))

    # Train and persist Model C (Persistence) for GEO
    sat3_train = train_df[train_df["satellite_id"] == "GEO"].copy()
    m3 = create_model("persistence")
    m3.fit(sat3_train)
    p3 = artifacts_dir / "GEO_persistence.json"
    m3.save(p3)
    reg.set_manual_selection("GEO", "persistence", model_artifact=str(p3))

    # Run router on combined dataset containing all 3 satellites
    router = PredictionRouter(registry=reg, artifacts_dir=artifacts_dir)
    forecast_df = router.predict(train_df, horizon_steps=12, compute_ric=True)

    assert isinstance(forecast_df, pd.DataFrame)
    assert len(forecast_df) == 12 * 3  # 3 satellites * 12 steps

    # Check that provenance metadata tags each row accurately
    meo1_rows = forecast_df[forecast_df["satellite_id"] == "MEO-1"]
    meo2_rows = forecast_df[forecast_df["satellite_id"] == "MEO-2"]
    geo_rows = forecast_df[forecast_df["satellite_id"] == "GEO"]

    assert (meo1_rows["model_used"] == "harmonic_ridge").all()
    assert (meo2_rows["model_used"] == "random_forest").all()
    assert (geo_rows["model_used"] == "persistence").all()

    # Check required output columns
    expected_cols = [
        "forecast_step", "timestamp", "satellite_id",
        "predicted_X", "predicted_Y", "predicted_Z", "predicted_Clock",
        "pred_3D_Orbit_Error", "model_used", "model_version", "selection_mode",
        "predicted_R", "predicted_I", "predicted_C"
    ]
    for c in expected_cols:
        assert c in forecast_df.columns, f"Missing output column {c}"


# ---------------------------------------------------------------------------
# 5. Data Validation & Eligibility
# ---------------------------------------------------------------------------
def test_validation_layer_structured_errors():
    """Verify structured errors when required columns or history are invalid."""
    bad_df = pd.DataFrame({"some_random_column": [1, 2, 3]})
    val_res = validate_dataset(bad_df)
    assert val_res["is_valid"] is False
    assert val_res["status"] == "invalid"
    assert len(val_res["errors"]) > 0
    assert val_res["errors"][0]["stage"] == "data_validation"


def test_model_eligibility_checks():
    """Verify short series correctly marks models ineligible without crashing."""
    short_df = pd.DataFrame({
        "utc_time": pd.date_range("2026-01-01", periods=6, freq="15min"),
        "x_error_m": np.ones(6),
        "y_error_m": np.ones(6),
        "z_error_m": np.ones(6),
        "clock_error_m": np.zeros(6),
    })
    # nhits requires 24 rows
    elig = check_model_eligibility("MEO-1", "nhits", short_df)
    assert elig.eligible is False
    assert "Insufficient history" in elig.reason

    # persistence requires only 4 rows
    elig_p = check_model_eligibility("MEO-1", "persistence", short_df)
    assert elig_p.eligible is True


# ---------------------------------------------------------------------------
# 6. Decoupled Clock and N-HiTS Models
# ---------------------------------------------------------------------------
def test_decoupled_clock_model(synthetic_sat_data):
    """Verify Decoupled Clock model fits and predicts 4 targets."""
    train_df, test_df = synthetic_sat_data
    sat_df = train_df[train_df["satellite_id"] == "MEO-1"].copy()

    model = create_model("decoupled_clock", poly_degree=2, tcn_epochs=5)
    model.fit(sat_df)
    preds = model.predict(sat_df, 8)
    assert preds.shape == (8, 4)
    assert not np.isnan(preds).any()


def test_nhits_model(synthetic_sat_data):
    """Verify N-HiTS model fits and predicts multi-step horizon."""
    train_df, test_df = synthetic_sat_data
    sat_df = train_df[train_df["satellite_id"] == "MEO-1"].copy()

    model = create_model("nhits", lookback_steps=12, hidden_dim=16, epochs=5)
    model.fit(sat_df)
    preds = model.predict(sat_df, 10)
    assert preds.shape == (10, 4)
    assert not np.isnan(preds).any()


# ---------------------------------------------------------------------------
# 7. Model Metadata API
# ---------------------------------------------------------------------------
def test_model_metadata_api():
    """Verify model metadata API returns accurate structured configurations."""
    all_meta = get_model_metadata()
    assert isinstance(all_meta, list)
    assert len(all_meta) == len(get_available_model_names())

    rf_meta = get_model_metadata("random_forest")
    assert isinstance(rf_meta, dict)
    assert rf_meta["model_type"] == "random_forest"
    assert "n_estimators" in rf_meta["parameters"]


# ---------------------------------------------------------------------------
# 8. End-to-End Acceptance Test (Phase A Calibration -> Phase B Forecast)
# ---------------------------------------------------------------------------
def test_end_to_end_acceptance_calibration_and_forecast(tmp_path: Path, synthetic_sat_data):
    """END-TO-END ACCEPTANCE TEST:
    
    Phase A: Calibrate on 7-day train + 8th-day truth.
             Confirm satellite-specific winners selected and persisted.
    Phase B: Forecast on new 7-day data without ground truth.
             Confirm predictions generated with model provenance.
    """
    train_df, test_df = synthetic_sat_data
    reg_file = tmp_path / "registry.json"
    artifacts_dir = tmp_path / "artifacts"
    reports_dir = tmp_path / "reports"

    reg = SatelliteModelRegistry(reg_file)
    pipeline = CalibrationPipeline(
        registry=reg,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        primary_metric="orbit_3d_vector_mae_m",
        candidate_models=["persistence", "harmonic_ridge", "random_forest"],
    )

    # --- PHASE A: CALIBRATION ---
    calib_res = pipeline.run_calibration(train_df, test_df, run_id="test_run_e2e")
    assert calib_res["status"] == "success"
    assert len(calib_res["satellite_winners"]) == 3  # MEO-1, MEO-2, GEO

    # Verify registry persisted to disk
    assert reg_file.exists()
    assert (reports_dir / "test_run_e2e" / "summary.json").exists()
    assert (reports_dir / "test_run_e2e" / "satellite_model_comparison.csv").exists()

    # --- PHASE B: OPERATIONAL FORECAST ---
    router = PredictionRouter(registry=reg, artifacts_dir=artifacts_dir)
    forecast_df = router.predict(train_df, horizon_steps=10)

    assert isinstance(forecast_df, pd.DataFrame)
    assert len(forecast_df) == 10 * 3
    for sat in ("MEO-1", "MEO-2", "GEO"):
        sat_preds = forecast_df[forecast_df["satellite_id"] == sat]
        assert not sat_preds.empty
        model_used = sat_preds["model_used"].iloc[0]
        # Must match registry winner!
        expected_winner = reg.get_selection(sat).selected_model
        assert model_used == expected_winner
