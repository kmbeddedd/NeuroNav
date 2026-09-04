"""Integration tests for satellite-specific independent upload, physics features, and calibration pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecasting.api import (
    get_satellite_model,
    predict_satellite,
    train_satellite,
    validate_satellite_dataset,
)
from src.forecasting.features import (
    FeatureManifest,
    build_inference_features,
    build_training_features,
)
from src.forecasting.models.harmonic_ridge import HarmonicRidgeModel
from src.forecasting.models.random_forest import RandomForestModel
from src.forecasting.pipeline import (
    CalibrationPipeline,
    compare_models_hierarchical,
    evaluate_residuals_official_hierarchy,
)
from src.forecasting.registry import SatelliteModelRegistry, get_satellite_artifact_dir
from src.forecasting.validation import (
    SatelliteDataset,
    infer_orbit_type,
    regularize_cadence,
    validate_satellite_dataset as run_sat_val,
)
from src.physics import (
    NominalStateProvider,
    ProvidedStateProvider,
    build_ric_features,
    get_orbital_state_provider,
)

@pytest.fixture
def synthetic_geo_csv(tmp_path):
    """Creates a synthetic GEO CSV without a satellite_id column."""
    p = tmp_path / "satellite_geo_upload.csv"
    times = pd.date_range("2026-01-01 00:00:00", periods=50, freq=pd.Timedelta(minutes=15))
    df = pd.DataFrame({
        "utc_time": times,
        "x_error_m": np.sin(np.linspace(0, 4 * np.pi, 50)) * 2.0,
        "y_error_m": np.cos(np.linspace(0, 4 * np.pi, 50)) * 1.5,
        "z_error_m": np.sin(np.linspace(0, 2 * np.pi, 50)) * 0.5,
        "clock_error_m": np.linspace(0.1, 1.0, 50),
    })
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def synthetic_geo_test_csv(tmp_path):
    """Creates a synthetic Day 8 test CSV for GEO."""
    p = tmp_path / "satellite_geo_test.csv"
    times = pd.date_range("2026-01-01 12:30:00", periods=20, freq=pd.Timedelta(minutes=15))
    df = pd.DataFrame({
        "utc_time": times,
        "x_error_m": np.sin(np.linspace(4 * np.pi, 6 * np.pi, 20)) * 2.0,
        "y_error_m": np.cos(np.linspace(4 * np.pi, 6 * np.pi, 20)) * 1.5,
        "z_error_m": np.sin(np.linspace(2 * np.pi, 3 * np.pi, 20)) * 0.5,
        "clock_error_m": np.linspace(1.0, 1.5, 20),
    })
    df.to_csv(p, index=False)
    return p


def test_csv_without_satellite_id_accepted(synthetic_geo_csv):
    """Test 1: CSV with no satellite_id column accepted via explicit parameters."""
    sat_ds = run_sat_val(synthetic_geo_csv, satellite_id="CUSTOM_GEO", orbit_type="GEO")
    assert isinstance(sat_ds, SatelliteDataset)
    assert sat_ds.satellite_id == "CUSTOM_GEO"
    assert sat_ds.orbit_type == "GEO"
    assert "satellite_id" in sat_ds.dataframe.columns
    assert len(sat_ds.dataframe) == 50


def test_filename_fallback_resolves_satellite_and_orbit(tmp_path):
    """Test 2: Filename fallback correctly infers satellite ID and orbit class."""
    geo_path = tmp_path / "DATA_GEO_Train.csv"
    times = pd.date_range("2026-01-01", periods=10, freq="15min")
    pd.DataFrame({
        "utc_time": times,
        "x_error_m": 1.0,
        "y_error_m": 2.0,
        "z_error_m": 3.0,
        "clock_error_m": 4.0,
    }).to_csv(geo_path, index=False)

    ds = run_sat_val(geo_path)
    assert ds.satellite_id == "GEO"
    assert ds.orbit_type == "GEO"

    meo_path = tmp_path / "DATA_MEO_Train.csv"
    pd.DataFrame({
        "utc_time": times,
        "x_error_m": 1.0,
        "y_error_m": 2.0,
        "z_error_m": 3.0,
        "clock_error_m": 4.0,
    }).to_csv(meo_path, index=False)

    ds_meo = run_sat_val(meo_path)
    assert ds_meo.satellite_id == "MEO-1"
    assert ds_meo.orbit_type == "MEO"


def test_irregular_cadence_detected():
    """Test 3: Irregular cadence detected and SamplingMetadata populated."""
    # Create irregular timestamps: 10m, then 2h gap, then 10m
    t0 = pd.Timestamp("2026-01-01 00:00:00")
    times = [
        t0,
        t0 + pd.Timedelta(minutes=10),
        t0 + pd.Timedelta(minutes=20),
        t0 + pd.Timedelta(minutes=140),  # 2hr gap
        t0 + pd.Timedelta(minutes=150),
    ]
    df = pd.DataFrame({
        "utc_time": times,
        "x_error_m": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    _, meta = regularize_cadence(df, target_cadence_minutes=10.0, resample_if_irregular=False)
    assert meta.is_irregular is True
    assert meta.original_cadence_minutes == 10.0
    assert meta.max_gap_minutes == 120.0
    assert meta.cadence_warning is not None


def test_provided_state_provider_interpolation():
    """Test 4: ProvidedStateProvider interpolates orbital state correctly."""
    t0 = pd.Timestamp("2026-01-01 00:00:00")
    state_df = pd.DataFrame({
        "utc_time": [t0, t0 + pd.Timedelta(hours=2)],
        "position_x": [42000000.0, 42000000.0],
        "position_y": [0.0, 100000.0],
        "position_z": [0.0, 0.0],
        "velocity_x": [0.0, 0.0],
        "velocity_y": [3000.0, 3000.0],
        "velocity_z": [0.0, 0.0],
    })
    provider = ProvidedStateProvider(state_df)
    assert provider.source_name == "provided"

    # Query mid-point: 1 hour in
    query_ts = [t0 + pd.Timedelta(hours=1)]
    pos, vel = provider.get_state(query_ts)
    assert pos.shape == (1, 3)
    assert np.isclose(pos[0, 1], 50000.0)  # linear mid-point of y


def test_use_ric_and_srp_change_feature_manifest():
    """Test 5 & 6: use_ric and use_srp modify manifest and expand feature matrix."""
    times = pd.date_range("2026-01-01", periods=20, freq="15min")
    df = pd.DataFrame({
        "utc_time": times,
        "x_error_m": np.random.randn(20),
        "y_error_m": np.random.randn(20),
        "z_error_m": np.random.randn(20),
        "clock_error_m": np.random.randn(20),
    })

    # Baseline features
    m_base = FeatureManifest(use_ric=False, use_srp=False)
    x_base, names_base, _ = build_training_features(df, m_base)
    # 2 elapsed + 6 sin + 6 cos = 14
    assert x_base.shape[1] == 14

    # With SRP
    m_srp = FeatureManifest(use_ric=False, use_srp=True)
    x_srp, names_srp, _ = build_training_features(df, m_srp)
    assert x_srp.shape[1] == 14 + 3
    assert "sun_beta_angle" in names_srp

    # With RIC
    m_ric = FeatureManifest(use_ric=True, use_srp=False)
    x_ric, names_ric, _ = build_training_features(df, m_ric)
    assert x_ric.shape[1] == 14 + 3
    assert "ric_r" in names_ric

    # Both
    m_both = FeatureManifest(use_ric=True, use_srp=True)
    x_both, names_both, _ = build_training_features(df, m_both)
    assert x_both.shape[1] == 14 + 6


def test_feature_manifest_train_inference_parity():
    """Test 7: Strict parity between training and inference feature matrices."""
    times_train = pd.date_range("2026-01-01", periods=20, freq="15min")
    df_train = pd.DataFrame({
        "utc_time": times_train,
        "x_error_m": np.random.randn(20),
        "y_error_m": np.random.randn(20),
        "z_error_m": np.random.randn(20),
        "clock_error_m": np.random.randn(20),
    })

    manifest = FeatureManifest(use_ric=True, use_srp=True)
    x_train, names_train, origin = build_training_features(df_train, manifest)

    times_test = pd.date_range("2026-01-01 05:00:00", periods=5, freq="15min")
    x_infer, names_infer = build_inference_features(
        times_test,
        origin=origin,
        manifest=manifest,
        history_df=df_train,
    )

    assert x_train.shape[1] == x_infer.shape[1]
    assert names_train == names_infer


def test_official_hierarchy_precedence():
    """Test 8: Official hierarchy strictly prefers higher Shapiro-Wilk W_avg."""
    # Model A: W_avg = 0.95, but higher MAE
    # Model B: W_avg = 0.85, but lower MAE
    eval_a = {
        "eligible": True,
        "priority_1": {"W": {"average": 0.95}},
        "priority_2": {"mean": {"aggregate": 10.0}, "std": {"aggregate": 10.0}},
        "priority_3": {"total_outliers": 5, "aggregate_max_discrepancy": 1.0},
    }
    eval_b = {
        "eligible": True,
        "priority_1": {"W": {"average": 0.85}},
        "priority_2": {"mean": {"aggregate": 0.1}, "std": {"aggregate": 0.1}},
        "priority_3": {"total_outliers": 0, "aggregate_max_discrepancy": 0.1},
    }
    winner_idx, reason = compare_models_hierarchical(eval_a, eval_b)
    assert winner_idx == 1
    assert "Priority 1" in reason


def test_end_to_end_single_satellite_upload_and_training(synthetic_geo_csv, synthetic_geo_test_csv, tmp_path):
    """Test 9 & 10: Single-satellite upload, calibration, artifact persistence, and inference."""
    reg_path = tmp_path / "registry.json"
    artifacts_dir = tmp_path / "artifacts"
    reports_dir = tmp_path / "reports"

    registry = SatelliteModelRegistry(reg_path)
    pipeline = CalibrationPipeline(
        registry=registry,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        candidate_models=["persistence", "harmonic_ridge", "random_forest"],
    )

    # 1. Validate dataset
    ds = run_sat_val(synthetic_geo_csv, satellite_id="SAT_TEST_1", orbit_type="GEO")

    # 2. Train single satellite
    res = pipeline.train_single_satellite(
        dataset=ds,
        test_data=synthetic_geo_test_csv,
        use_ric=True,
        use_srp=True,
    )

    assert res["satellite_id"] == "SAT_TEST_1"
    assert "selected_model" in res
    assert res["winning_score"] > 0

    # 3. Check nested artifact layout
    sat_dir = get_satellite_artifact_dir("SAT_TEST_1", artifacts_dir)
    assert (sat_dir / "metadata.json").exists()
    assert (sat_dir / "feature_manifest.json").exists()
    assert (sat_dir / "evaluation.json").exists()

    # 4. Check registry was atomically updated
    sel = registry.get_selection("SAT_TEST_1")
    assert sel is not None
    assert sel.selected_model == res["selected_model"]
    assert sel.use_ric is True
    assert sel.use_srp is True

    # 5. Independent second satellite
    ds2 = run_sat_val(synthetic_geo_csv, satellite_id="SAT_TEST_2", orbit_type="MEO")
    res2 = pipeline.train_single_satellite(
        dataset=ds2,
        test_data=synthetic_geo_test_csv,
        use_ric=False,
        use_srp=False,
    )
    assert res2["satellite_id"] == "SAT_TEST_2"

    # Verify both satellites coexist independently in the registry
    all_sels = registry.get_all_selections()
    assert "SAT_TEST_1" in all_sels
    assert "SAT_TEST_2" in all_sels
    assert all_sels["SAT_TEST_1"].use_ric is True
    assert all_sels["SAT_TEST_2"].use_ric is False


def test_public_api_single_satellite_workflow(synthetic_geo_csv, synthetic_geo_test_csv, tmp_path, monkeypatch):
    """Test 11: Public API endpoints for single satellite validation, training, inspection, and inference."""
    import src.forecasting.api as api_mod
    from src.forecasting.registry import SatelliteModelRegistry
    from src.forecasting.pipeline import CalibrationPipeline
    from src.forecasting.router import PredictionRouter

    test_reg = SatelliteModelRegistry(tmp_path / "test_reg.json")
    test_artifacts = tmp_path / "artifacts"
    test_pipeline = CalibrationPipeline(registry=test_reg, artifacts_dir=test_artifacts)
    test_router = PredictionRouter(registry=test_reg, artifacts_dir=test_artifacts)

    monkeypatch.setattr(api_mod, "_default_registry", test_reg)
    monkeypatch.setattr(api_mod, "_default_pipeline", test_pipeline)
    monkeypatch.setattr(api_mod, "_default_router", test_router)

    # 1. Validation endpoint
    val_info = validate_satellite_dataset(synthetic_geo_csv, satellite_id="API_SAT", orbit_type="GEO")
    assert val_info["satellite_id"] == "API_SAT"
    assert val_info["orbit_type"] == "GEO"
    assert val_info["row_count"] == 50

    # 2. Train endpoint
    train_res = train_satellite(
        dataset=synthetic_geo_csv,
        test_dataset=synthetic_geo_test_csv,
        satellite_id="API_SAT",
        orbit_type="GEO",
        use_ric=True,
        candidate_models=["harmonic_ridge", "persistence"],
    )
    assert train_res["satellite_id"] == "API_SAT"
    assert "selected_model" in train_res

    # 3. Model inspection endpoint
    model_info = get_satellite_model("API_SAT")
    assert model_info is not None
    assert model_info["selected_model"] == train_res["selected_model"]
    assert model_info["use_ric"] is True

    # 4. Single-satellite prediction endpoint
    pred_df = predict_satellite("API_SAT", history_data=synthetic_geo_csv, horizon_steps=12, compute_ric=True)
    assert isinstance(pred_df, pd.DataFrame)
    assert len(pred_df) == 12
    assert "predicted_R" in pred_df.columns
    assert "predicted_I" in pred_df.columns
    assert "predicted_C" in pred_df.columns
    assert pred_df["satellite_id"].unique().tolist() == ["API_SAT"]


def test_physics_mode_none_workflow(synthetic_geo_csv, synthetic_geo_test_csv, tmp_path):
    """Verifies that physics_mode='none' disables all RIC and SRP physics context completely."""
    reg_path = tmp_path / "registry.json"
    artifacts_dir = tmp_path / "artifacts"
    registry = SatelliteModelRegistry(reg_path)
    pipeline = CalibrationPipeline(
        registry=registry,
        artifacts_dir=artifacts_dir,
        candidate_models=["harmonic_ridge", "persistence"],
    )

    ds = run_sat_val(synthetic_geo_csv, satellite_id="SAT_NONE", orbit_type="GEO")
    res = pipeline.train_single_satellite(
        dataset=ds,
        test_data=synthetic_geo_test_csv,
        physics_mode="none",
    )
    assert res["satellite_id"] == "SAT_NONE"
    assert res["physics_mode"] == "none"

    sel = registry.get_selection("SAT_NONE")
    assert sel.physics_mode == "none"
    assert sel.use_ric is False
    assert sel.use_srp is False

    from src.forecasting.router import PredictionRouter
    router = PredictionRouter(registry=registry, artifacts_dir=artifacts_dir)
    preds = router.predict_single_satellite(
        satellite_id="SAT_NONE",
        history_df=ds.dataframe,
        horizon_steps=6,
        compute_ric=False,
    )
    assert len(preds) == 6
    assert "predicted_X" in preds.columns
    assert "predicted_Clock" in preds.columns


def test_physics_mode_provided_workflow(synthetic_geo_csv, synthetic_geo_test_csv, tmp_path):
    """Verifies that physics_mode='provided' ingests state_df, persists orbital_state.csv, and uses it on reload."""
    reg_path = tmp_path / "registry.json"
    artifacts_dir = tmp_path / "artifacts"
    registry = SatelliteModelRegistry(reg_path)
    pipeline = CalibrationPipeline(
        registry=registry,
        artifacts_dir=artifacts_dir,
        candidate_models=["harmonic_ridge"],
    )

    times = pd.date_range("2026-01-01 00:00:00", periods=100, freq=pd.Timedelta(minutes=15))
    state_df = pd.DataFrame({
        "utc_time": times,
        "position_x": np.linspace(42164000.0, 42165000.0, 100),
        "position_y": np.linspace(0.0, 1000.0, 100),
        "position_z": np.linspace(0.0, 500.0, 100),
        "velocity_x": np.zeros(100),
        "velocity_y": np.full(100, 3075.0),
        "velocity_z": np.zeros(100),
    })

    ds = run_sat_val(synthetic_geo_csv, satellite_id="SAT_PROV", orbit_type="GEO")
    res = pipeline.train_single_satellite(
        dataset=ds,
        test_data=synthetic_geo_test_csv,
        physics_mode="provided",
        state_df=state_df,
    )
    assert res["satellite_id"] == "SAT_PROV"
    assert res["physics_mode"] == "provided"
    assert res["state_artifact"] is not None
    assert Path(res["state_artifact"]).exists()

    sel = registry.get_selection("SAT_PROV")
    assert sel.physics_mode == "provided"
    assert sel.state_artifact is not None
    assert Path(sel.state_artifact).exists()

    from src.forecasting.router import PredictionRouter
    router = PredictionRouter(registry=registry, artifacts_dir=artifacts_dir)
    preds = router.predict_single_satellite(
        satellite_id="SAT_PROV",
        history_df=ds.dataframe,
        horizon_steps=6,
        compute_ric=True,
    )
    assert len(preds) == 6
    assert "predicted_R" in preds.columns
    assert "predicted_I" in preds.columns
    assert "predicted_C" in preds.columns


def test_physics_mode_provided_without_state_raises():
    """Verifies that physics_mode='provided' without state_df raises a clear ValueError."""
    pipeline = CalibrationPipeline()
    times = pd.date_range("2026-01-01", periods=20, freq="15min")
    ds = SatelliteDataset(
        satellite_id="SAT_ERR",
        orbit_type="GEO",
        dataframe=pd.DataFrame({
            "utc_time": times,
            "x_error_m": np.ones(20),
            "y_error_m": np.ones(20),
            "z_error_m": np.ones(20),
            "clock_error_m": np.ones(20),
        }),
    )
    with pytest.raises(ValueError, match="state_df"):
        pipeline.train_single_satellite(
            dataset=ds,
            test_data=ds.dataframe.iloc[-5:],
            physics_mode="provided",
            state_df=None,
        )

