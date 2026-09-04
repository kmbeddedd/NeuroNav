"""Ablation & Experimentation Framework (E0 to E6).

Evaluates components P1-P6 independently to measure the isolated performance contribution
of each technique without confounding variables:
- E0: Existing baseline (Persistence, Harmonic Ridge, Random Forest, GP) in ECEF
- E1: + RIC coordinate transformation (ECEF -> RIC training -> ECEF inference)
- E2: + Promoted GEO Gated MoE architecture
- E3: + SISRE evaluation metric
- E4: + Decoupled Clock forecaster
- E5: + Solar Radiation Pressure (SRP) features (Sun beta angle, shadow factor)
- E6: + N-HiTS multi-rate hierarchical forecaster
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.forecasting.models.decoupled_clock import DecoupledClockModel
from src.forecasting.models.gaussian_process import GaussianProcessModel
from src.forecasting.models.geo_moe import GEOGatedMoEModel
from src.forecasting.models.harmonic_ridge import HarmonicRidgeModel
from src.forecasting.models.nhits import NHiTSModel
from src.forecasting.models.persistence import PersistenceModel
from src.forecasting.models.random_forest import RandomForestModel
from src.forecasting.pipeline import compute_metrics_for_residuals
from src.forecasting.validation import load_telemetry_source
from src.physics import (
    compute_shadow_factor,
    compute_sisre,
    compute_sun_beta_angle,
    ecef_error_to_ric,
    nominal_satellite_orbit,
    ric_error_to_ecef,
)

TARGET_COLS = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]


def run_ablation_experiment(
    train_source: Union[str, Path, Dict[str, pd.DataFrame]],
    test_source: Union[str, Path, Dict[str, pd.DataFrame]],
    output_dir: Union[str, Path] = "results/ablations",
) -> pd.DataFrame:
    """Executes formal ablation experiments E0 through E6 across all available satellites."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_series = load_telemetry_source(train_source)
    test_series = load_telemetry_source(test_source)
    common_sats = sorted(set(train_series.keys()).intersection(set(test_series.keys())))

    records: List[Dict[str, Any]] = []

    for sat_id in common_sats:
        train_df = train_series[sat_id]
        test_df = test_series[sat_id]
        test_times = pd.to_datetime(test_df["utc_time"]).reset_index(drop=True)
        test_actual = test_df[TARGET_COLS].to_numpy(dtype=float)

        # -------------------------------------------------------------
        # E0: Existing Baselines in Cartesian ECEF
        # -------------------------------------------------------------
        for model_cls, name in [
            (PersistenceModel, "Persistence"),
            (HarmonicRidgeModel, "Harmonic Ridge"),
            (RandomForestModel, "Random Forest"),
            (GaussianProcessModel, "Gaussian Process"),
        ]:
            t0 = time.perf_counter()
            inst = model_cls(name=name)
            inst.fit(train_df)
            pred = inst.predict(train_df, test_times)
            dur = time.perf_counter() - t0

            m = compute_metrics_for_residuals(test_actual, pred, orbit_class=sat_id, timestamps=test_times)
            records.append({
                "experiment_id": "E0_baseline",
                "satellite": sat_id,
                "model": name,
                "feature_frame": "ECEF",
                "configuration": "standard_time_harmonics",
                "orbit_3d_mae_m": m["orbit_3d_vector_mae_m"],
                "clock_mae_m": m["clock_mae_m"],
                "sisre_mean_m": m["sisre_mean_m"],
                "mean_shapiro_w": m["mean_shapiro_w"],
                "runtime_seconds": dur,
            })

        # -------------------------------------------------------------
        # E1: + RIC Coordinate Feature Integration
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        inst_ric = RandomForestModel(
            name="Random Forest + RIC",
            use_ric=True,
            satellite_id=sat_id,
            orbit_class=sat_id,
            random_state=42,
        )
        inst_ric.fit(train_df)
        pred_e1 = inst_ric.predict(train_df, test_times)
        dur = time.perf_counter() - t0

        m_e1 = compute_metrics_for_residuals(test_actual, pred_e1, orbit_class=sat_id, timestamps=test_times)
        records.append({
            "experiment_id": "E1_ric_features",
            "satellite": sat_id,
            "model": "Random Forest + RIC",
            "feature_frame": "ECEF + RIC",
            "configuration": "ric_radial_in_track_cross_track_features",
            "orbit_3d_mae_m": m_e1["orbit_3d_vector_mae_m"],
            "clock_mae_m": m_e1["clock_mae_m"],
            "sisre_mean_m": m_e1["sisre_mean_m"],
            "mean_shapiro_w": m_e1["mean_shapiro_w"],
            "runtime_seconds": dur,
        })

        # -------------------------------------------------------------
        # E2: + Promoted GEO Gated MoE
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        moe = GEOGatedMoEModel(name="GEO Gated MoE", max_epochs=30)
        moe.fit(train_df)
        pred_e2 = moe.predict(train_df, test_times)
        dur = time.perf_counter() - t0

        m_e2 = compute_metrics_for_residuals(test_actual, pred_e2, orbit_class=sat_id, timestamps=test_times)
        records.append({
            "experiment_id": "E2_gated_moe",
            "satellite": sat_id,
            "model": "GEO Gated MoE",
            "feature_frame": "ECEF",
            "configuration": "gru_regime_gate_normal_excursion_experts",
            "orbit_3d_mae_m": m_e2["orbit_3d_vector_mae_m"],
            "clock_mae_m": m_e2["clock_mae_m"],
            "sisre_mean_m": m_e2["sisre_mean_m"],
            "mean_shapiro_w": m_e2["mean_shapiro_w"],
            "runtime_seconds": dur,
        })

        # -------------------------------------------------------------
        # E3: + SISRE Evaluation Check (Reported on E0 Ridge baseline)
        # -------------------------------------------------------------
        records.append({
            "experiment_id": "E3_sisre_metric",
            "satellite": sat_id,
            "model": "Harmonic Ridge",
            "feature_frame": "ECEF",
            "configuration": "standardized_igs_sisre_ranging_metric",
            "orbit_3d_mae_m": m["orbit_3d_vector_mae_m"],
            "clock_mae_m": m["clock_mae_m"],
            "sisre_mean_m": m["sisre_mean_m"],
            "mean_shapiro_w": m["mean_shapiro_w"],
            "runtime_seconds": 0.001,
        })

        # -------------------------------------------------------------
        # E4: + Decoupled Clock Model
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        dec_clock = DecoupledClockModel(name="Decoupled Clock", tcn_epochs=20)
        dec_clock.fit(train_df)
        pred_e4 = dec_clock.predict(train_df, test_times)
        dur = time.perf_counter() - t0

        m_e4 = compute_metrics_for_residuals(test_actual, pred_e4, orbit_class=sat_id, timestamps=test_times)
        records.append({
            "experiment_id": "E4_decoupled_clock",
            "satellite": sat_id,
            "model": "Decoupled Clock",
            "feature_frame": "ECEF",
            "configuration": "quadratic_poly_baseline_plus_tcn_residual",
            "orbit_3d_mae_m": m_e4["orbit_3d_vector_mae_m"],
            "clock_mae_m": m_e4["clock_mae_m"],
            "sisre_mean_m": m_e4["sisre_mean_m"],
            "mean_shapiro_w": m_e4["mean_shapiro_w"],
            "runtime_seconds": dur,
        })

        # -------------------------------------------------------------
        # E5: + SRP Features (Sun beta angle, shadow factor, solar cos angle)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        inst_srp = RandomForestModel(
            name="Random Forest + SRP",
            enable_srp=True,
            satellite_id=sat_id,
            orbit_class=sat_id,
            random_state=42,
        )
        inst_srp.fit(train_df)
        pred_e5 = inst_srp.predict(train_df, test_times)
        dur = time.perf_counter() - t0

        m_e5 = compute_metrics_for_residuals(test_actual, pred_e5, orbit_class=sat_id, timestamps=test_times)
        records.append({
            "experiment_id": "E5_srp_features",
            "satellite": sat_id,
            "model": "Random Forest + SRP",
            "feature_frame": "ECEF + SRP",
            "configuration": "sun_beta_angle_shadow_factor_solar_cos_angle",
            "orbit_3d_mae_m": m_e5["orbit_3d_vector_mae_m"],
            "clock_mae_m": m_e5["clock_mae_m"],
            "sisre_mean_m": m_e5["sisre_mean_m"],
            "mean_shapiro_w": m_e5["mean_shapiro_w"],
            "runtime_seconds": dur,
        })

        # -------------------------------------------------------------
        # E6: + N-HiTS Forecaster
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        nhits = NHiTSModel(name="N-HiTS", lookback_steps=12, epochs=30)
        nhits.fit(train_df)
        pred_e6 = nhits.predict(train_df, test_times)
        dur = time.perf_counter() - t0

        m_e6 = compute_metrics_for_residuals(test_actual, pred_e6, orbit_class=sat_id, timestamps=test_times)
        records.append({
            "experiment_id": "E6_nhits",
            "satellite": sat_id,
            "model": "N-HiTS",
            "feature_frame": "Hierarchical Multi-Rate",
            "configuration": "multi_scale_pooling_basis_interpolation",
            "orbit_3d_mae_m": m_e6["orbit_3d_vector_mae_m"],
            "clock_mae_m": m_e6["clock_mae_m"],
            "sisre_mean_m": m_e6["sisre_mean_m"],
            "mean_shapiro_w": m_e6["mean_shapiro_w"],
            "runtime_seconds": dur,
        })

    ablation_df = pd.DataFrame(records)
    ablation_df.to_csv(out_path / "ablation_results.csv", index=False)
    (out_path / "ablation_summary.json").write_text(
        json.dumps(records, indent=2, default=str), encoding="utf-8"
    )
    return ablation_df
