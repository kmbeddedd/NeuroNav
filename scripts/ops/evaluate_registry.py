"""Reproducible audit of the currently registered satellite forecasters.

Uses only the repository's supplied train/test CSV files.  It does not create,
interpolate, or augment observations.  Outputs are evaluation reports, not data
used for training.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from src.forecasting.models import create_model
from src.forecasting.evaluation.official import evaluate_residuals_official_hierarchy
from src.forecasting.registry.store import SatelliteModelRegistry
from src.forecasting.inference.router import PredictionRouter
from src.forecasting.data.validation import TARGET_COLS_INTERNAL, load_telemetry_source, validate_dataset


OUT = ROOT / "reports" / "evaluation" / "registered_models_20260904"
TARGETS = TARGET_COLS_INTERNAL
SAT_FILES = {
    "GEO": ("DATA_GEO_Train.csv", "DATA_GEO_Test.csv"),
    "MEO-1": ("DATA_MEO_Train.csv", "DATA_MEO_Test.csv"),
    "MEO-2": ("DATA_MEO_Train2.csv", "DATA_MEO_Test2.csv"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def diagnostics(actual: np.ndarray, pred: np.ndarray) -> dict:
    residual = pred - actual
    orbit_ae = np.linalg.norm(residual[:, :3], axis=1)
    target = {}
    for i, col in enumerate(TARGETS):
        err = residual[:, i]
        target[col] = {
            "mae_m": float(np.mean(np.abs(err))),
            "rmse_m": float(np.sqrt(np.mean(err**2))),
            "bias_m": float(np.mean(err)),
            "max_abs_error_m": float(np.max(np.abs(err))),
        }
    return {
        "n": int(len(actual)),
        "targets": target,
        "orbit_3d_vector_mae_m": float(np.mean(orbit_ae)),
        "orbit_3d_vector_rmse_m": float(np.sqrt(np.mean(orbit_ae**2))),
        "clock_mae_m": target["clock_error_m"]["mae_m"],
    }


def bootstrap_ci(actual: np.ndarray, pred: np.ndarray, seed: int = 20260904) -> dict:
    rng = np.random.default_rng(seed)
    n = len(actual)
    orbit = np.linalg.norm(pred[:, :3] - actual[:, :3], axis=1)
    clock = np.abs(pred[:, 3] - actual[:, 3])
    idx = rng.integers(0, n, size=(5000, n))
    return {
        "method": "percentile bootstrap over unique epochs",
        "resamples": 5000,
        "orbit_3d_vector_mae_m_95ci": np.quantile(orbit[idx].mean(axis=1), [.025, .975]).tolist(),
        "clock_mae_m_95ci": np.quantile(clock[idx].mean(axis=1), [.025, .975]).tolist(),
    }


def paired_delta_ci(actual: np.ndarray, winner: np.ndarray, baseline: np.ndarray, seed: int = 20260904) -> dict:
    """Winner minus baseline; negative is better for error metrics."""
    rng = np.random.default_rng(seed)
    n = len(actual)
    d_orbit = np.linalg.norm(winner[:, :3] - actual[:, :3], axis=1) - np.linalg.norm(
        baseline[:, :3] - actual[:, :3], axis=1
    )
    d_clock = np.abs(winner[:, 3] - actual[:, 3]) - np.abs(baseline[:, 3] - actual[:, 3])
    idx = rng.integers(0, n, size=(5000, n))
    return {
        "winner_minus_persistence_orbit_mae_m": float(d_orbit.mean()),
        "orbit_delta_95ci": np.quantile(d_orbit[idx].mean(axis=1), [.025, .975]).tolist(),
        "winner_minus_persistence_clock_mae_m": float(d_clock.mean()),
        "clock_delta_95ci": np.quantile(d_clock[idx].mean(axis=1), [.025, .975]).tolist(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = SatelliteModelRegistry(ROOT / "models" / "registry" / "satellite_model_registry.json")
    router = PredictionRouter(registry=registry, artifacts_dir=ROOT / "models" / "registry" / "artifacts")
    train_val = validate_dataset(ROOT / "data" / "ps08", min_history_rows=8)
    test_val = validate_dataset(ROOT / "data" / "ps08", min_history_rows=1, is_test_dataset=True)
    assert train_val.is_valid and test_val.is_valid

    audit = {
        "evaluation_role": "Day-8 model-selection/calibration set; not an untouched final holdout",
        "bootstrap_seed": 20260904,
        "versions": {
            "git_head": git("rev-parse", "HEAD"),
            "git_status_before_report": git("status", "--short"),
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "torch": torch.__version__,
        },
        "data_files": {}, "satellites": {},
    }
    prediction_rows, worst_rows, slice_rows = [], [], []

    for sat, (train_name, test_name) in SAT_FILES.items():
        train_path = ROOT / "data" / "ps08" / train_name
        test_path = ROOT / "data" / "ps08" / test_name
        raw_train, raw_test = pd.read_csv(train_path), pd.read_csv(test_path)
        train = train_val.normalized_data[sat]
        test = test_val.normalized_data[sat]
        times = pd.DatetimeIndex(test["utc_time"])
        actual = test[TARGETS].to_numpy(float)
        model, selection = router.get_assigned_model(sat)
        pred1 = model.predict(train, times)
        pred2 = model.predict(train, times)
        repeat_max = float(np.max(np.abs(pred1 - pred2)))

        persistence = create_model("persistence").fit(train)
        base_pred = persistence.predict(train, times)
        official = evaluate_residuals_official_hierarchy(actual, pred1, orbit_class=sat, timestamps=times)
        base_official = evaluate_residuals_official_hierarchy(actual, base_pred, orbit_class=sat, timestamps=times)

        # Serving path does not deduplicate input. Quantify raw-vs-calibration history sensitivity.
        raw_loaded = load_telemetry_source(train_path)
        raw_hist = next(iter(raw_loaded.values()))
        raw_pred = model.predict(raw_hist, times)
        history_skew = np.linalg.norm(raw_pred[:, :3] - pred1[:, :3], axis=1)

        # Prediction sensitivity to a small timestamp perturbation (no observations invented).
        plus_one = model.predict(train, times + pd.Timedelta(minutes=1))
        time_sensitivity = np.linalg.norm(plus_one[:, :3] - pred1[:, :3], axis=1)

        sat_result = {
            "selected_model": selection.selected_model,
            "model_version": selection.model_version,
            "artifact": selection.model_artifact,
            "artifact_sha256": sha256(ROOT / Path(selection.model_artifact)),
            "train_rows_raw": len(raw_train), "train_epochs_unique": len(train),
            "test_rows_raw": len(raw_test), "test_epochs_unique": len(test),
            "train_duplicate_epochs_removed": train_val.satellite_reports[sat].duplicate_epochs,
            "test_duplicate_epochs_removed": test_val.satellite_reports[sat].duplicate_epochs,
            "train_test_timestamp_overlap": int(len(set(train.utc_time) & set(test.utc_time))),
            "train_test_full_row_overlap": int(len(pd.merge(train, test, on=["utc_time", *TARGETS]))),
            "winner": diagnostics(actual, pred1),
            "winner_official": {
                "W_avg": official["priority_1"]["W"]["average"],
                "W": official["priority_1"]["W"],
                "p_value": official["priority_1"]["p_value"],
                "rejected_tests": official["priority_1"]["total_rejected_tests"],
                "sisre_mean_m": official["supplementary"]["sisre_mean_m"],
            },
            "confidence_intervals": bootstrap_ci(actual, pred1),
            "persistence": diagnostics(actual, base_pred),
            "persistence_W_avg": base_official["priority_1"]["W"]["average"],
            "paired_vs_persistence": paired_delta_ci(actual, pred1, base_pred),
            "repeat_prediction_max_abs_delta_m": repeat_max,
            "raw_vs_dedup_history": {
                "max_abs_component_delta_m": float(np.max(np.abs(raw_pred - pred1))),
                "mean_orbit_prediction_delta_m": float(history_skew.mean()),
                "max_orbit_prediction_delta_m": float(history_skew.max()),
            },
            "plus_1_min_timestamp_sensitivity": {
                "mean_orbit_prediction_delta_m": float(time_sensitivity.mean()),
                "max_orbit_prediction_delta_m": float(time_sensitivity.max()),
                "max_clock_prediction_delta_m": float(np.max(np.abs(plus_one[:, 3] - pred1[:, 3]))),
            },
        }

        # The pipeline exposes no global seed. Two clean retrains quantify the
        # resulting run-to-run variance without touching registered artifacts.
        retrains = []
        for run in range(2):
            retrained = create_model(selection.selected_model).fit(train)
            retrain_pred = retrained.predict(train, times)
            retrain_official = evaluate_residuals_official_hierarchy(
                actual, retrain_pred, orbit_class=sat, timestamps=times
            )
            retrains.append({
                "run": run + 1,
                "W_avg": retrain_official["priority_1"]["W"]["average"],
                "orbit_3d_vector_mae_m": retrain_official["supplementary"]["orbit_3d_vector_mae_m"],
                "clock_mae_m": retrain_official["supplementary"]["clock_mae_m"],
            })
        sat_result["clean_retrain_repeats_without_pipeline_seed"] = retrains
        sat_result["clean_retrain_range"] = {
            "W_avg": float(np.ptp([x["W_avg"] for x in retrains])),
            "orbit_3d_vector_mae_m": float(np.ptp([x["orbit_3d_vector_mae_m"] for x in retrains])),
            "clock_mae_m": float(np.ptp([x["clock_mae_m"] for x in retrains])),
        }

        n = len(test)
        split = max(1, n // 2)
        for label, slc in (("early_half", slice(0, split)), ("late_half", slice(split, n))):
            d = diagnostics(actual[slc], pred1[slc])
            slice_rows.append({"satellite": sat, "slice": label, **{k: v for k, v in d.items() if k != "targets"}})
        sat_result["time_slices"] = {r["slice"]: r for r in slice_rows if r["satellite"] == sat}

        err = pred1 - actual
        orbit_loss = np.linalg.norm(err[:, :3], axis=1)
        order = np.argsort(orbit_loss)[::-1]
        for rank, i in enumerate(order[: min(5, n)], 1):
            row = {"satellite": sat, "rank": rank, "utc_time": times[i].isoformat(),
                   "orbit_3d_error_m": orbit_loss[i], "clock_abs_error_m": abs(err[i, 3])}
            for j, col in enumerate(TARGETS):
                row[f"actual_{col}"] = actual[i, j]
                row[f"predicted_{col}"] = pred1[i, j]
                row[f"residual_{col}"] = err[i, j]
            worst_rows.append(row)
        for i in range(n):
            prediction_rows.append({"satellite": sat, "utc_time": times[i].isoformat(),
                                    **{f"actual_{c}": actual[i, j] for j, c in enumerate(TARGETS)},
                                    **{f"predicted_{c}": pred1[i, j] for j, c in enumerate(TARGETS)}})
        audit["data_files"][train_name] = sha256(train_path)
        audit["data_files"][test_name] = sha256(test_path)
        audit["satellites"][sat] = sat_result

    pd.DataFrame(prediction_rows).to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(worst_rows).to_csv(OUT / "worst_predictions.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(OUT / "time_slice_metrics.csv", index=False)
    (OUT / "evaluation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({sat: audit["satellites"][sat] for sat in SAT_FILES}, indent=2))


if __name__ == "__main__":
    main()
