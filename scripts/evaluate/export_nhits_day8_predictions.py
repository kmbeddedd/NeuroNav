"""Generates predicted 8th day values for GEO, MEO train, and MEO Train2 using N-HiTS.

Outputs:
1. Dedicated per-dataset CSV files with full error diagnostics and ground truth comparison:
   - predictions_GEO_nhits.csv
   - predictions_MEO_train_nhits.csv
   - predictions_MEO_Train2_nhits.csv
2. Submission-ready CSV files matching the raw test dataset schema:
   - DATA_GEO_Test_nhits_predicted.csv
   - DATA_MEO_Test_nhits_predicted.csv
   - DATA_MEO_Test2_nhits_predicted.csv
3. Combined multi-satellite evaluation CSV:
   - all_satellites_nhits_day8_predictions.csv
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = PROJECT_ROOT / "reports" / "evaluation" / "nhits_day8"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch

from src.forecasting.models.nhits import NHiTSModel
from src.forecasting.evaluation.official import evaluate_residuals_official_hierarchy
from src.forecasting.data.validation import validate_dataset

# Set seed for deterministic reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Load validated datasets
train_val = validate_dataset(PROJECT_ROOT / "data" / "ps08", min_history_rows=8)
test_val = validate_dataset(PROJECT_ROOT / "data" / "ps08", min_history_rows=1, is_test_dataset=True)

# Specifications
SERIES_MAP = {
    "GEO": {
        "dataset_name": "GEO",
        "train_file": "data/ps08/DATA_GEO_Train.csv",
        "test_file": "data/ps08/DATA_GEO_Test.csv",
        "model_artifact": "models/registry/artifacts/GEO_nhits.pt",
        "csv_stem": "GEO",
    },
    "MEO-1": {
        "dataset_name": "MEO train",
        "train_file": "data/ps08/DATA_MEO_Train.csv",
        "test_file": "data/ps08/DATA_MEO_Test.csv",
        "model_artifact": "models/registry/artifacts/MEO-1_nhits.pt",
        "csv_stem": "MEO_train",
    },
    "MEO-2": {
        "dataset_name": "MEO Train2",
        "train_file": "data/ps08/DATA_MEO_Train2.csv",
        "test_file": "data/ps08/DATA_MEO_Test2.csv",
        "model_artifact": "models/registry/artifacts/MEO-2_nhits.pt",
        "csv_stem": "MEO_Train2",
    },
}

models: dict[str, NHiTSModel] = {}
eval_results: dict[str, dict] = {}
combined_rows = []

for sat_id, cfg in SERIES_MAP.items():
    train_df = train_val.normalized_data[sat_id]
    test_df = test_val.normalized_data[sat_id]
    
    art_path = PROJECT_ROOT / cfg["model_artifact"]
    if art_path.exists():
        model = NHiTSModel.load(art_path)
        print(f"Loaded existing artifact: {art_path}")
    else:
        model = NHiTSModel().fit(train_df)
        model.save(art_path)
        print(f"Fitted and saved artifact: {art_path}")
    models[sat_id] = model

    # Chronologically sorted, unique timestamps
    unique_times = pd.DatetimeIndex(test_df["utc_time"])
    actual_unique = test_df[["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]].to_numpy()
    preds_unique = model.predict(train_df, unique_times)

    # Official evaluation metrics
    metrics = evaluate_residuals_official_hierarchy(actual_unique, preds_unique, orbit_class=sat_id, timestamps=unique_times)
    eval_results[sat_id] = metrics
    print(f"[{cfg['dataset_name']} ({sat_id})] Shapiro-Wilk W_avg: {metrics['priority_1']['W']['average']:.6f}, 3D MAE: {metrics['supplementary']['orbit_3d_vector_mae_m']:.4f} m, Clock MAE: {metrics['supplementary']['clock_mae_m']:.4f} m")

    # Create mapping from normalized utc_time string to predicted values
    time_to_pred = {
        str(t): preds_unique[i]
        for i, t in enumerate(test_df["utc_time"])
    }

    # Load raw test CSV
    raw_test_path = PROJECT_ROOT / cfg["test_file"]
    raw_test_df = pd.read_csv(raw_test_path)
    
    # Parse raw utc_time to match normalized format
    raw_times_parsed = pd.to_datetime(raw_test_df["utc_time"])
    
    # Map predictions to raw test rows
    preds_raw = []
    for t in raw_times_parsed:
        t_str = str(t)
        if t_str in time_to_pred:
            preds_raw.append(time_to_pred[t_str])
        else:
            # Predict directly for this timestamp if not in dict
            single_pred = model.predict(train_df, pd.DatetimeIndex([t]))[0]
            preds_raw.append(single_pred)
    preds_raw = np.array(preds_raw)

    # Target column names in raw test
    raw_col_map = {
        "x_error (m)": "actual_x_error_m",
        "y_error (m)": "actual_y_error_m",
        "z_error (m)": "actual_z_error_m",
        "satclockerror (m)": "actual_clock_error_m",
    }
    # Handle possible spaces in raw column names
    norm_cols = {c: " ".join(c.strip().lower().split()) for c in raw_test_df.columns}
    raw_test_df_renamed = raw_test_df.rename(columns=norm_cols)

    actual_raw_x = raw_test_df_renamed["x_error (m)"].to_numpy(dtype=float)
    actual_raw_y = raw_test_df_renamed["y_error (m)"].to_numpy(dtype=float)
    actual_raw_z = raw_test_df_renamed["z_error (m)"].to_numpy(dtype=float)
    actual_raw_c = raw_test_df_renamed["satclockerror (m)"].to_numpy(dtype=float)

    pred_raw_x = preds_raw[:, 0]
    pred_raw_y = preds_raw[:, 1]
    pred_raw_z = preds_raw[:, 2]
    pred_raw_c = preds_raw[:, 3]

    res_raw_x = pred_raw_x - actual_raw_x
    res_raw_y = pred_raw_y - actual_raw_y
    res_raw_z = pred_raw_z - actual_raw_z
    res_raw_c = pred_raw_c - actual_raw_c
    orbit_3d_err = np.sqrt(res_raw_x**2 + res_raw_y**2 + res_raw_z**2)

    # 1. Comprehensive predictions DataFrame (aligned with raw test file row-by-row)
    detailed_df = pd.DataFrame({
        "satellite": sat_id,
        "dataset": cfg["dataset_name"],
        "utc_time": raw_test_df["utc_time"],
        "predicted_x_error (m)": pred_raw_x,
        "predicted_y_error (m)": pred_raw_y,
        "predicted_z_error (m)": pred_raw_z,
        "predicted_satclockerror (m)": pred_raw_c,
        "actual_x_error (m)": actual_raw_x,
        "actual_y_error (m)": actual_raw_y,
        "actual_z_error (m)": actual_raw_z,
        "actual_satclockerror (m)": actual_raw_c,
        "residual_x_error (m)": res_raw_x,
        "residual_y_error (m)": res_raw_y,
        "residual_z_error (m)": res_raw_z,
        "residual_satclockerror (m)": res_raw_c,
        "orbit_3d_error (m)": orbit_3d_err,
    })

    # Save generated evaluation output in the reports tree.
    detailed_csv_results = RESULTS_DIR / f"predictions_{cfg['csv_stem']}_nhits.csv"
    detailed_df.to_csv(detailed_csv_results, index=False)
    print(f"Saved: {detailed_csv_results} ({len(detailed_df)} rows)")

    # 2. Submission-format DataFrame (exact 5 columns with predicted values)
    submission_df = pd.DataFrame({
        "utc_time": raw_test_df["utc_time"],
        "x_error (m)": pred_raw_x,
        "y_error (m)": pred_raw_y,
        "z_error (m)": pred_raw_z,
        "satclockerror (m)": pred_raw_c,
    })
    sub_csv_results = RESULTS_DIR / f"DATA_{cfg['csv_stem']}_Test_nhits_predicted.csv"
    submission_df.to_csv(sub_csv_results, index=False)
    print(f"Saved submission format: {sub_csv_results}")

    # 3. Deduplicated unique-epoch version
    dedup_df = pd.DataFrame({
        "satellite": sat_id,
        "dataset": cfg["dataset_name"],
        "utc_time": test_df["utc_time"],
        "predicted_x_error (m)": preds_unique[:, 0],
        "predicted_y_error (m)": preds_unique[:, 1],
        "predicted_z_error (m)": preds_unique[:, 2],
        "predicted_satclockerror (m)": preds_unique[:, 3],
        "actual_x_error (m)": actual_unique[:, 0],
        "actual_y_error (m)": actual_unique[:, 1],
        "actual_z_error (m)": actual_unique[:, 2],
        "actual_satclockerror (m)": actual_unique[:, 3],
        "residual_x_error (m)": preds_unique[:, 0] - actual_unique[:, 0],
        "residual_y_error (m)": preds_unique[:, 1] - actual_unique[:, 1],
        "residual_z_error (m)": preds_unique[:, 2] - actual_unique[:, 2],
        "residual_satclockerror (m)": preds_unique[:, 3] - actual_unique[:, 3],
        "orbit_3d_error (m)": np.sqrt(
            (preds_unique[:, 0] - actual_unique[:, 0])**2 +
            (preds_unique[:, 1] - actual_unique[:, 1])**2 +
            (preds_unique[:, 2] - actual_unique[:, 2])**2
        ),
    })
    dedup_csv_results = RESULTS_DIR / f"predictions_{cfg['csv_stem']}_nhits_unique_epochs.csv"
    dedup_df.to_csv(dedup_csv_results, index=False)

    combined_rows.append(detailed_df)

# Combined CSV
combined_df = pd.concat(combined_rows, ignore_index=True)
combined_csv_results = RESULTS_DIR / "all_satellites_nhits_day8_predictions.csv"
combined_df.to_csv(combined_csv_results, index=False)
print(f"Saved combined CSV: {combined_csv_results} ({len(combined_df)} total rows)")

# Save evaluation summary JSON
summary_path = RESULTS_DIR / "nhits_day8_evaluation_summary.json"
summary_payload = {
    sat: {
        "dataset": SERIES_MAP[sat]["dataset_name"],
        "shapiro_w_avg": eval_results[sat]["priority_1"]["W"]["average"],
        "shapiro_w": eval_results[sat]["priority_1"]["W"],
        "shapiro_p": eval_results[sat]["priority_1"]["p_value"],
        "residual_mean": eval_results[sat]["priority_2"]["mean"],
        "residual_std": eval_results[sat]["priority_2"]["std"],
        "orbit_3d_vector_mae_m": eval_results[sat]["supplementary"]["orbit_3d_vector_mae_m"],
        "orbit_3d_vector_rmse_m": eval_results[sat]["supplementary"]["orbit_3d_vector_rmse_m"],
        "clock_mae_m": eval_results[sat]["supplementary"]["clock_mae_m"],
        "sisre_mean_m": eval_results[sat]["supplementary"]["sisre_mean_m"],
    }
    for sat in SERIES_MAP
}
import json
summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
print(f"Saved evaluation summary: {summary_path}")
