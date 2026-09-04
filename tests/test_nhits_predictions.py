"""Regression test verifying N-HiTS Day-8 prediction CSV exports."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "evaluation" / "nhits_day8"

def test_nhits_prediction_files_exist_and_valid():
    files = {
        "GEO": (OUTPUT_DIR / "predictions_GEO_nhits.csv", 69),
        "MEO_train": (OUTPUT_DIR / "predictions_MEO_train_nhits.csv", 11),
        "MEO_Train2": (OUTPUT_DIR / "predictions_MEO_Train2_nhits.csv", 30),
    }
    
    expected_cols = [
        "satellite", "dataset", "utc_time",
        "predicted_x_error (m)", "predicted_y_error (m)", "predicted_z_error (m)", "predicted_satclockerror (m)",
        "actual_x_error (m)", "actual_y_error (m)", "actual_z_error (m)", "actual_satclockerror (m)",
        "residual_x_error (m)", "residual_y_error (m)", "residual_z_error (m)", "residual_satclockerror (m)",
        "orbit_3d_error (m)",
    ]

    for key, (path, expected_len) in files.items():
        assert path.exists(), f"Missing file: {path}"
        df = pd.read_csv(path)
        assert len(df) == expected_len, f"{key} row count mismatch: {len(df)} != {expected_len}"
        for col in expected_cols:
            assert col in df.columns, f"Missing column {col} in {key}"
        
        numeric_cols = [c for c in expected_cols if "(m)" in c]
        for col in numeric_cols:
            assert np.all(np.isfinite(df[col].to_numpy())), f"Non-finite values found in {key} column {col}"

def test_nhits_submission_format_files():
    sub_files = [
        (OUTPUT_DIR / "DATA_GEO_Test_nhits_predicted.csv", 69),
        (OUTPUT_DIR / "DATA_MEO_Test_nhits_predicted.csv", 11),
        (OUTPUT_DIR / "DATA_MEO_Train2_Test_nhits_predicted.csv", 30),
    ]
    cols = ["utc_time", "x_error (m)", "y_error (m)", "z_error (m)", "satclockerror (m)"]
    for path, expected_len in sub_files:
        assert path.exists(), f"Missing submission file: {path}"
        df = pd.read_csv(path)
        assert len(df) == expected_len, f"Submission row count mismatch in {path}"
        assert list(df.columns) == cols, f"Columns mismatch in {path}"
        assert np.all(np.isfinite(df[["x_error (m)", "y_error (m)", "z_error (m)", "satclockerror (m)"]].to_numpy()))

if __name__ == "__main__":
    test_nhits_prediction_files_exist_and_valid()
    test_nhits_submission_format_files()
    print("All N-HiTS prediction tests passed successfully!")
