import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import SATELLITES


OUTPUT_FILE = Path("visualization/generated/prediction_comparison.json")


def normalize_columns(df):
    """Normalize column names so minor whitespace differences do not matter."""
    df = df.copy()

    df.columns = [column.strip() for column in df.columns]

    rename_map = {
        "x_error (m)": "Ex",
        "y_error (m)": "Ey",
        "z_error (m)": "Ez",
        "satclockerror (m)": "Eclk",
    }

    df.rename(columns=rename_map, inplace=True)

    required = {"utc_time", "Ex", "Ey", "Ez", "Eclk"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df


def compare_satellite(satellite_id, config):
    print(f"\n{'=' * 60}")
    print(f"Processing {satellite_id}")
    print(f"{'=' * 60}")

    test_file = Path(config["test_file"])
    prediction_file = Path(config["prediction_file"])

    if not test_file.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_file}"
        )

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_file}"
        )

    actual = normalize_columns(
        pd.read_csv(test_file)
    )

    predicted = normalize_columns(
        pd.read_csv(prediction_file)
    )

    # Remove completely identical observations.
    # Some MEO datasets contain duplicated rows with identical values.
    actual = actual.drop_duplicates()
    predicted = predicted.drop_duplicates()

    print(f"Unique actual observations:    {len(actual)}")
    print(f"Unique predicted observations: {len(predicted)}")

    print(f"Actual observations:    {len(actual)}")
    print(f"Predicted observations: {len(predicted)}")

    merged = actual.merge(
        predicted,
        on="utc_time",
        how="inner",
        suffixes=("_actual", "_predicted"),
    )

    print(f"Matched observations:   {len(merged)}")

    if len(merged) == 0:
        raise ValueError(
            f"No matching timestamps found for {satellite_id}"
        )

    records = []

    for _, row in merged.iterrows():

        actual_values = {
            "Ex": float(row["Ex_actual"]),
            "Ey": float(row["Ey_actual"]),
            "Ez": float(row["Ez_actual"]),
            "Eclk": float(row["Eclk_actual"]),
        }

        predicted_values = {
            "Ex": float(row["Ex_predicted"]),
            "Ey": float(row["Ey_predicted"]),
            "Ez": float(row["Ez_predicted"]),
            "Eclk": float(row["Eclk_predicted"]),
        }

        residual_x = (
            predicted_values["Ex"] - actual_values["Ex"]
        )

        residual_y = (
            predicted_values["Ey"] - actual_values["Ey"]
        )

        residual_z = (
            predicted_values["Ez"] - actual_values["Ez"]
        )

        residual_3d = float(
            np.sqrt(
                residual_x ** 2
                + residual_y ** 2
                + residual_z ** 2
            )
        )

        residual_clock = (
            predicted_values["Eclk"]
            - actual_values["Eclk"]
        )

        records.append({
            "utc_time": row["utc_time"],

            "actual": actual_values,

            "predicted": predicted_values,

            "residual": {
                "Ex": residual_x,
                "Ey": residual_y,
                "Ez": residual_z,
                "position_3d": residual_3d,
                "Eclk": residual_clock,
            },
        })

    return {
        "satellite": config["name"],
        "orbit": config["orbit"],
        "observation_count": len(records),
        "observations": records,
    }


def main():
    output = {}

    for satellite_id, config in SATELLITES.items():
        output[satellite_id] = compare_satellite(
            satellite_id,
            config,
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_FILE, "w") as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(f"\n{'=' * 60}")
    print("Comparison pipeline completed")
    print(f"{'=' * 60}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
