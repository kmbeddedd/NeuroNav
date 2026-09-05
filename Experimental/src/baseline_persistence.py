import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

DATASETS = {
    "GEO": "DATA_GEO_Train.csv",
    "MEO": "DATA_MEO_Train.csv",
    "MEO2": "DATA_MEO_Train2.csv",
}

ERROR_COLS = [
    "x_error (m)",
    "y_error (m)",
    "z_error (m)",
    "satclockerror (m)",
]

MIN_VALIDATION_OBS = 5


def load_data(path):

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["utc_time"] = pd.to_datetime(df["utc_time"])

    for col in ERROR_COLS:
        df[col] = pd.to_numeric(df[col])

    return (
        df.drop_duplicates()
        .sort_values("utc_time")
        .reset_index(drop=True)
    )


def predict_latest(history, targets):

    history_times = (
        history["utc_time"]
        .astype("int64")
        .to_numpy()
    )

    target_times = (
        targets["utc_time"]
        .astype("int64")
        .to_numpy()
    )

    predictions = []

    for target_time in target_times:

        idx = np.searchsorted(
            history_times,
            target_time,
            side="left"
        ) - 1

        if idx < 0:
            predictions.append(
                [np.nan] * len(ERROR_COLS)
            )
        else:
            predictions.append(
                history.iloc[idx][ERROR_COLS]
                .astype(float)
                .to_numpy()
            )

    return np.asarray(
        predictions,
        dtype=float
    )


def main():

    print("\nPERSISTENCE BASELINE")
    print("=" * 80)

    summary_results = []
    residual_results = []

    for name, filename in DATASETS.items():

        df = load_data(DATA_DIR / filename)

        df["date"] = df["utc_time"].dt.normalize()

        days = sorted(df["date"].unique())

        print(f"\n{name}")
        print("-" * 80)

        for validation_day in days[1:]:

            train = df[
                df["date"] < validation_day
            ]

            validation = df[
                df["date"] == validation_day
            ]

            if len(validation) < MIN_VALIDATION_OBS:
                continue

            predictions = predict_latest(
                train,
                validation
            )

            actual = validation[
                ERROR_COLS
            ].astype(float).to_numpy()

            valid_mask = (
                ~np.isnan(predictions).any(axis=1)
            )

            predictions = predictions[valid_mask]
            actual = actual[valid_mask]

            residuals = actual - predictions

            print(
                f"\nValidation day: "
                f"{validation_day.date()}"
            )

            print(
                f"History observations: "
                f"{len(train)}"
            )

            print(
                f"Validation observations: "
                f"{len(actual)}"
            )

            for i, col in enumerate(ERROR_COLS):

                r = residuals[:, i]

                mae = np.mean(np.abs(r))
                rmse = np.sqrt(np.mean(r ** 2))
                mean = np.mean(r)
                std = np.std(r)

                print(
                    f"  {col:<20} "
                    f"MAE={mae:.6f} "
                    f"RMSE={rmse:.6f} "
                    f"mean={mean: .6f} "
                    f"std={std: .6f}"
                )

                summary_results.append({
                    "dataset": name,
                    "validation_day": validation_day.date(),
                    "variable": col,
                    "n": len(r),
                    "mae": mae,
                    "rmse": rmse,
                    "residual_mean": mean,
                    "residual_std": std,
                })

                for j, value in enumerate(r):

                    residual_results.append({
                        "dataset": name,
                        "validation_day":
                            validation_day.date(),
                        "variable": col,
                        "sample_index": j,
                        "residual": value,
                    })

    output_dir = Path("results/baseline")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(summary_results).to_csv(
        output_dir /
        "persistence_results.csv",
        index=False
    )

    pd.DataFrame(residual_results).to_csv(
        output_dir /
        "persistence_residuals.csv",
        index=False
    )

    print("\nSaved:")
    print(
        "results/baseline/persistence_results.csv"
    )
    print(
        "results/baseline/persistence_residuals.csv"
    )


if __name__ == "__main__":
    main()
