import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

FILES = [
    "DATA_GEO_Train.csv",
    "DATA_GEO_Test.csv",
    "DATA_MEO_Train.csv",
    "DATA_MEO_Test.csv",
    "DATA_MEO_Train2.csv",
    "DATA_MEO_Test2.csv",
]

ERROR_COLS = [
    "x_error (m)",
    "y_error (m)",
    "z_error (m)",
    "satclockerror (m)",
]

TARGET_LAGS = [10, 15, 30, 60, 120]

TOLERANCE = {
    10: 2,
    15: 2,
    30: 5,
    60: 10,
    120: 15,
}


def load_data(path):
    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["utc_time"] = pd.to_datetime(
        df["utc_time"],
        errors="coerce"
    )

    for col in ERROR_COLS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return (
        df.dropna(subset=["utc_time"] + ERROR_COLS)
        .drop_duplicates()
        .sort_values("utc_time")
        .reset_index(drop=True)
    )


def find_pairs(df, target_lag):

    tolerance = TOLERANCE[target_lag]

    times = df["utc_time"].to_numpy()

    pairs = []

    for i in range(1, len(df)):

        previous_times = times[:i]

        actual_gaps = (
            times[i] - previous_times
        ) / np.timedelta64(1, "m")

        valid = (
            np.abs(actual_gaps - target_lag)
            <= tolerance
        )

        if not np.any(valid):
            continue

        candidates = np.where(valid)[0]

        best = candidates[
            np.argmin(
                np.abs(
                    actual_gaps[candidates]
                    - target_lag
                )
            )
        ]

        pairs.append((best, i))

    return pairs


def correlation_matrix(df, pairs):

    matrix = pd.DataFrame(
        index=ERROR_COLS,
        columns=ERROR_COLS,
        dtype=float
    )

    for past_col in ERROR_COLS:

        previous = np.array([
            df.iloc[j][past_col]
            for j, i in pairs
        ])

        for future_col in ERROR_COLS:

            current = np.array([
                df.iloc[i][future_col]
                for j, i in pairs
            ])

            if len(previous) < 5:
                corr = np.nan

            elif (
                np.std(previous) == 0
                or np.std(current) == 0
            ):
                corr = np.nan

            else:
                corr = np.corrcoef(
                    previous,
                    current
                )[0, 1]

            matrix.loc[past_col, future_col] = corr

    return matrix


def main():

    print("\nTIME-AWARE CROSS-VARIABLE CORRELATION")
    print("=" * 80)

    output = []

    for filename in FILES:

        df = load_data(DATA_DIR / filename)

        print(f"\n{filename}")
        print("-" * 80)

        for lag in TARGET_LAGS:

            pairs = find_pairs(df, lag)

            print(
                f"\nLag ≈ {lag} min | "
                f"pairs = {len(pairs)}"
            )

            matrix = correlation_matrix(
                df,
                pairs
            )

            print(
                matrix.round(3).to_string()
            )

            for past_col in ERROR_COLS:
                for future_col in ERROR_COLS:

                    output.append({
                        "dataset": filename,
                        "lag_min": lag,
                        "pairs": len(pairs),
                        "past_variable": past_col,
                        "future_variable": future_col,
                        "correlation":
                            matrix.loc[
                                past_col,
                                future_col
                            ],
                    })

    output_dir = Path("results/eda")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(output).to_csv(
        output_dir /
        "08_time_aware_cross_correlation.csv",
        index=False
    )

    print("\nSaved:")
    print(
        "results/eda/"
        "08_time_aware_cross_correlation.csv"
    )


if __name__ == "__main__":
    main()
