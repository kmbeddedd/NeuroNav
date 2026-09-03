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

TARGET_LAGS = [5, 10, 15, 30, 60, 120, 240]

TOLERANCE = {
    5: 2,
    10: 2,
    15: 2,
    30: 5,
    60: 10,
    120: 15,
    240: 30,
}


def load_data(path):
    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["utc_time"] = pd.to_datetime(df["utc_time"], errors="coerce")

    for col in ERROR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["utc_time"] + ERROR_COLS)
        .drop_duplicates()
        .sort_values("utc_time")
        .reset_index(drop=True)
    )

    return df


def calculate_lag_pairs(df, target_lag):
    """
    Match each observation with the closest previous observation
    to the requested elapsed-time lag.
    """

    tolerance = TOLERANCE[target_lag]

    times = df["utc_time"].to_numpy()

    pairs = []

    for i in range(1, len(df)):
        current_time = times[i]

        # Look at ALL previous observations.
        previous_times = times[:i]

        time_diff_minutes = (
            current_time - previous_times
        ) / np.timedelta64(1, "m")

        valid = np.abs(time_diff_minutes - target_lag) <= tolerance

        if not np.any(valid):
            continue

        valid_indices = np.where(valid)[0]

        best_local_index = valid_indices[
            np.argmin(
                np.abs(
                    time_diff_minutes[valid_indices] - target_lag
                )
            )
        ]

        actual_lag = time_diff_minutes[best_local_index]

        pairs.append(
            (
                best_local_index,
                i,
                actual_lag,
            )
        )

    return pairs


def main():

    print("\nTIME-AWARE LAG ANALYSIS")
    print("=" * 80)

    all_results = []

    for filename in FILES:

        path = DATA_DIR / filename
        df = load_data(path)

        print(f"\n{filename}")
        print("-" * 80)
        print(f"Observations after exact deduplication: {len(df)}")

        for target_lag in TARGET_LAGS:

            pairs = calculate_lag_pairs(
                df,
                target_lag
            )

            pair_count = len(pairs)

            row = {
                "dataset": filename,
                "target_lag_min": target_lag,
                "pairs": pair_count,
            }

            for col in ERROR_COLS:

                if pair_count < 5:
                    corr = np.nan

                else:

                    previous_values = np.array([
                        df.iloc[j][col]
                        for j, i, actual_lag in pairs
                    ])

                    current_values = np.array([
                        df.iloc[i][col]
                        for j, i, actual_lag in pairs
                    ])

                    if (
                        np.std(previous_values) == 0
                        or np.std(current_values) == 0
                    ):
                        corr = np.nan
                    else:
                        corr = np.corrcoef(
                            previous_values,
                            current_values
                        )[0, 1]

                row[col] = corr

            all_results.append(row)

        result_df = pd.DataFrame(
            [r for r in all_results
             if r["dataset"] == filename]
        )

        correlation_table = result_df.set_index(
            "target_lag_min"
        )[ERROR_COLS]

        print("\nCorrelation:")
        print(correlation_table.round(3).to_string())

        print("\nNumber of matched pairs:")
        print(
            result_df[
                ["target_lag_min", "pairs"]
            ].to_string(index=False)
        )

    output_dir = Path("results/eda")
    output_dir.mkdir(parents=True, exist_ok=True)

    final_df = pd.DataFrame(all_results)

    final_df.to_csv(
        output_dir / "06_time_aware_lag_results.csv",
        index=False
    )

    print("\nSaved:")
    print("results/eda/06_time_aware_lag_results.csv")


if __name__ == "__main__":
    main()
