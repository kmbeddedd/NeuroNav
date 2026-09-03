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

# Actual elapsed-time ranges in minutes.
BINS = [0, 7, 12, 20, 40, 90, 180, 360, np.inf]

LABELS = [
    "0-7",
    "7-12",
    "12-20",
    "20-40",
    "40-90",
    "90-180",
    "180-360",
    "360+",
]


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

    df = (
        df.dropna(subset=["utc_time"] + ERROR_COLS)
        .drop_duplicates()
        .sort_values("utc_time")
        .reset_index(drop=True)
    )

    return df


def main():

    print("\nGAP-BINNED AUTOCORRELATION")
    print("=" * 80)

    output_rows = []

    for filename in FILES:

        df = load_data(DATA_DIR / filename)

        # Actual time difference between consecutive observations.
        gap_minutes = (
            df["utc_time"].diff()
            .dt.total_seconds()
            .div(60)
        )

        df["gap_minutes"] = gap_minutes

        print(f"\n{filename}")
        print("-" * 80)

        for label, lower, upper in zip(
            LABELS,
            BINS[:-1],
            BINS[1:]
        ):

            mask = (
                (df["gap_minutes"] > lower)
                & (df["gap_minutes"] <= upper)
            )

            subset = df.loc[mask]

            row = {
                "dataset": filename,
                "gap_bin_min": label,
                "pairs": len(subset),
            }

            print(f"\nGap {label} min | pairs = {len(subset)}")

            for col in ERROR_COLS:

                if len(subset) < 5:
                    corr = np.nan

                else:

                    previous = df.loc[
                        subset.index - 1,
                        col
                    ].to_numpy()

                    current = subset[col].to_numpy()

                    if (
                        np.std(previous) == 0
                        or np.std(current) == 0
                    ):
                        corr = np.nan

                    else:
                        corr = np.corrcoef(
                            previous,
                            current
                        )[0, 1]

                row[col] = corr

                if pd.isna(corr):
                    text = "NaN"
                else:
                    text = f"{corr: .3f}"

                print(f"  {col:<20} {text}")

            output_rows.append(row)

    output_dir = Path("results/eda")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    result = pd.DataFrame(output_rows)

    result.to_csv(
        output_dir / "07_gap_binned_autocorrelation.csv",
        index=False
    )

    print("\nSaved:")
    print(
        "results/eda/07_gap_binned_autocorrelation.csv"
    )


if __name__ == "__main__":
    main()
