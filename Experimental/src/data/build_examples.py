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


def load_data(path):

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["utc_time"] = pd.to_datetime(
        df["utc_time"]
    )

    for col in ERROR_COLS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return (
        df
        .dropna(
            subset=["utc_time"] + ERROR_COLS
        )
        .drop_duplicates()
        .sort_values("utc_time")
        .reset_index(drop=True)
    )


def build_examples(df, history_days=7):

    examples = []

    for i in range(1, len(df)):

        target_time = df.loc[i, "utc_time"]

        history_start = (
            target_time
            - pd.Timedelta(days=history_days)
        )

        history = df[
            (df["utc_time"] >= history_start)
            & (df["utc_time"] < target_time)
        ].copy()

        if len(history) < 3:
            continue

        history["delta_to_target_min"] = (
            (
                target_time
                - history["utc_time"]
            )
            .dt.total_seconds()
            / 60.0
        )

        history["gap_from_previous_min"] = (
            history["utc_time"]
            .diff()
            .dt.total_seconds()
            / 60.0
        )

        # Store one complete variable-length history.
        examples.append({
            "target_time": target_time,
            "history_times": history[
                "utc_time"
            ].tolist(),
            "history_values": history[
                ERROR_COLS
            ].to_numpy(dtype=float),
            "delta_to_target_min": history[
                "delta_to_target_min"
            ].to_numpy(dtype=float),
            "gap_from_previous_min": history[
                "gap_from_previous_min"
            ].fillna(0).to_numpy(dtype=float),
            "target": df.loc[
                i,
                ERROR_COLS
            ].to_numpy(dtype=float),
        })

    return examples


def main():

    output_dir = Path(
        "results/data"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for name, filename in DATASETS.items():

        df = load_data(
            DATA_DIR / filename
        )

        examples = build_examples(df)

        print(f"\n{name}")
        print("=" * 70)
        print(
            f"Raw observations: {len(df)}"
        )
        print(
            f"Training examples: {len(examples)}"
        )

        if not examples:
            continue

        history_lengths = np.array([
            len(x["history_values"])
            for x in examples
        ])

        print(
            f"History length: "
            f"min={history_lengths.min()}, "
            f"median={np.median(history_lengths):.0f}, "
            f"max={history_lengths.max()}"
        )

        print(
            f"Mean history length: "
            f"{history_lengths.mean():.2f}"
        )

        output = {
            "target_time": [
                x["target_time"]
                for x in examples
            ],
            "history_length": [
                len(x["history_values"])
                for x in examples
            ],
        }

        pd.DataFrame(output).to_csv(
            output_dir /
            f"{name}_example_summary.csv",
            index=False
        )


if __name__ == "__main__":
    main()
