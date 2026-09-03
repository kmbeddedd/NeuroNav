from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path("data")
RESULTS_DIR = Path("results/eda")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILES = [
    "DATA_GEO_Train.csv",
    "DATA_GEO_Test.csv",
    "DATA_MEO_Train.csv",
    "DATA_MEO_Test.csv",
    "DATA_MEO_Train2.csv",
    "DATA_MEO_Test2.csv",
]

ERROR_COLUMNS = [
    "x_error (m)",
    "y_error (m)",
    "z_error (m)",
    "satclockerror (m)",
]

# A gap larger than this is treated as missing time,
# so we do not draw a misleading line across it.
MAX_GAP_MINUTES = 30


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["utc_time"] = pd.to_datetime(df["utc_time"])
    df = df.drop_duplicates()
    df = df.sort_values("utc_time").reset_index(drop=True)

    for col in ERROR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main() -> None:
    for filename in CSV_FILES:
        df = load_data(DATA_DIR / filename)

        # Calculate the time gap between consecutive observations.
        gap_minutes = (
            df["utc_time"]
            .diff()
            .dt.total_seconds()
            .div(60)
        )

        # Insert NaN after large gaps.
        # Matplotlib will break the line at NaN.
        for col in ERROR_COLUMNS:
            df.loc[gap_minutes > MAX_GAP_MINUTES, col] = float("nan")

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(14, 12),
            sharex=True,
        )

        for ax, col in zip(axes, ERROR_COLUMNS):
            ax.plot(
                df["utc_time"],
                df[col],
                marker="o",
                markersize=3,
            )
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("UTC time")
        fig.suptitle(
            f"{filename} — gap-aware ({MAX_GAP_MINUTES} min threshold)"
        )

        fig.tight_layout()

        output = RESULTS_DIR / filename.replace(
            ".csv", "_gap_aware.png"
        )

        fig.savefig(output, dpi=150)
        plt.close(fig)

        print(f"Saved: {output}")


if __name__ == "__main__":
    main()