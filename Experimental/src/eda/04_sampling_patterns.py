from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")

CSV_FILES = [
    "DATA_GEO_Train.csv",
    "DATA_GEO_Test.csv",
    "DATA_MEO_Train.csv",
    "DATA_MEO_Test.csv",
    "DATA_MEO_Train2.csv",
    "DATA_MEO_Test2.csv",
]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize column names.
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Parse timestamps.
    df["utc_time"] = pd.to_datetime(df["utc_time"])

    # Remove only exact duplicate records.
    df = df.drop_duplicates()

    # Sort chronologically.
    df = df.sort_values("utc_time").reset_index(drop=True)

    return df


def analyze_file(filename: str) -> None:
    df = load_data(DATA_DIR / filename)

    gaps = (
        df["utc_time"]
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60.0
    )

    print("\n" + "=" * 90)
    print(f"FILE: {filename}")
    print("=" * 90)

    print(f"Observations: {len(df)}")

    if gaps.empty:
        print("Only one observation.")
        return

    print(f"Median interval: {gaps.median():.2f} minutes")
    print(f"Mean interval:   {gaps.mean():.2f} minutes")
    print(f"Minimum interval:{gaps.min():.2f} minutes")
    print(f"Maximum interval:{gaps.max():.2f} minutes")

    print("\nInterval distribution:")

    counts = gaps.round(2).value_counts().sort_index()

    for interval, count in counts.items():
        percentage = count / len(gaps) * 100
        print(
            f"  {interval:8.2f} min : "
            f"{count:3d} observations "
            f"({percentage:5.1f}%)"
        )

    print("\nLargest gaps:")

    gap_df = pd.DataFrame(
        {
            "previous_time": df["utc_time"].shift(1),
            "current_time": df["utc_time"],
            "gap_minutes": gaps,
        }
    ).dropna()

    print(
        gap_df
        .sort_values("gap_minutes", ascending=False)
        .head(10)
        .to_string(index=False)
    )


def main() -> None:
    for filename in CSV_FILES:
        analyze_file(filename)


if __name__ == "__main__":
    main()
