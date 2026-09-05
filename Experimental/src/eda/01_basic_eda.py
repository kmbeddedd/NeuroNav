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

ERROR_COLUMNS = [
    "x_error (m)",
    "y_error (m)",
    "z_error (m)",
    "satclockerror (m)",
]


def analyze_file(filename: str) -> None:
    path = DATA_DIR / filename
    df = pd.read_csv(path)

    # Normalize whitespace in column names.
    df.columns = (
    df.columns
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

    # Parse timestamps.
    df["utc_time"] = pd.to_datetime(df["utc_time"], errors="coerce")

    # Convert error columns to numeric.
    for col in ERROR_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("utc_time").reset_index(drop=True)

    print("\n" + "=" * 90)
    print(f"FILE: {filename}")
    print("=" * 90)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Time range: {df['utc_time'].min()} -> {df['utc_time'].max()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Duplicate timestamps: {df['utc_time'].duplicated().sum()}")

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    # Time-gap analysis.
    gaps = (
        df["utc_time"]
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60.0
    )

    if not gaps.empty:
        print("\nTime gaps (minutes):")
        print(f"  Minimum : {gaps.min():.2f}")
        print(f"  Median  : {gaps.median():.2f}")
        print(f"  Maximum : {gaps.max():.2f}")

        print("\nMost common time gaps:")
        print(gaps.round(2).value_counts().head(10).to_string())

    print("\nError statistics:")
    stats = df[ERROR_COLUMNS].describe().T
    stats["missing"] = df[ERROR_COLUMNS].isna().sum()
    stats["unique"] = df[ERROR_COLUMNS].nunique()
    print(stats.to_string())

    print("\nLargest absolute values:")
    for col in ERROR_COLUMNS:
        if col in df.columns:
            max_abs = df[col].abs().max()
            print(f"  {col}: {max_abs:.6f} m")


def main() -> None:
    for filename in CSV_FILES:
        analyze_file(filename)


if __name__ == "__main__":
    main()
