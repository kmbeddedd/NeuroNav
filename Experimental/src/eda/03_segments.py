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

# A gap larger than this starts a new observation segment.
# This is an analysis threshold, not a model assumption.
SEGMENT_GAP_MINUTES = 30


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Make harmless header-spacing differences consistent.
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["utc_time"] = pd.to_datetime(df["utc_time"])
    df = df.drop_duplicates()
    df = df.sort_values("utc_time").reset_index(drop=True)

    return df


def analyze_segments(filename: str) -> None:
    df = load_data(DATA_DIR / filename)

    gaps = (
        df["utc_time"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    # A new segment starts after a large time gap.
    df["segment_id"] = (gaps > SEGMENT_GAP_MINUTES).cumsum()

    print("\n" + "=" * 90)
    print(f"FILE: {filename}")
    print("=" * 90)

    print(f"Total observations: {len(df)}")
    print(f"Number of segments: {df['segment_id'].nunique()}")

    print("\nSegment details:")

    for segment_id, seg in df.groupby("segment_id"):
        seg_gaps = (
            seg["utc_time"]
            .diff()
            .dropna()
            .dt.total_seconds()
            .div(60)
        )

        duration_hours = (
            seg["utc_time"].iloc[-1] - seg["utc_time"].iloc[0]
        ).total_seconds() / 3600

        print(
            f"\nSegment {segment_id + 1}"
            f"\n  Start           : {seg['utc_time'].iloc[0]}"
            f"\n  End             : {seg['utc_time'].iloc[-1]}"
            f"\n  Observations    : {len(seg)}"
            f"\n  Duration (hours): {duration_hours:.2f}"
        )

        if not seg_gaps.empty:
            print(
                f"  Median gap (min): {seg_gaps.median():.2f}"
                f"\n  Min gap (min)   : {seg_gaps.min():.2f}"
                f"\n  Max gap (min)   : {seg_gaps.max():.2f}"
            )
        else:
            print("  Internal gaps    : none (single observation)")


def main() -> None:
    for filename in CSV_FILES:
        analyze_segments(filename)


if __name__ == "__main__":
    main()
