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

ERRORS = [
    "x_error (m)",
    "y_error (m)",
    "z_error (m)",
    "satclockerror (m)",
]

# We use row lags first to understand temporal memory in each series.
LAGS = [1, 2, 3, 4, 5, 6, 12]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["utc_time"] = pd.to_datetime(df["utc_time"])

    # Keep only genuinely distinct observations.
    df = (
        df.drop_duplicates()
        .sort_values("utc_time")
        .reset_index(drop=True)
    )

    return df


def analyze_file(filename: str) -> None:
    df = load_data(DATA_DIR / filename)

    print("\n" + "=" * 90)
    print(filename)
    print("=" * 90)

    print("\nAutocorrelation by row lag:")
    print("lag = number of previous observations, not fixed minutes")

    for col in ERRORS:
        print(f"\n{col}")

        series = pd.to_numeric(df[col], errors="coerce")

        for lag in LAGS:
            if len(series) > lag:
                value = series.autocorr(lag=lag)
                print(f"  lag {lag:2d}: {value: .4f}")


def main() -> None:
    for filename in CSV_FILES:
        analyze_file(filename)


if __name__ == "__main__":
    main()
