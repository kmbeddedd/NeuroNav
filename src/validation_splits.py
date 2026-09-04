import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

DATASETS = {
    "GEO": "DATA_GEO_Train.csv",
    "MEO": "DATA_MEO_Train.csv",
    "MEO2": "DATA_MEO_Train2.csv",
}


def load_data(path):
    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["utc_time"] = pd.to_datetime(df["utc_time"])

    return (
        df.drop_duplicates()
        .sort_values("utc_time")
        .reset_index(drop=True)
    )


def main():

    print("\nLEAKAGE-SAFE DAY-BASED VALIDATION")
    print("=" * 80)

    for name, filename in DATASETS.items():

        df = load_data(DATA_DIR / filename)

        days = sorted(
            df["utc_time"]
            .dt.normalize()
            .unique()
        )

        print(f"\n{name}")
        print("-" * 80)

        # Use every available day after the first
        # as a possible next-day validation target.
        for i in range(1, len(days)):

            train_days = days[:i]
            validation_day = days[i]

            train_df = df[
                df["utc_time"].dt.normalize().isin(train_days)
            ]

            val_df = df[
                df["utc_time"].dt.normalize() == validation_day
            ]

            print(
                f"Train: "
                f"{train_days[0].date()} → {train_days[-1].date()} "
                f"({len(train_df):3d} obs)"
                f" | "
                f"Validation: {validation_day.date()} "
                f"({len(val_df):3d} obs)"
            )


if __name__ == "__main__":
    main()
