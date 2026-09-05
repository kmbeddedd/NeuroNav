
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from evaluation import evaluate_all_targets


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

MIN_VALIDATION_OBS = 5


def load_data(path):

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["utc_time"] = pd.to_datetime(df["utc_time"])

    for col in ERROR_COLS:
        df[col] = pd.to_numeric(df[col])

    return (
        df.drop_duplicates()
        .sort_values("utc_time")
        .reset_index(drop=True)
    )


def make_features(history, target_time):

    last = history.iloc[-1]
    previous = history.iloc[-2]

    values_last = last[ERROR_COLS].astype(float).to_numpy()
    values_previous = previous[ERROR_COLS].astype(float).to_numpy()

    delta_values = values_last - values_previous

    gap_previous = (
        last["utc_time"] - previous["utc_time"]
    ).total_seconds() / 60.0

    time_to_target = (
        target_time - last["utc_time"]
    ).total_seconds() / 60.0

    hour = (
        target_time.hour
        + target_time.minute / 60.0
    )

    # Cyclic encoding of time of day.
    hour_sin = np.sin(
        2 * np.pi * hour / 24.0
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24.0
    )

    return np.concatenate([
        values_last,
        values_previous,
        delta_values,
        [gap_previous],
        [time_to_target],
        [hour_sin],
        [hour_cos],
    ])


def build_day_examples(df, validation_day):

    history = df[
        df["utc_time"] < validation_day
    ].copy()

    validation = df[
        df["utc_time"].dt.normalize()
        == validation_day
    ].copy()

    if len(history) < 3:
        return None, None, None

    X = []
    y = []
    timestamps = []

    for _, row in validation.iterrows():

        target_time = row["utc_time"]

        available_history = history[
            history["utc_time"] < target_time
        ]

        if len(available_history) < 2:
            continue

        features = make_features(
            available_history,
            target_time
        )

        X.append(features)
        y.append(
            row[ERROR_COLS]
            .astype(float)
            .to_numpy()
        )
        timestamps.append(target_time)

    if not X:
        return None, None, None

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=float),
        timestamps,
    )


def main():

    print("\nRANDOM FOREST TEMPORAL BASELINE")
    print("=" * 90)

    fold_results = []

    for dataset_name, filename in DATASETS.items():

        df = load_data(
            DATA_DIR / filename
        )

        days = sorted(
            df["utc_time"]
            .dt.normalize()
            .unique()
        )

        print(f"\n{dataset_name}")
        print("-" * 90)

        for validation_day in days[1:]:

            validation = df[
                df["utc_time"].dt.normalize()
                == validation_day
            ]

            if len(validation) < MIN_VALIDATION_OBS:
                continue

            X_val, y_val, _ = build_day_examples(
                df,
                validation_day
            )

            if X_val is None:
                continue

            train_end = validation_day

            train_df = df[
                df["utc_time"] < train_end
            ].copy()

            # Create one-step-ahead training examples
            # from historical observations.
            X_train = []
            y_train = []

            for i in range(2, len(train_df)):

                history = train_df.iloc[:i]

                target_time = train_df.iloc[
                    i
                ]["utc_time"]

                X_train.append(
                    make_features(
                        history,
                        target_time
                    )
                )

                y_train.append(
                    train_df.iloc[i][ERROR_COLS]
                    .astype(float)
                    .to_numpy()
                )

            X_train = np.asarray(
                X_train,
                dtype=float
            )

            y_train = np.asarray(
                y_train,
                dtype=float
            )

            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )

            model = MultiOutputRegressor(
                model
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_val
            )

            residuals = (
                y_val - predictions
            )

            results = evaluate_all_targets(
                residuals
            )

            print(
                f"\nValidation day: "
                f"{validation_day.date()}"
            )

            for col in ERROR_COLS:

                r = results[col]

                print(
                    f"  {col:<20} "
                    f"W={r['W']:.6f} "
                    f"p={r['p']:.6f} "
                    f"mean={r['mean']: .6f} "
                    f"std={r['std']:.6f}"
                )

            print(
                f"  Average W = "
                f"{results['average_W']:.6f}"
            )

            fold_results.append({
                "dataset": dataset_name,
                "validation_day":
                    validation_day.date(),
                "average_W":
                    results["average_W"],
            })

    output_dir = Path(
        "results/models"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(fold_results).to_csv(
        output_dir /
        "random_forest_fold_scores.csv",
        index=False
    )

    print("\n")
    print("=" * 90)
    print("RANDOM FOREST OVERALL")
    print("=" * 90)

    result_df = pd.DataFrame(
        fold_results
    )

    print(
        f"Mean fold Average W: "
        f"{result_df['average_W'].mean():.6f}"
    )

    print(
        f"Median fold Average W: "
        f"{result_df['average_W'].median():.6f}"
    )

    print("\nSaved:")
    print(
        "results/models/"
        "random_forest_fold_scores.csv"
    )


if __name__ == "__main__":
    main()