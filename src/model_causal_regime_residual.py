import numpy as np
import pandas as pd

from pathlib import Path

from evaluation import ERROR_COLS


DATA_DIR = Path("data")

DATASETS = {
    "GEO": "DATA_GEO_Train.csv",
    "MEO": "DATA_MEO_Train.csv",
    "MEO2": "DATA_MEO_Train2.csv",
}

HISTORY_DAYS = 7
MIN_HISTORY = 3

# Excursion definition is deliberately kept simple for the
# framework. We will later learn/validate the regime detector.
EXCURSION_THRESHOLD_M = 20.0


def load_data(path):

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    )

    df["utc_time"] = pd.to_datetime(
        df["utc_time"]
    )

    for col in ERROR_COLS:
        df[col] = pd.to_numeric(
            df[col]
        )

    return (
        df
        .drop_duplicates()
        .dropna(
            subset=[
                "utc_time"
            ] + ERROR_COLS
        )
        .sort_values("utc_time")
        .reset_index(drop=True)
    )


def build_history(
    df,
    target_time,
):
    """
    Strictly causal history.

    Only observations strictly before target_time
    are returned.
    """

    target_time = pd.Timestamp(
        target_time
    )

    start = (
        target_time
        - pd.Timedelta(
            days=HISTORY_DAYS
        )
    )

    history = df[
        (df["utc_time"] >= start)
        & (df["utc_time"] < target_time)
    ].copy()

    return (
        history
        .sort_values("utc_time")
        .reset_index(drop=True)
    )


def persistence_baseline(
    history,
):
    """
    Simple leakage-free baseline:
    last observed error vector.
    """

    if len(history) == 0:
        return None

    return (
        history[
            ERROR_COLS
        ]
        .iloc[-1]
        .astype(float)
        .to_numpy()
    )


def one_step_residual(
    history,
    target,
):
    """
    Residual relative to the persistence baseline.

        residual = actual - last_observation

    This is only the first baseline residual definition.
    Later we will replace the baseline with the actual
    champion model in a walk-forward fashion.
    """

    baseline = persistence_baseline(
        history
    )

    if baseline is None:
        return None, None

    actual = (
        target[
            ERROR_COLS
        ]
        .astype(float)
        .to_numpy()
    )

    residual = actual - baseline

    return baseline, residual


def regime_label(
    actual,
):
    """
    Binary regime label based on the multivariate
    magnitude of the actual error.

    This label is used ONLY for offline analysis.
    It must never be available as an input at inference.
    """

    return int(
        np.max(
            np.abs(actual)
        ) >= EXCURSION_THRESHOLD_M
    )


def temporal_features(
    history,
):
    """
    Causal state representation.

    Uses only the observed history.
    """

    if len(history) < MIN_HISTORY:
        return None

    values = (
        history[
            ERROR_COLS
        ]
        .astype(float)
        .to_numpy()
    )

    d1 = np.diff(
        values,
        axis=0,
    )

    current = values[-1]
    previous = values[-2]

    latest_change = (
        current - previous
    )

    recent_mean = (
        np.mean(
            values,
            axis=0,
        )
    )

    recent_std = (
        np.std(
            values,
            axis=0,
        )
    )

    recent_abs_mean = (
        np.mean(
            np.abs(values),
            axis=0,
        )
    )

    recent_abs_max = (
        np.max(
            np.abs(values),
            axis=0,
        )
    )

    latest_change_abs = (
        np.abs(
            latest_change
        )
    )

    if len(d1) >= 2:

        previous_change = d1[-2]

        change_acceleration = (
            latest_change
            - previous_change
        )

    else:

        change_acceleration = (
            np.zeros(4)
        )

    gap_minutes = (
        history["utc_time"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
        .dropna()
    )

    latest_gap = (
        float(gap_minutes.iloc[-1])
        if len(gap_minutes)
        else 0.0
    )

    feature_vector = np.concatenate([
        current,
        previous,
        latest_change,
        latest_change_abs,
        recent_mean,
        recent_std,
        recent_abs_mean,
        recent_abs_max,
        change_acceleration,
        np.array([
            latest_gap
        ]),
    ])

    return feature_vector


def build_residual_dataset(
    df,
):
    """
    Build a strictly chronological residual dataset.

    At each target timestamp:
      - history contains only earlier observations
      - baseline uses only that history
      - residual uses the current target
      - regime label is stored only as a training/evaluation label
    """

    X = []
    Y = []
    regimes = []
    timestamps = []

    df = (
        df
        .sort_values("utc_time")
        .reset_index(drop=True)
    )

    for i in range(len(df)):

        target_time = df.loc[
            i,
            "utc_time",
        ]

        history = build_history(
            df.iloc[:i],
            target_time,
        )

        if len(history) < MIN_HISTORY:
            continue

        features = temporal_features(
            history
        )

        if features is None:
            continue

        baseline, residual = (
            one_step_residual(
                history,
                df.loc[i],
            )
        )

        if residual is None:
            continue

        actual = (
            df.loc[
                i,
                ERROR_COLS,
            ]
            .astype(float)
            .to_numpy()
        )

        X.append(features)
        Y.append(residual)

        regimes.append(
            regime_label(
                actual
            )
        )

        timestamps.append(
            target_time
        )

    return (
        np.asarray(
            X,
            dtype=float,
        ),
        np.asarray(
            Y,
            dtype=float,
        ),
        np.asarray(
            regimes,
            dtype=np.int64,
        ),
        pd.to_datetime(
            timestamps
        ),
    )


def inspect_dataset(
    name,
    df,
):
    """
    Offline inspection only.
    """

    X, residuals, regimes, times = (
        build_residual_dataset(
            df
        )
    )

    print()
    print("=" * 80)
    print(
        f"{name} — CAUSAL RESIDUAL DATASET"
    )
    print("=" * 80)

    print(
        f"Observations      : {len(df)}"
    )

    print(
        f"Residual samples  : {len(residuals)}"
    )

    print(
        f"Feature dimension : {X.shape[1]}"
    )

    print(
        f"Excursion samples : {int(regimes.sum())}"
    )

    print(
        f"Normal samples    : "
        f"{int((regimes == 0).sum())}"
    )

    if len(residuals) > 0:

        print(
            "\nResidual statistics:"
        )

        for j, col in enumerate(
            ERROR_COLS
        ):

            values = residuals[:, j]

            print(
                f"{col:20s} "
                f"mean={values.mean(): .6f} "
                f"std={values.std(ddof=1): .6f} "
                f"MAE={np.mean(np.abs(values)): .6f} "
                f"max_abs={np.max(np.abs(values)): .6f}"
            )

    if len(times) > 0:

        print(
            "\nTime range:"
        )

        print(
            f"{times.min()} → {times.max()}"
        )


def main():

    print(
        "\nCAUSAL REGIME RESIDUAL FRAMEWORK"
    )

    for dataset_name, filename in (
        DATASETS.items()
    ):

        df = load_data(
            DATA_DIR / filename
        )

        inspect_dataset(
            dataset_name,
            df,
        )


if __name__ == "__main__":
    main()
