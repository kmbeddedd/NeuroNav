import numpy as np
import pandas as pd
import torch

from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler

from model_gru_horizon_regime import (
    load_data,
    build_training_examples,
    build_validation_examples,
    pad_sequences,
    train_model,
    DEVICE,
    DATA_DIR,
    ERROR_COLS,
)


TRAIN_FILE = DATA_DIR / "DATA_GEO_Train.csv"

VALIDATION_DAYS = [
    pd.Timestamp("2025-09-02"),
    pd.Timestamp("2025-09-03"),
    pd.Timestamp("2025-09-04"),
    pd.Timestamp("2025-09-05"),
    pd.Timestamp("2025-09-06"),
    pd.Timestamp("2025-09-07"),
]

# Calibration history lengths.
CAL_WINDOWS_HOURS = [6, 12, 24, 48]

# Fraction of estimated bias to remove.
SHRINKAGES = [0.25, 0.50, 0.75, 1.00]

CHAMPION_MEAN_W = 0.904758
CHAMPION_MEDIAN_W = 0.897807


def average_shapiro_w(residuals):
    Ws = []

    for j in range(4):
        W, _ = shapiro(residuals[:, j])
        Ws.append(W)

    return float(np.mean(Ws))


def train_and_predict(
    train_df,
    target_df,
):
    """
    Exact champion preprocessing/model pipeline.

    train_df:
        information available before all target timestamps.

    target_df:
        timestamps and truth used only for evaluation.
    """

    (
        train_sequences,
        train_horizons,
        train_horizon_regimes,
        y_train,
    ) = build_training_examples(
        train_df
    )

    (
        target_sequences,
        target_horizons,
        target_horizon_regimes,
        y_target,
    ) = build_validation_examples(
        train_df,
        target_df,
    )

    if len(train_sequences) == 0:
        raise RuntimeError(
            "No training examples."
        )

    if len(target_sequences) == 0:
        raise RuntimeError(
            "No target examples."
        )

    # --------------------------------------------------------
    # EXACT CHAMPION SCALING
    # --------------------------------------------------------

    all_train_steps = np.concatenate(
        train_sequences,
        axis=0,
    )

    x_scaler = StandardScaler()
    x_scaler.fit(all_train_steps)

    train_sequences = [
        x_scaler.transform(seq)
        for seq in train_sequences
    ]

    target_sequences = [
        x_scaler.transform(seq)
        for seq in target_sequences
    ]

    horizon_scaler = StandardScaler()
    horizon_scaler.fit(
        train_horizons.reshape(-1, 1)
    )

    train_horizons_scaled = (
        horizon_scaler
        .transform(
            train_horizons.reshape(-1, 1)
        )
        .ravel()
    )

    target_horizons_scaled = (
        horizon_scaler
        .transform(
            target_horizons.reshape(-1, 1)
        )
        .ravel()
    )

    y_scaler = StandardScaler()
    y_scaler.fit(y_train)

    y_train_scaled = y_scaler.transform(
        y_train
    )

    X_train, train_lengths = pad_sequences(
        train_sequences
    )

    X_target, target_lengths = pad_sequences(
        target_sequences
    )

    # --------------------------------------------------------
    # EXACT CHAMPION TRAINING
    # --------------------------------------------------------

    torch.manual_seed(42)
    np.random.seed(42)

    model = train_model(
        X_train,
        train_lengths,
        train_horizons_scaled,
        train_horizons,
        train_horizon_regimes,
        y_train_scaled,
    )

    model.eval()

    # --------------------------------------------------------
    # EXACT CHAMPION INFERENCE
    # --------------------------------------------------------

    X_target_t = torch.tensor(
        X_target,
        dtype=torch.float32,
        device=DEVICE,
    )

    target_lengths_t = torch.tensor(
        target_lengths,
        dtype=torch.long,
        device=DEVICE,
    )

    target_horizons_t = torch.tensor(
        target_horizons_scaled.reshape(-1, 1),
        dtype=torch.float32,
        device=DEVICE,
    )

    target_regimes_t = torch.tensor(
        target_horizon_regimes,
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():

        pred_scaled = model(
            X_target_t,
            target_lengths_t,
            target_horizons_t,
            target_regimes_t,
        )

    pred_scaled = (
        pred_scaled
        .detach()
        .cpu()
        .numpy()
    )

    predictions = y_scaler.inverse_transform(
        pred_scaled
    )

    return (
        predictions,
        target_horizons,
        y_target,
    )


def main():

    full_df = load_data(
        TRAIN_FILE
    )

    print("=" * 110)
    print("GEO — LEAKAGE-SAFE RECENT RESIDUAL CALIBRATION")
    print("=" * 110)

    print(
        f"Device:             {DEVICE}"
    )

    print(
        f"Champion mean W:    "
        f"{CHAMPION_MEAN_W:.6f}"
    )

    print(
        f"Champion median W:  "
        f"{CHAMPION_MEDIAN_W:.6f}"
    )

    results = []

    # ========================================================
    # CHRONOLOGICAL VALIDATION
    # ========================================================

    for validation_day in VALIDATION_DAYS:

        print("\n" + "=" * 110)
        print(
            f"VALIDATION DAY: "
            f"{validation_day.date()}"
        )
        print("=" * 110)

        train_df = full_df[
            full_df["utc_time"] < validation_day
        ].copy()

        validation_df = (
            full_df[
                (
                    full_df["utc_time"].dt.date
                    == validation_day.date()
                )
            ]
            .sort_values("utc_time")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # BASE CHAMPION FOR THIS FOLD
        # ----------------------------------------------------

        (
            champion_pred,
            horizons,
            truth,
        ) = train_and_predict(
            train_df,
            validation_df,
        )

        champion_W = average_shapiro_w(
            truth - champion_pred
        )

        print(
            f"Champion W: "
            f"{champion_W:.6f}"
        )

        # ----------------------------------------------------
        # TRY DIFFERENT RECENT CALIBRATION WINDOWS
        # ----------------------------------------------------

        for window_hours in CAL_WINDOWS_HOURS:

            calibration_start = (
                validation_day
                - pd.Timedelta(
                    hours=window_hours
                )
            )

            calibration_df = train_df[
                train_df["utc_time"]
                >= calibration_start
            ].copy()

            calibration_train_df = train_df[
                train_df["utc_time"]
                < calibration_start
            ].copy()

            if len(calibration_train_df) < 3:
                continue

            if len(calibration_df) < 3:
                continue

            # ------------------------------------------------
            # WALK-FORWARD CALIBRATION MODEL
            #
            # It only sees data strictly before the
            # calibration window.
            # ------------------------------------------------

            (
                calibration_pred,
                _,
                calibration_truth,
            ) = train_and_predict(
                calibration_train_df,
                calibration_df,
            )

            calibration_residuals = (
                
                calibration_truth
                - calibration_pred
            )

            print("DEBUG calibration residual mean:", np.mean(calibration_residuals, axis=0))
            bias_mean = (
                np.mean(
                    calibration_residuals,
                    axis=0,
                )
            )

            bias_median = (
                np.median(
                    calibration_residuals,
                    axis=0,
                )
            )

            for bias_type, bias in [
                ("mean", bias_mean),
                ("median", bias_median),
            ]:

                for shrinkage in SHRINKAGES:

                    correction = (
                        shrinkage
                        * bias
                    )

                    calibrated_pred = (
                        champion_pred
                        - correction
                    )

                    residuals = (
                        truth
                        - calibrated_pred
                    )

                    W = average_shapiro_w(
                        residuals
                    )

                    results.append(
                        {
                            "validation_day":
                                validation_day.date(),
                            "window_hours":
                                window_hours,
                            "bias_type":
                                bias_type,
                            "shrinkage":
                                shrinkage,
                            "W":
                                W,
                            "champion_W":
                                champion_W,
                            "delta_W":
                                W - champion_W,
                        }
                    )

                    print(
                        f"window={window_hours:>2}h "
                        f"type={bias_type:>6s} "
                        f"shrink={shrinkage:.2f} "
                        f"W={W:.6f} "
                        f"delta={W - champion_W:+.6f}"
                    )


    # ========================================================
    # AGGREGATE
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    aggregate = (
        result_df
        .groupby(
            [
                "window_hours",
                "bias_type",
                "shrinkage",
            ],
            as_index=False,
        )
        .agg(
            mean_W=("W", "mean"),
            median_W=("W", "median"),
            mean_delta_W=("delta_W", "mean"),
        )
        .sort_values(
            ["mean_W", "median_W"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 110)
    print("AGGREGATED CALIBRATION RESULTS")
    print("=" * 110)

    print(
        aggregate.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ========================================================
    # BEST
    # ========================================================

    best = aggregate.iloc[0]

    print("\n" + "=" * 110)
    print("BEST CALIBRATION")
    print("=" * 110)

    print(
        f"Window:       {int(best['window_hours'])} h"
    )

    print(
        f"Bias type:    {best['bias_type']}"
    )

    print(
        f"Shrinkage:    {best['shrinkage']:.2f}"
    )

    print(
        f"Mean W:       {best['mean_W']:.6f}"
    )

    print(
        f"Median W:     {best['median_W']:.6f}"
    )

    print(
        f"Mean delta W: "
        f"{best['mean_delta_W']:+.6f}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    out_dir = (
        DATA_DIR.parent
        / "results"
        / "recent_calibration"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        out_dir
        / "geo_recent_calibration_fold_results.csv",
        index=False,
    )

    aggregate.to_csv(
        out_dir
        / "geo_recent_calibration_aggregate.csv",
        index=False,
    )

    print("\nSaved to:")
    print(out_dir)


if __name__ == "__main__":
    main()
