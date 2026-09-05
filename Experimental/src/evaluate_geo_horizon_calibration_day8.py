import numpy as np
import pandas as pd
from scipy.stats import shapiro

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

from sklearn.preprocessing import StandardScaler
import torch


TRAIN_FILE = DATA_DIR / "DATA_GEO_Train.csv"
TEST_FILE = DATA_DIR / "DATA_GEO_Test.csv"
PRED_FILE = (
    DATA_DIR.parent
    / "results"
    / "predictions"
    / "GEO_champion_predictions.csv"
)

WINDOW_HOURS = 12
HORIZON_LO = 720
HORIZON_HI = 1200
SHRINKAGE = 0.25


def average_w(residuals):
    Ws = []

    for j in range(4):
        W, _ = shapiro(residuals[:, j])
        Ws.append(W)

    return float(np.mean(Ws))


def champion_predict(train_df, target_df):

    (
        train_sequences,
        train_horizons,
        train_regimes,
        y_train,
    ) = build_training_examples(train_df)

    (
        target_sequences,
        target_horizons,
        target_regimes,
        y_target,
    ) = build_validation_examples(
        train_df,
        target_df,
    )

    x_scaler = StandardScaler()

    x_scaler.fit(
        np.concatenate(
            train_sequences,
            axis=0,
        )
    )

    train_sequences = [
        x_scaler.transform(x)
        for x in train_sequences
    ]

    target_sequences = [
        x_scaler.transform(x)
        for x in target_sequences
    ]

    horizon_scaler = StandardScaler()

    horizon_scaler.fit(
        train_horizons.reshape(-1, 1)
    )

    train_h_scaled = (
        horizon_scaler
        .transform(
            train_horizons.reshape(-1, 1)
        )
        .ravel()
    )

    target_h_scaled = (
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

    torch.manual_seed(42)
    np.random.seed(42)

    model = train_model(
        X_train,
        train_lengths,
        train_h_scaled,
        train_horizons,
        train_regimes,
        y_train_scaled,
    )

    model.eval()

    with torch.no_grad():

        pred_scaled = model(
            torch.tensor(
                X_target,
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.tensor(
                target_lengths,
                dtype=torch.long,
                device=DEVICE,
            ),
            torch.tensor(
                target_h_scaled.reshape(-1, 1),
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.tensor(
                target_regimes,
                dtype=torch.long,
                device=DEVICE,
            ),
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

    train = load_data(
        TRAIN_FILE
    )

    test_truth = load_data(
        TEST_FILE
    )

    champion = pd.read_csv(
        PRED_FILE
    )

    champion["utc_time"] = pd.to_datetime(
        champion["utc_time"]
    )

    test_truth["utc_time"] = pd.to_datetime(
        test_truth["utc_time"]
    )

    # ========================================================
    # STEP 1: EXACT EXISTING DAY-8 CHAMPION
    # ========================================================

    merged = champion.merge(
        test_truth[
            ["utc_time"] + ERROR_COLS
        ],
        on="utc_time",
        how="inner",
        suffixes=("", "_truth"),
    )

    truth_day8 = merged[
        ERROR_COLS
    ].to_numpy(dtype=float)

    champion_pred = merged[
        [
            "pred_x_error (m)",
            "pred_y_error (m)",
            "pred_z_error (m)",
            "pred_satclockerror (m)",
        ]
    ].to_numpy(dtype=float)

    day8_times = (
        merged["utc_time"]
        .sort_values()
        .reset_index(drop=True)
    )

    # Make sure truth/prediction order matches time order.
    merged = (
        merged
        .sort_values("utc_time")
        .reset_index(drop=True)
    )

    truth_day8 = merged[
        ERROR_COLS
    ].to_numpy(dtype=float)

    champion_pred = merged[
        [
            "pred_x_error (m)",
            "pred_y_error (m)",
            "pred_z_error (m)",
            "pred_satclockerror (m)",
        ]
    ].to_numpy(dtype=float)

    horizons_day8 = merged[
        "horizon_min"
    ].to_numpy(dtype=float)

    # ========================================================
    # STEP 2: BUILD 12-HOUR CALIBRATION SET
    #
    # Day 8 begins 2025-09-08.
    # Calibration window is the preceding 12 hours:
    # 2025-09-07 12:00 -> 2025-09-08 00:00.
    #
    # The model is trained only before that window.
    # ========================================================

    day8_start = pd.Timestamp(
        "2025-09-08 00:00:00"
    )

    calibration_start = (
        day8_start
        - pd.Timedelta(
            hours=WINDOW_HOURS
        )
    )

    calibration_train = train[
        train["utc_time"]
        < calibration_start
    ].copy()

    calibration_df = train[
        (
            train["utc_time"]
            >= calibration_start
        )
        & (
            train["utc_time"]
            < day8_start
        )
    ].copy()

    (
        calibration_pred,
        calibration_horizons,
        calibration_truth,
    ) = champion_predict(
        calibration_train,
        calibration_df,
    )

    calibration_residuals = (
        calibration_truth
        - calibration_pred
    )

    # ========================================================
    # STEP 3: ESTIMATE BIAS IN THE WINNING HORIZON BIN
    # ========================================================

    mask = (
        (calibration_horizons >= HORIZON_LO)
        & (calibration_horizons < HORIZON_HI)
    )

    if not np.any(mask):
        raise RuntimeError(
            "No calibration observations in the "
            "720-1200 minute horizon bin."
        )

    bias = np.mean(
        calibration_residuals[mask],
        axis=0,
    )

    correction = (
        SHRINKAGE
        * bias
    )

    print("=" * 105)
    print("REAL GEO DAY-8 — HORIZON CALIBRATION")
    print("=" * 105)

    print(
        f"Calibration window: "
        f"{calibration_start} -> {day8_start}"
    )

    print(
        f"Calibration points: "
        f"{len(calibration_df)}"
    )

    print(
        f"720-1200 min calibration points: "
        f"{int(mask.sum())}"
    )

    print(
        f"Estimated mean residual "
        f"[X,Y,Z,Clock]:\n{bias}"
    )

    print(
        f"Applied correction "
        f"(25%):\n{correction}"
    )

    # ========================================================
    # STEP 4: APPLY ONLY TO 720-1200 MIN DAY-8 PREDICTIONS
    # ========================================================

    final_pred = champion_pred.copy()

    apply_mask = (
        (horizons_day8 >= HORIZON_LO)
        & (horizons_day8 < HORIZON_HI)
    )

    final_pred[apply_mask] -= correction

    print(
        f"Day-8 corrected points: "
        f"{int(apply_mask.sum())}"
    )

    # ========================================================
    # STEP 5: SCORE
    # ========================================================

    champion_residuals = (
        truth_day8
        - champion_pred
    )

    calibrated_residuals = (
        truth_day8
        - final_pred
    )

    print("\n" + "-" * 105)
    print("CHAMPION")
    print("-" * 105)

    champion_W = average_w(
        champion_residuals
    )

    for j, label in enumerate(
        ["X", "Y", "Z", "Clock"]
    ):
        W, p = shapiro(
            champion_residuals[:, j]
        )

        print(
            f"{label:5s} "
            f"W={W:.6f} "
            f"p={p:.6g} "
            f"mean={champion_residuals[:, j].mean():+.4f} "
            f"std={champion_residuals[:, j].std(ddof=1):.4f}"
        )

    print(
        f"Average W = {champion_W:.6f}"
    )

    print("\n" + "-" * 105)
    print("HORIZON CALIBRATED")
    print("-" * 105)

    calibrated_W = average_w(
        calibrated_residuals
    )

    for j, label in enumerate(
        ["X", "Y", "Z", "Clock"]
    ):
        W, p = shapiro(
            calibrated_residuals[:, j]
        )

        print(
            f"{label:5s} "
            f"W={W:.6f} "
            f"p={p:.6g} "
            f"mean={calibrated_residuals[:, j].mean():+.4f} "
            f"std={calibrated_residuals[:, j].std(ddof=1):.4f}"
        )

    print(
        f"Average W = {calibrated_W:.6f}"
    )

    print("\n" + "=" * 105)
    print("RESULT")
    print("=" * 105)

    print(
        f"Champion W      : {champion_W:.6f}"
    )

    print(
        f"Calibrated W    : {calibrated_W:.6f}"
    )

    print(
        f"Delta            : "
        f"{calibrated_W - champion_W:+.6f}"
    )

    if calibrated_W > champion_W:
        print("WIN: calibration improves Day-8 W.")
    else:
        print("NO WIN: calibration does not improve Day-8 W.")


if __name__ == "__main__":
    main()
