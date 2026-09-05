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


# ============================================================
# SETTINGS
# ============================================================

TRAIN_FILE = DATA_DIR / "DATA_GEO_Train.csv"

VALIDATION_DAYS = [
    pd.Timestamp("2025-09-02"),
    pd.Timestamp("2025-09-03"),
    pd.Timestamp("2025-09-04"),
    pd.Timestamp("2025-09-05"),
    pd.Timestamp("2025-09-06"),
    pd.Timestamp("2025-09-07"),
]

# The previously tested +/-5 min analog was the strongest
# standalone tolerance, so use it here.
ANALOG_TOLERANCE_MIN = 5

# Only activate the analog for sufficiently long horizons.
GATES = [360, 480, 600, 720, 840, 960, 1200]

LAMBDAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

CHAMPION_MEAN_W = 0.904758
CHAMPION_MEDIAN_W = 0.897807


# ============================================================
# METRICS
# ============================================================

def average_shapiro_w(residuals):
    """
    residuals: shape [n, 4]
    """
    values = []

    for j in range(4):
        W, _ = shapiro(residuals[:, j])
        values.append(W)

    return float(np.mean(values))


# ============================================================
# SAME-UTC-TIME ANALOG
# ============================================================

def build_same_time_analog(
    train_df,
    target_times,
    tolerance_minutes,
):
    """
    Causal historical same-time analog.

    For every validation target:
      - only observations strictly before target are eligible
      - compare UTC time-of-day
      - use circular 24-hour distance
      - choose closest match
      - ties prefer the latest historical observation

    Returns:
      predictions: [n, 4]
      matched:     [n]
    """

    predictions = np.full(
        (len(target_times), 4),
        np.nan,
        dtype=np.float64,
    )

    matched = np.zeros(
        len(target_times),
        dtype=bool,
    )

    # Precompute historical TOD in seconds.
    historical = train_df.copy()

    tod_seconds = (
        historical["utc_time"].dt.hour * 3600
        + historical["utc_time"].dt.minute * 60
        + historical["utc_time"].dt.second
        + historical["utc_time"].dt.microsecond / 1e6
    ).to_numpy(dtype=float)

    utc_values = historical["utc_time"].to_numpy()

    error_values = historical[
        ERROR_COLS
    ].to_numpy(dtype=float)

    max_diff = tolerance_minutes * 60.0

    for i, target_time in enumerate(target_times):

        target_seconds = (
            target_time.hour * 3600
            + target_time.minute * 60
            + target_time.second
            + target_time.microsecond / 1e6
        )

        diff = np.abs(
            tod_seconds - target_seconds
        )

        # Circular distance across midnight.
        diff = np.minimum(
            diff,
            86400.0 - diff,
        )

        # Causal constraint.
        causal = utc_values < np.datetime64(
            target_time.to_datetime64()
        )

        eligible = (
            causal
            & (diff <= max_diff)
        )

        if not np.any(eligible):
            continue

        candidate_idx = np.where(
            eligible
        )[0]

        # Closest time-of-day first,
        # latest UTC timestamp for ties.
        candidate_idx = sorted(
            candidate_idx,
            key=lambda k: (
                diff[k],
                -utc_values[k].astype("datetime64[ns]").astype(np.int64),
            ),
        )

        k = candidate_idx[0]

        predictions[i] = error_values[k]
        matched[i] = True

    return predictions, matched


# ============================================================
# EXACT CHAMPION FOLD
# ============================================================

def run_champion_fold(
    train_df,
    validation_df,
):
    """
    Reproduce the champion pipeline exactly using the
    actual API in model_gru_horizon_regime.py.
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
        valid_sequences,
        valid_horizons,
        valid_horizon_regimes,
        y_valid,
    ) = build_validation_examples(
        train_df,
        validation_df,
    )

    if len(train_sequences) == 0:
        raise RuntimeError(
            "No training examples created."
        )

    if len(valid_sequences) == 0:
        raise RuntimeError(
            "No validation examples created."
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

    train_sequences_scaled = [
        x_scaler.transform(seq)
        for seq in train_sequences
    ]

    valid_sequences_scaled = [
        x_scaler.transform(seq)
        for seq in valid_sequences
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

    valid_horizons_scaled = (
        horizon_scaler
        .transform(
            valid_horizons.reshape(-1, 1)
        )
        .ravel()
    )

    y_scaler = StandardScaler()
    y_scaler.fit(y_train)

    y_train_scaled = y_scaler.transform(
        y_train
    )

    # --------------------------------------------------------
    # EXACT CHAMPION PADDING
    # --------------------------------------------------------

    X_train, train_lengths = pad_sequences(
        train_sequences_scaled
    )

    X_valid, valid_lengths = pad_sequences(
        valid_sequences_scaled
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

    # --------------------------------------------------------
    # EXACT CHAMPION INFERENCE
    # --------------------------------------------------------

    model.eval()

    X_valid_tensor = torch.tensor(
        X_valid,
        dtype=torch.float32,
        device=DEVICE,
    )

    valid_lengths_tensor = torch.tensor(
        valid_lengths,
        dtype=torch.long,
        device=DEVICE,
    )

    valid_horizons_tensor = torch.tensor(
        valid_horizons_scaled.reshape(-1, 1),
        dtype=torch.float32,
        device=DEVICE,
    )

    valid_regimes_tensor = torch.tensor(
        valid_horizon_regimes,
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():

        pred_scaled = model(
            X_valid_tensor,
            valid_lengths_tensor,
            valid_horizons_tensor,
            valid_regimes_tensor,
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
        valid_horizons,
        y_valid,
        validation_df[
            validation_df["utc_time"].isin(
                [
                    seq_time
                    for seq_time in validation_df["utc_time"]
                ]
            )
        ].copy(),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(42)
    torch.manual_seed(42)

    full_df = load_data(
        TRAIN_FILE
    )

    print("=" * 110)
    print("GEO — HORIZON-GATED SAME-TIME ANALOG BLEND")
    print("=" * 110)
    print(f"Device:              {DEVICE}")
    print(f"Training rows:       {len(full_df)}")
    print(
        f"Analog tolerance:    +/- "
        f"{ANALOG_TOLERANCE_MIN} min"
    )
    print(
        f"Champion mean W:     "
        f"{CHAMPION_MEAN_W:.6f}"
    )
    print(
        f"Champion median W:   "
        f"{CHAMPION_MEDIAN_W:.6f}"
    )

    all_results = []
    fold_coverage = []

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

        validation_df = full_df[
            (
                full_df["utc_time"].dt.date
                == validation_day.date()
            )
        ].copy()

        (
            champion_pred,
            horizons,
            truth,
            _
        ) = run_champion_fold(
            train_df,
            validation_df,
        )

        # build_validation_examples preserves target order after sorting
        usable_validation = (
            validation_df
            .sort_values("utc_time")
            .reset_index(drop=True)
        )

        # It is possible for some early points to be unusable.
        # Rebuild exact usable timestamps from make_sequence.
        usable_sequences = []

        for _, row in usable_validation.iterrows():

            result = __import__(
                "model_gru_horizon_regime",
                fromlist=["make_sequence"],
            ).make_sequence(
                train_df,
                row["utc_time"],
            )

            if result is not None:
                usable_sequences.append(
                    row["utc_time"]
                )

        usable_times = pd.to_datetime(
            usable_sequences
        )

        analog_pred, matched = build_same_time_analog(
            train_df,
            list(usable_times),
            ANALOG_TOLERANCE_MIN,
        )

        champion_W = average_shapiro_w(
            truth - champion_pred
        )

        print(
            f"Champion fold W:    "
            f"{champion_W:.6f}"
        )

        print(
            f"Analog coverage:    "
            f"{int(matched.sum())}/{len(matched)} "
            f"({100.0 * matched.mean():.1f}%)"
        )

        fold_coverage.append(
            {
                "validation_day": validation_day.date(),
                "n": len(matched),
                "analog_matches": int(matched.sum()),
                "coverage_pct": 100.0 * matched.mean(),
            }
        )

        # ----------------------------------------------------
        # GRID
        # ----------------------------------------------------

        for gate in GATES:

            eligible = (
                matched
                & (horizons >= gate)
            )

            for lam in LAMBDAS:

                pred = champion_pred.copy()

                pred[eligible] = (
                    (1.0 - lam)
                    * champion_pred[eligible]
                    + lam
                    * analog_pred[eligible]
                )

                residuals = (
                    truth - pred
                )

                W = average_shapiro_w(
                    residuals
                )

                all_results.append(
                    {
                        "validation_day":
                            validation_day.date(),
                        "gate_min": gate,
                        "lambda": lam,
                        "W": W,
                        "n_blended":
                            int(eligible.sum()),
                    }
                )

                print(
                    f"gate>={gate:4d}  "
                    f"lambda={lam:>4.2f}  "
                    f"blended={int(eligible.sum()):2d}  "
                    f"W={W:.6f}"
                )

    # ========================================================
    # AGGREGATE
    # ========================================================

    result_df = pd.DataFrame(
        all_results
    )

    aggregate = (
        result_df
        .groupby(
            ["gate_min", "lambda"],
            as_index=False,
        )
        .agg(
            mean_W=("W", "mean"),
            median_W=("W", "median"),
            mean_blended=("n_blended", "mean"),
        )
        .sort_values(
            ["mean_W", "median_W"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    aggregate["delta_mean_W"] = (
        aggregate["mean_W"]
        - CHAMPION_MEAN_W
    )

    aggregate["delta_median_W"] = (
        aggregate["median_W"]
        - CHAMPION_MEDIAN_W
    )

    print("\n" + "=" * 110)
    print("AGGREGATED RESULTS")
    print("=" * 110)

    print(
        aggregate.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n" + "=" * 110)
    print("BEST CANDIDATE")
    print("=" * 110)

    best = aggregate.iloc[0]

    print(
        f"Gate:              >= {int(best['gate_min'])} min"
    )
    print(
        f"Lambda:            {best['lambda']:.2f}"
    )
    print(
        f"Mean W:            {best['mean_W']:.6f}"
    )
    print(
        f"Median W:          {best['median_W']:.6f}"
    )
    print(
        f"Delta mean W:      "
        f"{best['delta_mean_W']:+.6f}"
    )
    print(
        f"Delta median W:    "
        f"{best['delta_median_W']:+.6f}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    out_dir = (
        DATA_DIR.parent
        / "results"
        / "analog_blend"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        out_dir
        / "geo_analog_blend_fold_results.csv",
        index=False,
    )

    aggregate.to_csv(
        out_dir
        / "geo_analog_blend_aggregate.csv",
        index=False,
    )

    pd.DataFrame(
        fold_coverage
    ).to_csv(
        out_dir
        / "geo_analog_blend_coverage.csv",
        index=False,
    )

    print("\nSaved results to:")
    print(out_dir)


if __name__ == "__main__":
    main()
