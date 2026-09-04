import numpy as np
import pandas as pd
import torch
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler

from model_gru_horizon_regime import (
    load_data,
    build_training_examples,
    pad_sequences,
    train_model,
    DEVICE,
    DATA_DIR,
)


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = DATA_DIR / "DATA_GEO_Train.csv"
TEST_FILE = DATA_DIR / "DATA_GEO_Test.csv"


# ============================================================
# SETTINGS
# ============================================================

ERROR_COLS = [
    "x_error (m)",
    "y_error (m)",
    "z_error (m)",
    "satclockerror (m)",
]

ERR_NAMES = [
    "X",
    "Y",
    "Z",
    "Clock",
]

HORIZON_BINS = [
    (0, 120),
    (120, 360),
    (360, 720),
    (720, 1200),
    (1200, 1e9),
]

ABS_ERROR_BINS = [
    (0, 5),
    (5, 10),
    (10, 20),
    (20, 1e9),
]


# ============================================================
# LOAD TEST TIMESTAMPS
# Same logic as predict_geo.py
# ============================================================

def load_test_times(path):

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["utc_time"] = pd.to_datetime(
        df["utc_time"]
    )

    return (
        df[["utc_time"]]
        .drop_duplicates()
        .sort_values("utc_time")
        .reset_index(drop=True)
    )


# ============================================================
# MAKE TEST SEQUENCE
# Exact logic from predict_geo.py
# ============================================================

def make_test_sequence(history, target_time):

    MIN_HISTORY = 2
    HISTORY_DAYS = 7

    if len(history) < MIN_HISTORY:
        return None

    history = (
        history
        .sort_values("utc_time")
        .copy()
    )

    target_time = pd.Timestamp(target_time)

    day_minutes = (
        target_time.hour * 60
        + target_time.minute
        + target_time.second / 60.0
    )

    day_fraction = (
        day_minutes / (24.0 * 60.0)
    )

    target_sin = np.sin(
        2.0 * np.pi * day_fraction
    )

    target_cos = np.cos(
        2.0 * np.pi * day_fraction
    )

    history_start = (
        target_time
        - pd.Timedelta(
            days=HISTORY_DAYS
        )
    )

    history = history[
        (history["utc_time"] >= history_start)
        & (history["utc_time"] < target_time)
    ]

    if len(history) < MIN_HISTORY:
        return None

    values = (
        history[ERROR_COLS]
        .astype(float)
        .to_numpy()
    )

    time_to_target = (
        target_time
        - history["utc_time"]
    ).dt.total_seconds().to_numpy() / 60.0

    gaps = (
        history["utc_time"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
        .fillna(0.0)
        .to_numpy()
    )

    target_phase = np.column_stack([
        np.full(
            len(values),
            target_sin,
        ),
        np.full(
            len(values),
            target_cos,
        ),
    ])

    sequence = np.column_stack([
        values,
        gaps,
        time_to_target,
        target_phase,
    ])

    horizon = (
        target_time
        - pd.Timestamp(
            history["utc_time"].iloc[-1]
        )
    ).total_seconds() / 60.0

    if horizon < 120:
        horizon_regime = 0

    elif horizon < 360:
        horizon_regime = 1

    elif horizon < 720:
        horizon_regime = 2

    elif horizon < 1200:
        horizon_regime = 3

    elif horizon < 1800:
        horizon_regime = 4

    else:
        horizon_regime = 5

    return (
        sequence.astype(float),
        float(horizon),
        int(horizon_regime),
    )


# ============================================================
# BUILD TEST EXAMPLES
# ============================================================

def build_test_examples(
    train_df,
    test_times,
):

    sequences = []
    horizons = []
    regimes = []
    target_times = []

    for target_time in test_times:

        result = make_test_sequence(
            train_df,
            target_time,
        )

        if result is None:
            continue

        sequence, horizon, regime = result

        sequences.append(sequence)
        horizons.append(horizon)
        regimes.append(regime)
        target_times.append(target_time)

    return (
        sequences,
        np.asarray(
            horizons,
            dtype=float,
        ),
        np.asarray(
            regimes,
            dtype=np.int64,
        ),
        pd.to_datetime(
            target_times,
        ),
    )


# ============================================================
# LOAD TRAIN
# ============================================================

train_df = load_data(
    TRAIN_FILE
)


# ============================================================
# LOAD REAL DAY-8 TRUTH
# ============================================================

test_truth = pd.read_csv(
    TEST_FILE
)

test_truth.columns = (
    test_truth.columns
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

test_truth["utc_time"] = pd.to_datetime(
    test_truth["utc_time"]
)

for col in ERROR_COLS:

    test_truth[col] = pd.to_numeric(
        test_truth[col]
    )

test_truth = (
    test_truth
    .drop_duplicates()
    .dropna(
        subset=["utc_time"] + ERROR_COLS
    )
    .sort_values("utc_time")
    .reset_index(drop=True)
)

test_times = load_test_times(
    TEST_FILE
)


# ============================================================
# HEADER
# ============================================================

print("=" * 110)
print("REAL GEO DAY-8 CHAMPION ATTRIBUTION")
print("=" * 110)

print(
    f"Device:          {DEVICE}"
)

print(
    f"Training rows:   {len(train_df)}"
)

print(
    f"Test truth rows: {len(test_truth)}"
)


# ============================================================
# BUILD TRAINING EXAMPLES
# ============================================================

(
    train_sequences,
    train_horizons,
    train_horizon_regimes,
    y_train,
) = build_training_examples(
    train_df
)


# ============================================================
# BUILD TEST EXAMPLES
# ============================================================

(
    test_sequences,
    test_horizons,
    test_horizon_regimes,
    predicted_times,
) = build_test_examples(
    train_df,
    test_times["utc_time"].tolist(),
)


if len(test_sequences) == 0:

    raise RuntimeError(
        "No valid Day-8 test sequences were created."
    )


# ============================================================
# EXACT CHAMPION SCALING
# ============================================================

all_train_steps = np.concatenate(
    train_sequences,
    axis=0,
)

x_scaler = StandardScaler()

x_scaler.fit(
    all_train_steps
)

train_sequences = [
    x_scaler.transform(seq)
    for seq in train_sequences
]

test_sequences = [
    x_scaler.transform(seq)
    for seq in test_sequences
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

test_horizons_scaled = (
    horizon_scaler
    .transform(
        test_horizons.reshape(-1, 1)
    )
    .ravel()
)


y_scaler = StandardScaler()

y_scaler.fit(
    y_train
)

y_train_scaled = (
    y_scaler.transform(
        y_train
    )
)


# ============================================================
# PAD
# ============================================================

X_train, train_lengths = (
    pad_sequences(
        train_sequences
    )
)

X_test, test_lengths = (
    pad_sequences(
        test_sequences
    )
)


# ============================================================
# TRAIN EXACT CHAMPION
# ============================================================

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


# ============================================================
# INFERENCE
# ============================================================

with torch.no_grad():

    predictions_scaled = model(
        torch.tensor(
            X_test,
            dtype=torch.float32,
            device=DEVICE,
        ),
        torch.tensor(
            test_lengths,
            dtype=torch.long,
            device=DEVICE,
        ),
        torch.tensor(
            test_horizons_scaled.reshape(
                -1,
                1,
            ),
            dtype=torch.float32,
            device=DEVICE,
        ),
        torch.tensor(
            test_horizon_regimes,
            dtype=torch.long,
            device=DEVICE,
        ),
    ).cpu().numpy()


predictions = (
    y_scaler.inverse_transform(
        predictions_scaled
    )
)


# ============================================================
# PREDICTION DATAFRAME
# ============================================================

pred_df = pd.DataFrame(
    {
        "utc_time": pd.to_datetime(
            predicted_times
        ),

        "horizon": test_horizons,

        "horizon_regime":
            test_horizon_regimes,

        "pred_x": predictions[:, 0],
        "pred_y": predictions[:, 1],
        "pred_z": predictions[:, 2],
        "pred_clock": predictions[:, 3],
    }
)


# ============================================================
# TRUTH DATAFRAME
# ============================================================

truth_df = test_truth[
    [
        "utc_time",
        *ERROR_COLS,
    ]
].copy()

truth_df = truth_df.rename(
    columns={
        ERROR_COLS[0]: "true_x",
        ERROR_COLS[1]: "true_y",
        ERROR_COLS[2]: "true_z",
        ERROR_COLS[3]: "true_clock",
    }
)


# ============================================================
# ALIGN
# ============================================================

merged = pred_df.merge(
    truth_df,
    on="utc_time",
    how="inner",
)

if len(merged) == 0:

    raise RuntimeError(
        "Prediction/truth timestamps did not align."
    )

print(
    f"\nAligned Day-8 points: "
    f"{len(merged)}"
)


# ============================================================
# RESIDUALS
# ============================================================

merged["rx"] = (
    merged["true_x"]
    - merged["pred_x"]
)

merged["ry"] = (
    merged["true_y"]
    - merged["pred_y"]
)

merged["rz"] = (
    merged["true_z"]
    - merged["pred_z"]
)

merged["rc"] = (
    merged["true_clock"]
    - merged["pred_clock"]
)

merged["mean_abs"] = merged[
    [
        "rx",
        "ry",
        "rz",
        "rc",
    ]
].abs().mean(axis=1)

merged["max_abs"] = merged[
    [
        "rx",
        "ry",
        "rz",
        "rc",
    ]
].abs().max(axis=1)


# ============================================================
# 1. OVERALL PERFORMANCE
# ============================================================

print("\n" + "=" * 110)
print("1. OVERALL REAL DAY-8 PERFORMANCE")
print("=" * 110)

overall_W = []

for col, name in zip(
    [
        "rx",
        "ry",
        "rz",
        "rc",
    ],
    ERR_NAMES,
):

    values = merged[col].to_numpy()

    W, p = shapiro(values)

    overall_W.append(W)

    print(
        f"{name:5s}: "
        f"W={W:.6f}  "
        f"p={p:.6g}  "
        f"mean={values.mean():+.4f}  "
        f"std={values.std(ddof=1):.4f}  "
        f"MAE={np.mean(np.abs(values)):.4f}"
    )

print(
    f"Average W = "
    f"{np.mean(overall_W):.6f}"
)


# ============================================================
# 2. PERFORMANCE BY HORIZON
# ============================================================

print("\n" + "=" * 110)
print("2. REAL DAY-8 PERFORMANCE BY FORECAST HORIZON")
print("=" * 110)

for lo, hi in HORIZON_BINS:

    mask = (
        (merged["horizon"] >= lo)
        & (merged["horizon"] < hi)
    )

    group = merged[mask]

    upper = (
        "inf"
        if hi >= 1e9
        else str(hi)
    )

    print("\n" + "-" * 110)

    print(
        f"Horizon: {lo}–{upper} min"
    )

    print(
        f"n = {len(group)}"
    )

    if len(group) < 3:

        print("Too few samples.")
        continue

    Ws = []

    for col, name in zip(
        [
            "rx",
            "ry",
            "rz",
            "rc",
        ],
        ERR_NAMES,
    ):

        values = group[col].to_numpy()

        W, p = shapiro(values)

        Ws.append(W)

        print(
            f"{name:5s}: "
            f"W={W:.6f}  "
            f"mean={values.mean():+.4f}  "
            f"std={values.std(ddof=1):.4f}  "
            f"MAE={np.mean(np.abs(values)):.4f}"
        )

    print(
        f"Average W = "
        f"{np.mean(Ws):.6f}"
    )


# ============================================================
# 3. PERFORMANCE BY RESIDUAL MAGNITUDE
# ============================================================

print("\n" + "=" * 110)
print("3. REAL DAY-8 PERFORMANCE BY RESIDUAL MAGNITUDE")
print("=" * 110)

for lo, hi in ABS_ERROR_BINS:

    mask = (
        (merged["mean_abs"] >= lo)
        & (merged["mean_abs"] < hi)
    )

    group = merged[mask]

    upper = (
        "inf"
        if hi >= 1e9
        else str(hi)
    )

    print("\n" + "-" * 110)

    print(
        f"Mean |residual|: "
        f"{lo}–{upper} m"
    )

    print(
        f"n = {len(group)}"
    )

    if len(group) < 3:

        print("Too few samples.")
        continue

    for col, name in zip(
        [
            "rx",
            "ry",
            "rz",
            "rc",
        ],
        ERR_NAMES,
    ):

        values = group[col].to_numpy()

        W, p = shapiro(values)

        print(
            f"{name:5s}: "
            f"W={W:.6f}  "
            f"mean={values.mean():+.4f}  "
            f"std={values.std(ddof=1):.4f}"
        )


# ============================================================
# 4. LARGE ERROR POINTS
# ============================================================

print("\n" + "=" * 110)
print("4. REAL DAY-8 LARGE-ERROR POINTS")
print("=" * 110)

large = (
    merged[
        merged["max_abs"] >= 20.0
    ]
    .sort_values(
        "max_abs",
        ascending=False,
    )
)

print(
    f">=20 m points: "
    f"{len(large)} / {len(merged)} "
    f"({100.0 * len(large) / len(merged):.2f}%)"
)

if len(large) > 0:

    for _, row in large.iterrows():

        print(
            f"{row['utc_time']}  "
            f"h={row['horizon']:7.1f}m  "
            f"X={row['rx']:+9.2f}  "
            f"Y={row['ry']:+9.2f}  "
            f"Z={row['rz']:+9.2f}  "
            f"C={row['rc']:+9.2f}  "
            f"max={row['max_abs']:+9.2f}"
        )


# ============================================================
# 5. NORMAL VS EXCURSION
#
# Retrospective attribution only.
# These labels are NOT used during training.
# ============================================================

merged["excursion"] = (
    merged["max_abs"] >= 20.0
)

print("\n" + "=" * 110)
print("5. NORMAL VS EXCURSION REGION")
print("=" * 110)

for state_name, mask in [
    (
        "NORMAL",
        ~merged["excursion"],
    ),
    (
        "EXCURSION",
        merged["excursion"],
    ),
]:

    group = merged[mask]

    print("\n" + "-" * 110)

    print(
        f"{state_name}: "
        f"n={len(group)}"
    )

    if len(group) < 3:

        print("Too few samples.")
        continue

    Ws = []

    for col, name in zip(
        [
            "rx",
            "ry",
            "rz",
            "rc",
        ],
        ERR_NAMES,
    ):

        values = group[col].to_numpy()

        W, p = shapiro(values)

        Ws.append(W)

        print(
            f"{name:5s}: "
            f"W={W:.6f}  "
            f"mean={values.mean():+.4f}  "
            f"std={values.std(ddof=1):.4f}  "
            f"MAE={np.mean(np.abs(values)):.4f}"
        )

    print(
        f"Average W = "
        f"{np.mean(Ws):.6f}"
    )


# ============================================================
# 6. LARGE-ERROR TIMING
# ============================================================

print("\n" + "=" * 110)
print("6. LARGE-ERROR TIMING")
print("=" * 110)

if len(large) > 0:

    print(
        f"First large-error point: "
        f"{large['utc_time'].min()}"
    )

    print(
        f"Last large-error point:  "
        f"{large['utc_time'].max()}"
    )

    print(
        f"Median horizon of large-error points: "
        f"{large['horizon'].median():.1f} min"
    )

    print(
        f"Mean horizon of large-error points: "
        f"{large['horizon'].mean():.1f} min"
    )

else:

    print("No residual >=20 m.")


# ============================================================
# 7. WORST 20
# ============================================================

print("\n" + "=" * 110)
print("7. WORST 20 DAY-8 RESIDUALS")
print("=" * 110)

worst = (
    merged
    .sort_values(
        "max_abs",
        ascending=False,
    )
    .head(20)
)

for _, row in worst.iterrows():

    print(
        f"{row['utc_time']}  "
        f"h={row['horizon']:7.1f}m  "
        f"X={row['rx']:+9.2f}  "
        f"Y={row['ry']:+9.2f}  "
        f"Z={row['rz']:+9.2f}  "
        f"C={row['rc']:+9.2f}"
    )


# ============================================================
# 8. HORIZON x EXCURSION
# ============================================================

print("\n" + "=" * 110)
print("8. HORIZON x EXCURSION")
print("=" * 110)

for lo, hi in HORIZON_BINS:

    mask = (
        (merged["horizon"] >= lo)
        & (merged["horizon"] < hi)
    )

    group = merged[mask]

    if len(group) == 0:
        continue

    n_exc = int(
        group["excursion"].sum()
    )

    pct = (
        100.0
        * n_exc
        / len(group)
    )

    upper = (
        "inf"
        if hi >= 1e9
        else str(hi)
    )

    print(
        f"{lo:4d}-{upper:>4s} min: "
        f"n={len(group):3d}  "
        f"excursion={n_exc:3d}  "
        f"fraction={pct:6.2f}%"
    )


print("\n" + "=" * 110)
print("END")
print("=" * 110)