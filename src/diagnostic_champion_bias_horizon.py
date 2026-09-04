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
    DATASETS,
)


# ============================================================
# SETTINGS
# ============================================================

DATASET = "GEO"

HORIZON_BINS = [
    (0, 120),
    (120, 360),
    (360, 720),
    (720, 1200),
    (1200, 1e9),
]

FINE_HORIZON_BINS = [
    (0, 120),
    (120, 240),
    (240, 360),
    (360, 480),
    (480, 600),
    (600, 720),
    (720, 840),
    (840, 960),
    (960, 1080),
    (1080, 1200),
    (1200, 1320),
    (1320, 1440),
    (1440, 1e9),
]

ERR_NAMES = [
    "X",
    "Y",
    "Z",
    "Clock",
]


# ============================================================
# LOAD DATA
# ============================================================

df = load_data(
    DATA_DIR / DATASETS[DATASET]
)

days = sorted(
    df["utc_time"]
    .dt.normalize()
    .unique()
)

print("=" * 100)
print("CHAMPION BIAS VS FORECAST HORIZON")
print("=" * 100)
print(f"Dataset: {DATASET}")
print(f"Device:  {DEVICE}")
print(f"Observations: {len(df)}")


# ============================================================
# STORAGE
# ============================================================

records = []


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

for validation_day in days[1:]:

    validation_df = df[
        df["utc_time"].dt.normalize()
        == validation_day
    ].copy()

    if len(validation_df) < 5:
        continue

    train_df = df[
        df["utc_time"] < validation_day
    ].copy()

    (
        train_sequences,
        train_horizons,
        train_horizon_regimes,
        y_train,
    ) = build_training_examples(
        train_df
    )

    (
        val_sequences,
        val_horizons,
        val_horizon_regimes,
        y_val,
    ) = build_validation_examples(
        train_df,
        validation_df,
    )

    if len(train_sequences) < 5:
        continue

    if len(val_sequences) == 0:
        continue

    # --------------------------------------------------------
    # Exact champion preprocessing
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

    val_sequences = [
        x_scaler.transform(seq)
        for seq in val_sequences
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

    val_horizons_scaled = (
        horizon_scaler
        .transform(
            val_horizons.reshape(-1, 1)
        )
        .ravel()
    )

    y_scaler = StandardScaler()

    y_scaler.fit(y_train)

    y_train_scaled = (
        y_scaler.transform(y_train)
    )

    X_train, train_lengths = (
        pad_sequences(train_sequences)
    )

    X_val, val_lengths = (
        pad_sequences(val_sequences)
    )

    # --------------------------------------------------------
    # Reproduce exact champion training
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

    with torch.no_grad():

        predictions_scaled = model(
            torch.tensor(
                X_val,
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.tensor(
                val_lengths,
                dtype=torch.long,
                device=DEVICE,
            ),
            torch.tensor(
                val_horizons_scaled.reshape(-1, 1),
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.tensor(
                val_horizon_regimes,
                dtype=torch.long,
                device=DEVICE,
            ),
        ).cpu().numpy()

    predictions = (
        y_scaler.inverse_transform(
            predictions_scaled
        )
    )

    residuals = y_val - predictions

    # --------------------------------------------------------
    # Validation timestamps
    # --------------------------------------------------------

    validation_sorted = (
        validation_df
        .sort_values("utc_time")
        .reset_index(drop=True)
    )

    n = min(
        len(validation_sorted),
        len(val_horizons),
        len(residuals),
    )

    for i in range(n):

        records.append(
            {
                "day": validation_day.date(),

                "time": validation_sorted.loc[
                    i,
                    "utc_time",
                ],

                "horizon": float(
                    val_horizons[i]
                ),

                "rx": float(
                    residuals[i, 0]
                ),

                "ry": float(
                    residuals[i, 1]
                ),

                "rz": float(
                    residuals[i, 2]
                ),

                "rc": float(
                    residuals[i, 3]
                ),
            }
        )


records = pd.DataFrame(records)

print(
    f"\nValidation points collected: "
    f"{len(records)}"
)


# ============================================================
# SAFETY CHECK
# ============================================================

if records.empty:

    raise RuntimeError(
        "No validation records were collected."
    )


# ============================================================
# ADD DERIVED VALUES
# ============================================================

records["mean_abs"] = records[
    ["rx", "ry", "rz", "rc"]
].abs().mean(axis=1)

records["max_abs"] = records[
    ["rx", "ry", "rz", "rc"]
].abs().max(axis=1)


# ============================================================
# 1. SHAPIRO BY HORIZON
# ============================================================

print("\n" + "=" * 100)
print("1. SHAPIRO-WILK BY FORECAST HORIZON")
print("=" * 100)

for lo, hi in HORIZON_BINS:

    mask = (
        (records["horizon"] >= lo)
        & (records["horizon"] < hi)
    )

    subset = records[mask]

    upper = (
        "inf"
        if hi >= 1e9
        else str(hi)
    )

    print("\n" + "-" * 100)
    print(
        f"Horizon: {lo}–{upper} min"
    )
    print(
        f"n = {len(subset)}"
    )

    if len(subset) == 0:

        print("No samples.")
        continue

    W_values = []

    for col, name in zip(
        ["rx", "ry", "rz", "rc"],
        ERR_NAMES,
    ):

        values = subset[col].to_numpy()

        if len(values) >= 3:

            W, p = shapiro(values)

        else:

            W = np.nan
            p = np.nan

        W_values.append(W)

        mean_value = np.mean(values)

        if len(values) >= 2:
            std_value = np.std(
                values,
                ddof=1,
            )
        else:
            std_value = np.nan

        print(
            f"{name:5s}: "
            f"W={W:.6f}  "
            f"p={p:.6g}  "
            f"mean={mean_value:+.4f}  "
            f"std={std_value:.4f}"
        )

    if np.isfinite(W_values).any():

        print(
            f"Average W = "
            f"{np.nanmean(W_values):.6f}"
        )


# ============================================================
# 2. MEAN BIAS VS HORIZON
# ============================================================

print("\n" + "=" * 100)
print("2. MEAN BIAS VS HORIZON")
print("=" * 100)

print(
    f"{'Range (min)':>18} "
    f"{'n':>5} "
    f"{'X mean':>12} "
    f"{'Y mean':>12} "
    f"{'Z mean':>12} "
    f"{'Clock mean':>12}"
)

for lo, hi in FINE_HORIZON_BINS:

    mask = (
        (records["horizon"] >= lo)
        & (records["horizon"] < hi)
    )

    subset = records[mask]

    if len(subset) == 0:
        continue

    if hi >= 1e9:
        label = f"{lo}-inf"
    else:
        label = f"{lo}-{hi}"

    print(
        f"{label:>18} "
        f"{len(subset):5d} "
        f"{subset['rx'].mean():+12.3f} "
        f"{subset['ry'].mean():+12.3f} "
        f"{subset['rz'].mean():+12.3f} "
        f"{subset['rc'].mean():+12.3f}"
    )


# ============================================================
# 3. MEDIAN BIAS VS HORIZON
# ============================================================

print("\n" + "=" * 100)
print("3. MEDIAN BIAS VS HORIZON")
print("=" * 100)

print(
    f"{'Range (min)':>18} "
    f"{'n':>5} "
    f"{'X median':>12} "
    f"{'Y median':>12} "
    f"{'Z median':>12} "
    f"{'Clock median':>12}"
)

for lo, hi in FINE_HORIZON_BINS:

    mask = (
        (records["horizon"] >= lo)
        & (records["horizon"] < hi)
    )

    subset = records[mask]

    if len(subset) == 0:
        continue

    if hi >= 1e9:
        label = f"{lo}-inf"
    else:
        label = f"{lo}-{hi}"

    print(
        f"{label:>18} "
        f"{len(subset):5d} "
        f"{subset['rx'].median():+12.3f} "
        f"{subset['ry'].median():+12.3f} "
        f"{subset['rz'].median():+12.3f} "
        f"{subset['rc'].median():+12.3f}"
    )


# ============================================================
# 4. LINEAR BIAS TREND
# ============================================================

print("\n" + "=" * 100)
print("4. LINEAR BIAS TREND")
print("=" * 100)

print(
    "Model: residual ~= intercept + slope * horizon"
)

for col, name in zip(
    ["rx", "ry", "rz", "rc"],
    ERR_NAMES,
):

    x = records["horizon"].to_numpy()
    y = records[col].to_numpy()

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        continue

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    if (
        np.std(x) > 0
        and np.std(y) > 0
    ):

        corr = np.corrcoef(
            x,
            y,
        )[0, 1]

    else:

        corr = np.nan

    bias_1440 = (
        intercept
        + slope * 1440.0
    )

    print(
        f"{name:5s}: "
        f"intercept={intercept:+.6f}  "
        f"slope={slope:+.6f} m/min  "
        f"bias@1440={bias_1440:+.3f} m  "
        f"corr={corr:+.6f}"
    )


# ============================================================
# 5. DAY-WISE BIAS
# ============================================================

print("\n" + "=" * 100)
print("5. VALIDATION-DAY BIAS")
print("=" * 100)

print(
    f"{'Day':<15} "
    f"{'n':>5} "
    f"{'X':>12} "
    f"{'Y':>12} "
    f"{'Z':>12} "
    f"{'Clock':>12}"
)

for day, group in records.groupby(
    "day",
    sort=True,
):

    print(
        f"{str(day):<15} "
        f"{len(group):5d} "
        f"{group['rx'].mean():+12.3f} "
        f"{group['ry'].mean():+12.3f} "
        f"{group['rz'].mean():+12.3f} "
        f"{group['rc'].mean():+12.3f}"
    )


# ============================================================
# 6. DAY-WISE HORIZON INFORMATION
# ============================================================

print("\n" + "=" * 100)
print("6. DAY-WISE HORIZON RANGE")
print("=" * 100)

print(
    f"{'Day':<15} "
    f"{'n':>5} "
    f"{'min_h':>10} "
    f"{'median_h':>10} "
    f"{'max_h':>10}"
)

for day, group in records.groupby(
    "day",
    sort=True,
):

    print(
        f"{str(day):<15} "
        f"{len(group):5d} "
        f"{group['horizon'].min():10.1f} "
        f"{group['horizon'].median():10.1f} "
        f"{group['horizon'].max():10.1f}"
    )


# ============================================================
# 7. WORST INDIVIDUAL RESIDUALS
# ============================================================

print("\n" + "=" * 100)
print("7. WORST VALIDATION POINTS")
print("=" * 100)

worst = (
    records
    .sort_values(
        "max_abs",
        ascending=False,
    )
    .head(25)
)

for _, row in worst.iterrows():

    print(
        f"{row['time']}  "
        f"h={row['horizon']:7.1f}m  "
        f"X={row['rx']:+10.2f}  "
        f"Y={row['ry']:+10.2f}  "
        f"Z={row['rz']:+10.2f}  "
        f"C={row['rc']:+10.2f}  "
        f"max={row['max_abs']:+10.2f}"
    )


# ============================================================
# 8. >=20 m EVENTS
# ============================================================

print("\n" + "=" * 100)
print("8. >=20 m RESIDUAL EVENTS")
print("=" * 100)

event_mask = (
    records["max_abs"] >= 20.0
)

events = records[event_mask]

event_fraction = (
    len(events) / len(records)
    if len(records) > 0
    else np.nan
)

print(
    f"Points with max absolute residual >=20m: "
    f"{len(events)} / {len(records)} "
    f"({100.0 * event_fraction:.2f}%)"
)

if len(events) > 0:

    print("\nFirst 50:")
    print(
        events[
            [
                "time",
                "horizon",
                "rx",
                "ry",
                "rz",
                "rc",
                "max_abs",
            ]
        ]
        .head(50)
        .to_string(
            index=False
        )
    )


# ============================================================
# 9. OVERALL CORRELATION WITH HORIZON
# ============================================================

print("\n" + "=" * 100)
print("9. HORIZON / RESIDUAL CORRELATION")
print("=" * 100)

for col, name in zip(
    ["rx", "ry", "rz", "rc"],
    ERR_NAMES,
):

    x = records["horizon"].to_numpy()
    y = records[col].to_numpy()

    if (
        np.std(x) > 0
        and np.std(y) > 0
    ):

        corr = np.corrcoef(
            x,
            y,
        )[0, 1]

    else:

        corr = np.nan

    print(
        f"{name:5s}: "
        f"corr(horizon,residual)="
        f"{corr:+.6f}"
    )


# ============================================================
# END
# ============================================================

print("\n" + "=" * 100)
print("END")
print("=" * 100)
