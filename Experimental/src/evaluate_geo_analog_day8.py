import numpy as np
import pandas as pd
from scipy.stats import shapiro

from model_gru_horizon_regime import DATA_DIR, ERROR_COLS


TRAIN_FILE = DATA_DIR / "DATA_GEO_Train.csv"
TEST_FILE = DATA_DIR / "DATA_GEO_Test.csv"
PRED_FILE = (
    DATA_DIR.parent
    / "results"
    / "predictions"
    / "GEO_champion_predictions.csv"
)

TOLERANCE_MIN = 5
GATE_MIN = 960
LAMBDA = 0.05


def same_time_analog(train_df, target_time):
    historical = train_df[
        train_df["utc_time"] < target_time
    ].copy()

    if historical.empty:
        return None

    target_sec = (
        target_time.hour * 3600
        + target_time.minute * 60
        + target_time.second
    )

    hist_sec = (
        historical["utc_time"].dt.hour * 3600
        + historical["utc_time"].dt.minute * 60
        + historical["utc_time"].dt.second
    )

    diff = np.abs(
        hist_sec.to_numpy(dtype=float)
        - target_sec
    )

    diff = np.minimum(
        diff,
        86400.0 - diff,
    )

    eligible = diff <= TOLERANCE_MIN * 60

    if not np.any(eligible):
        return None

    candidates = historical.loc[eligible].copy()
    candidate_diff = diff[eligible]

    # Closest UTC time-of-day; latest date for ties.
    order = np.lexsort(
        (
            -candidates["utc_time"]
            .astype("int64")
            .to_numpy(),
            candidate_diff,
        )
    )

    return candidates.iloc[order[0]]


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
pred = pd.read_csv(PRED_FILE)

for df in [train, test, pred]:
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

train["utc_time"] = pd.to_datetime(train["utc_time"])
test["utc_time"] = pd.to_datetime(test["utc_time"])
pred["utc_time"] = pd.to_datetime(pred["utc_time"])

for col in ERROR_COLS:
    test[col] = pd.to_numeric(test[col])

# ------------------------------------------------------------
# ALIGN TRUTH + CHAMPION
# ------------------------------------------------------------

df = pred.merge(
    test[
        ["utc_time"] + ERROR_COLS
    ].drop_duplicates("utc_time"),
    on="utc_time",
    how="inner",
)

if len(df) == 0:
    raise RuntimeError(
        "No prediction/truth timestamps aligned."
    )

# ------------------------------------------------------------
# BUILD ANALOG + BLEND
# ------------------------------------------------------------

analog = []
matched = []

for t in df["utc_time"]:

    row = same_time_analog(
        train,
        t,
    )

    if row is None:
        analog.append(
            [np.nan] * 4
        )
        matched.append(False)
    else:
        analog.append(
            row[ERROR_COLS]
            .to_numpy(dtype=float)
        )
        matched.append(True)

analog = np.asarray(analog)
matched = np.asarray(matched)

champion = df[
    [
        "pred_x_error (m)",
        "pred_y_error (m)",
        "pred_z_error (m)",
        "pred_satclockerror (m)",
    ]
].to_numpy(dtype=float)

truth = df[
    ERROR_COLS
].to_numpy(dtype=float)

horizon = df[
    "horizon_min"
].to_numpy(dtype=float)

blend = champion.copy()

eligible = (
    matched
    & (horizon >= GATE_MIN)
)

blend[eligible] = (
    (1.0 - LAMBDA)
    * champion[eligible]
    + LAMBDA
    * analog[eligible]
)

# ------------------------------------------------------------
# SCORE
# ------------------------------------------------------------

champion_resid = truth - champion
blend_resid = truth - blend

print("=" * 105)
print("REAL GEO DAY-8 — CHAMPION vs WINNING ANALOG BLEND")
print("=" * 105)

print(
    f"Analog tolerance: +/- {TOLERANCE_MIN} min"
)
print(
    f"Gate: horizon >= {GATE_MIN} min"
)
print(
    f"Lambda: {LAMBDA}"
)
print(
    f"Analog coverage: "
    f"{matched.sum()}/{len(matched)} "
    f"({100 * matched.mean():.2f}%)"
)
print(
    f"Blended points: "
    f"{eligible.sum()}/{len(eligible)} "
    f"({100 * eligible.mean():.2f}%)"
)


def report(name, residuals):
    Ws = []

    print("\n" + "-" * 105)
    print(name)
    print("-" * 105)

    for j, label in enumerate(
        ["X", "Y", "Z", "Clock"]
    ):
        W, p = shapiro(
            residuals[:, j]
        )
        Ws.append(W)

        values = residuals[:, j]

        print(
            f"{label:5s} "
            f"W={W:.6f}  "
            f"p={p:.6g}  "
            f"mean={values.mean():+.4f}  "
            f"std={values.std(ddof=1):.4f}  "
            f"MAE={np.mean(np.abs(values)):.4f}"
        )

    print(
        f"Average W = {np.mean(Ws):.6f}"
    )

    return np.mean(Ws)


champion_W = report(
    "CHAMPION",
    champion_resid,
)

blend_W = report(
    "ANALOG BLEND",
    blend_resid,
)

print("\n" + "=" * 105)
print("RESULT")
print("=" * 105)

print(
    f"Champion Average W : {champion_W:.6f}"
)

print(
    f"Blend Average W    : {blend_W:.6f}"
)

print(
    f"Delta               : "
    f"{blend_W - champion_W:+.6f}"
)

if blend_W > champion_W:
    print(
        "\nWIN: analog blend improves real Day-8 W."
    )
else:
    print(
        "\nNO WIN: analog blend does not improve real Day-8 W."
    )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

df["analog_x"] = analog[:, 0]
df["analog_y"] = analog[:, 1]
df["analog_z"] = analog[:, 2]
df["analog_clock"] = analog[:, 3]
df["analog_matched"] = matched
df["blend_used"] = eligible

df["final_x"] = blend[:, 0]
df["final_y"] = blend[:, 1]
df["final_z"] = blend[:, 2]
df["final_clock"] = blend[:, 3]

df.to_csv(
    DATA_DIR.parent
    / "results"
    / "predictions"
    / "GEO_day8_analog_blend.csv",
    index=False,
)

print(
    "\nSaved:"
)
print(
    "results/predictions/GEO_day8_analog_blend.csv"
)
