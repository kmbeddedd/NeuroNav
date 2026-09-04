import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.preprocessing import StandardScaler

from evaluation import evaluate_all_targets, ERROR_COLS


DATA_DIR = Path("data")

DATASETS = {
    "GEO": "DATA_GEO_Train.csv",
    "MEO": "DATA_MEO_Train.csv",
    "MEO2": "DATA_MEO_Train2.csv",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
HISTORY_DAYS = 7
MIN_HISTORY = 2
EPOCHS = 300

FAST_HALFLIFE_MIN = 60.0

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Causal slow / fast decomposition
# ============================================================

def causal_decompose(df):
    """
    Causal exponential decomposition.

    slow[t] uses only observations <= t.
    fast[t] = raw[t] - slow[t].

    The decay is based on actual elapsed time, not row index.
    """

    df = (
        df.sort_values("utc_time")
        .reset_index(drop=True)
        .copy()
    )

    values = (
        df[ERROR_COLS]
        .astype(float)
        .to_numpy()
    )

    slow = np.zeros_like(values)

    slow[0] = values[0]

    ln2 = np.log(2.0)

    for i in range(1, len(values)):

        dt_minutes = (
            df.loc[i, "utc_time"]
            - df.loc[i - 1, "utc_time"]
        ).total_seconds() / 60.0

        dt_minutes = max(
            float(dt_minutes),
            0.0,
        )

        alpha = 1.0 - np.exp(
            -ln2 * dt_minutes / FAST_HALFLIFE_MIN
        )

        slow[i] = (
            alpha * values[i]
            + (1.0 - alpha) * slow[i - 1]
        )

    fast = values - slow

    df["_slow_x"] = slow[:, 0]
    df["_slow_y"] = slow[:, 1]
    df["_slow_z"] = slow[:, 2]
    df["_slow_clock"] = slow[:, 3]

    df["_fast_x"] = fast[:, 0]
    df["_fast_y"] = fast[:, 1]
    df["_fast_z"] = fast[:, 2]
    df["_fast_clock"] = fast[:, 3]

    return df


def target_decomposition(train_df, row):
    """
    Compute the causal decomposition of one validation target.

    Only train_df + the current target row are used.
    No later validation observation is included.
    """

    history = train_df[
        ["utc_time"] + ERROR_COLS
    ].copy()

    target = pd.DataFrame([
        row[
            ["utc_time"] + ERROR_COLS
        ].to_dict()
    ])

    combined = pd.concat(
        [history, target],
        ignore_index=True,
    )

    combined["utc_time"] = pd.to_datetime(
        combined["utc_time"]
    )

    combined = causal_decompose(
        combined
    )

    last = combined.iloc[-1]

    slow_target = np.array([
        last["_slow_x"],
        last["_slow_y"],
        last["_slow_z"],
        last["_slow_clock"],
    ], dtype=float)

    fast_target = (
        row[ERROR_COLS]
        .astype(float)
        .to_numpy()
        - slow_target
    )

    return slow_target, fast_target


# ============================================================
# Model
# ============================================================

class TwoTimeScaleGRU(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=32,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.horizon_embedding = nn.Embedding(
            num_embeddings=6,
            embedding_dim=4,
        )

        fusion_size = (
            hidden_size
            + 2
            + 4
        )

        self.shared = nn.Sequential(
            nn.Linear(
                fusion_size,
                32,
            ),
            nn.ReLU(),
        )

        self.slow_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

        self.fast_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(
        self,
        x,
        lengths,
        horizon,
        horizon_regime,
    ):

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.gru(
            packed
        )

        history_repr = hidden[-1]

        regime_repr = (
            self.horizon_embedding(
                horizon_regime
            )
        )

        horizon_norm = (
            horizon / 1440.0
        )

        combined = torch.cat(
            [
                history_repr,
                horizon,
                horizon_norm,
                regime_repr,
            ],
            dim=1,
        )

        shared = self.shared(
            combined
        )

        slow = self.slow_head(
            shared
        )

        fast = self.fast_head(
            shared
        )

        return slow, fast


# ============================================================
# Data loading
# ============================================================

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


# ============================================================
# Feature construction
# ============================================================

def horizon_regime_from_value(horizon):

    if horizon < 120:
        return 0
    elif horizon < 360:
        return 1
    elif horizon < 720:
        return 2
    elif horizon < 1200:
        return 3
    elif horizon < 1800:
        return 4
    else:
        return 5


def make_sequence(
    history,
    target_time,
):

    if len(history) < MIN_HISTORY:
        return None

    history = (
        history
        .sort_values("utc_time")
        .copy()
    )

    target_time = pd.Timestamp(
        target_time
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
    ].copy()

    if len(history) < MIN_HISTORY:
        return None

    # --------------------------------------------------------
    # Causal fast history.
    # Every fast value was computed using data <= its own
    # timestamp.
    # --------------------------------------------------------

    history = causal_decompose(
        history
    )

    raw_values = (
        history[ERROR_COLS]
        .astype(float)
        .to_numpy()
    )

    fast_values = np.column_stack([
        history["_fast_x"].to_numpy(),
        history["_fast_y"].to_numpy(),
        history["_fast_z"].to_numpy(),
        history["_fast_clock"].to_numpy(),
    ])

    gaps = (
        history["utc_time"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
        .fillna(0.0)
        .to_numpy()
    )

    time_to_target = (
        target_time
        - history["utc_time"]
    ).dt.total_seconds().to_numpy() / 60.0

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

    target_phase = np.column_stack([
        np.full(
            len(history),
            target_sin,
        ),
        np.full(
            len(history),
            target_cos,
        ),
    ])

    sequence = np.column_stack([
        raw_values,
        fast_values,
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

    horizon_regime = (
        horizon_regime_from_value(
            horizon
        )
    )

    return (
        sequence.astype(float),
        float(horizon),
        int(horizon_regime),
    )


# ============================================================
# Training examples
# ============================================================

def build_training_examples(
    df
):

    df = causal_decompose(
        df
    )

    sequences = []
    horizons = []
    horizon_regimes = []

    slow_targets = []
    fast_targets = []

    df = (
        df.sort_values("utc_time")
        .reset_index(drop=True)
    )

    for i in range(
        MIN_HISTORY,
        len(df)
    ):

        target_time = df.loc[
            i,
            "utc_time"
        ]

        result = make_sequence(
            df.iloc[:i],
            target_time,
        )

        if result is None:
            continue

        sequence, horizon, regime = result

        sequences.append(
            sequence
        )

        horizons.append(
            horizon
        )

        horizon_regimes.append(
            regime
        )

        slow_targets.append(
            np.array([
                df.loc[i, "_slow_x"],
                df.loc[i, "_slow_y"],
                df.loc[i, "_slow_z"],
                df.loc[i, "_slow_clock"],
            ], dtype=float)
        )

        fast_targets.append(
            np.array([
                df.loc[i, "_fast_x"],
                df.loc[i, "_fast_y"],
                df.loc[i, "_fast_z"],
                df.loc[i, "_fast_clock"],
            ], dtype=float)
        )

    return (
        sequences,
        np.asarray(
            horizons,
            dtype=float,
        ),
        np.asarray(
            horizon_regimes,
            dtype=np.int64,
        ),
        np.asarray(
            slow_targets,
            dtype=float,
        ),
        np.asarray(
            fast_targets,
            dtype=float,
        ),
    )


# ============================================================
# Validation examples
# ============================================================

def build_validation_examples(
    train_df,
    validation_df,
):

    sequences = []
    horizons = []
    horizon_regimes = []

    slow_targets = []
    fast_targets = []
    raw_targets = []

    train_df = (
        train_df
        .sort_values("utc_time")
        .copy()
    )

    for _, row in (
        validation_df
        .sort_values("utc_time")
        .iterrows()
    ):

        result = make_sequence(
            train_df,
            row["utc_time"],
        )

        if result is None:
            continue

        sequence, horizon, regime = result

        sequences.append(
            sequence
        )

        horizons.append(
            horizon
        )

        horizon_regimes.append(
            regime
        )

        slow_target, fast_target = (
            target_decomposition(
                train_df,
                row,
            )
        )

        slow_targets.append(
            slow_target
        )

        fast_targets.append(
            fast_target
        )

        raw_targets.append(
            row[
                ERROR_COLS
            ]
            .astype(float)
            .to_numpy()
        )

    return (
        sequences,
        np.asarray(
            horizons,
            dtype=float,
        ),
        np.asarray(
            horizon_regimes,
            dtype=np.int64,
        ),
        np.asarray(
            slow_targets,
            dtype=float,
        ),
        np.asarray(
            fast_targets,
            dtype=float,
        ),
        np.asarray(
            raw_targets,
            dtype=float,
        ),
    )


# ============================================================
# Sequence padding
# ============================================================

def pad_sequences(
    sequences
):

    max_len = max(
        len(seq)
        for seq in sequences
    )

    n = len(sequences)
    feature_dim = sequences[0].shape[1]

    X = np.zeros(
        (
            n,
            max_len,
            feature_dim,
        ),
        dtype=np.float32,
    )

    lengths = np.zeros(
        n,
        dtype=np.int64,
    )

    for i, seq in enumerate(
        sequences
    ):

        length = len(seq)

        X[i, :length] = seq

        lengths[i] = length

    return X, lengths


# ============================================================
# Horizon weighting
# ============================================================

def horizon_weights(
    horizons
):

    weights = np.ones_like(
        horizons,
        dtype=np.float32,
    )

    weights[
        (horizons >= 30)
        & (horizons < 120)
    ] = 1.25

    weights[
        (horizons >= 120)
        & (horizons < 360)
    ] = 1.50

    weights[
        (horizons >= 360)
        & (horizons < 720)
    ] = 2.00

    weights[
        (horizons >= 720)
        & (horizons < 1200)
    ] = 2.50

    weights[
        horizons >= 1200
    ] = 3.00

    return weights


# ============================================================
# Training
# ============================================================

def train_model(
    X,
    lengths,
    horizons_scaled,
    horizons_raw,
    horizon_regimes,
    slow_y,
    fast_y,
):

    model = TwoTimeScaleGRU(
        input_size=X.shape[2]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE,
    )

    lengths_tensor = torch.tensor(
        lengths,
        dtype=torch.long,
        device=DEVICE,
    )

    horizon_tensor = torch.tensor(
        horizons_scaled.reshape(-1, 1),
        dtype=torch.float32,
        device=DEVICE,
    )

    regime_tensor = torch.tensor(
        horizon_regimes,
        dtype=torch.long,
        device=DEVICE,
    )

    slow_tensor = torch.tensor(
        slow_y,
        dtype=torch.float32,
        device=DEVICE,
    )

    fast_tensor = torch.tensor(
        fast_y,
        dtype=torch.float32,
        device=DEVICE,
    )

    weights = torch.tensor(
        horizon_weights(
            horizons_raw
        ),
        dtype=torch.float32,
        device=DEVICE,
    ).reshape(-1, 1)

    model.train()

    for _ in range(EPOCHS):

        optimizer.zero_grad()

        slow_pred, fast_pred = model(
            X_tensor,
            lengths_tensor,
            horizon_tensor,
            regime_tensor,
        )

        slow_loss = (
            (slow_pred - slow_tensor) ** 2
            * weights
        ).mean()

        fast_loss = (
            (fast_pred - fast_tensor) ** 2
            * weights
        ).mean()

        loss = (
            slow_loss
            + fast_loss
        )

        loss.backward()

        optimizer.step()

    return model


# ============================================================
# Main rolling validation
# ============================================================

def main():

    print(
        "\nCAUSAL TWO-TIMESCALE GRU"
    )
    print("=" * 90)
    print(
        f"Device: {DEVICE}"
    )
    print(
        f"Fast halflife: "
        f"{FAST_HALFLIFE_MIN:.1f} min"
    )

    fold_results = []

    for dataset_name, filename in (
        DATASETS.items()
    ):

        df = load_data(
            DATA_DIR / filename
        )

        days = sorted(
            df["utc_time"]
            .dt.normalize()
            .unique()
        )

        print(
            f"\n{dataset_name}"
        )
        print(
            "-" * 90
        )

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
                train_regimes,
                slow_train,
                fast_train,
            ) = build_training_examples(
                train_df
            )

            (
                val_sequences,
                val_horizons,
                val_regimes,
                slow_val,
                fast_val,
                y_val,
            ) = build_validation_examples(
                train_df,
                validation_df,
            )

            if len(train_sequences) < 5:
                continue

            if len(val_sequences) == 0:
                continue

            # ------------------------------------------------
            # Sequence scaling
            # ------------------------------------------------

            all_train_steps = (
                np.concatenate(
                    train_sequences,
                    axis=0,
                )
            )

            x_scaler = StandardScaler()

            x_scaler.fit(
                all_train_steps
            )

            train_sequences = [
                x_scaler.transform(seq)
                for seq in train_sequences
            ]

            val_sequences = [
                x_scaler.transform(seq)
                for seq in val_sequences
            ]

            # ------------------------------------------------
            # Horizon scaling
            # ------------------------------------------------

            horizon_scaler = StandardScaler()

            horizon_scaler.fit(
                train_horizons.reshape(-1, 1)
            )

            train_horizons_scaled = (
                horizon_scaler.transform(
                    train_horizons.reshape(-1, 1)
                )
                .ravel()
            )

            val_horizons_scaled = (
                horizon_scaler.transform(
                    val_horizons.reshape(-1, 1)
                )
                .ravel()
            )

            # ------------------------------------------------
            # Separate target scaling.
            # ------------------------------------------------

            slow_scaler = StandardScaler()
            fast_scaler = StandardScaler()

            slow_scaler.fit(
                slow_train
            )

            fast_scaler.fit(
                fast_train
            )

            slow_train_scaled = (
                slow_scaler.transform(
                    slow_train
                )
            )

            fast_train_scaled = (
                fast_scaler.transform(
                    fast_train
                )
            )

            # ------------------------------------------------
            # Padding
            # ------------------------------------------------

            X_train, train_lengths = (
                pad_sequences(
                    train_sequences
                )
            )

            X_val, val_lengths = (
                pad_sequences(
                    val_sequences
                )
            )

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            model = train_model(
                X_train,
                train_lengths,
                train_horizons_scaled,
                train_horizons,
                train_regimes,
                slow_train_scaled,
                fast_train_scaled,
            )

            model.eval()

            with torch.no_grad():

                slow_pred_scaled, fast_pred_scaled = (
                    model(
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
                            val_horizons_scaled.reshape(
                                -1,
                                1,
                            ),
                            dtype=torch.float32,
                            device=DEVICE,
                        ),
                        torch.tensor(
                            val_regimes,
                            dtype=torch.long,
                            device=DEVICE,
                        ),
                    )
                )

            slow_pred = (
                slow_scaler.inverse_transform(
                    slow_pred_scaled.cpu().numpy()
                )
            )

            fast_pred = (
                fast_scaler.inverse_transform(
                    fast_pred_scaled.cpu().numpy()
                )
            )

            predictions = (
                slow_pred
                + fast_pred
            )

            residuals = (
                y_val
                - predictions
            )

            results = evaluate_all_targets(
                residuals
            )

            print(
                f"\nValidation day: "
                f"{validation_day.date()}"
            )

            print(
                f"Train examples: "
                f"{len(train_sequences)}"
            )

            print(
                f"Validation examples: "
                f"{len(val_sequences)}"
            )

            print(
                f"Validation horizon: "
                f"{val_horizons.min():.1f} → "
                f"{val_horizons.max():.1f} min"
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
        exist_ok=True,
    )

    result_df = pd.DataFrame(
        fold_results
    )

    result_df.to_csv(
        output_dir
        / "two_timescale_gru_fold_scores.csv",
        index=False,
    )

    print("\n")
    print("=" * 90)
    print(
        "CAUSAL TWO-TIMESCALE GRU OVERALL"
    )
    print("=" * 90)

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
        "two_timescale_gru_fold_scores.csv"
    )


if __name__ == "__main__":
    main()
