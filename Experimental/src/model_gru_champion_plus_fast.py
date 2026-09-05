import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

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

LAMBDA_VALUES = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.75,
    1.00,
]

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Champion GRU
# ============================================================

class HorizonGRU(nn.Module):

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

        self.head = nn.Sequential(
            nn.Linear(
                hidden_size + 2 + 4,
                32,
            ),
            nn.ReLU(),
            nn.Linear(32, 4),
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

        return self.head(combined)


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
        df[col] = pd.to_numeric(df[col])

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
# Champion sequence
# ============================================================

def horizon_regime_from_value(
    horizon
):

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

    return (
        sequence.astype(float),
        float(horizon),
        horizon_regime_from_value(
            horizon
        ),
    )


def build_training_examples(
    df
):

    sequences = []
    horizons = []
    regimes = []
    targets = []

    df = (
        df
        .sort_values("utc_time")
        .reset_index(drop=True)
    )

    for i in range(
        MIN_HISTORY,
        len(df)
    ):

        result = make_sequence(
            df.iloc[:i],
            df.loc[i, "utc_time"],
        )

        if result is None:
            continue

        sequence, horizon, regime = result

        sequences.append(sequence)
        horizons.append(horizon)
        regimes.append(regime)

        targets.append(
            df.loc[
                i,
                ERROR_COLS,
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
            regimes,
            dtype=np.int64,
        ),
        np.asarray(
            targets,
            dtype=float,
        ),
    )


def build_validation_examples(
    train_df,
    validation_df,
):

    sequences = []
    horizons = []
    regimes = []
    targets = []

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

        sequences.append(sequence)
        horizons.append(horizon)
        regimes.append(regime)

        targets.append(
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
            regimes,
            dtype=np.int64,
        ),
        np.asarray(
            targets,
            dtype=float,
        ),
    )


# ============================================================
# Champion helpers
# ============================================================

def pad_sequences(
    sequences
):

    max_len = max(
        len(seq)
        for seq in sequences
    )

    n = len(sequences)
    feature_dim = (
        sequences[0].shape[1]
    )

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


def train_champion(
    X,
    lengths,
    horizons_scaled,
    horizons_raw,
    regimes,
    y,
):

    model = HorizonGRU(
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
        regimes,
        dtype=torch.long,
        device=DEVICE,
    )

    y_tensor = torch.tensor(
        y,
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

        prediction = model(
            X_tensor,
            lengths_tensor,
            horizon_tensor,
            regime_tensor,
        )

        loss = (
            (prediction - y_tensor) ** 2
            * weights
        ).mean()

        loss.backward()
        optimizer.step()

    return model


# ============================================================
# Causal fast decomposition
# ============================================================

def causal_decompose(
    df
):

    df = (
        df
        .sort_values("utc_time")
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

    for i in range(
        1,
        len(values)
    ):

        dt_minutes = (
            df.loc[i, "utc_time"]
            - df.loc[i - 1, "utc_time"]
        ).total_seconds() / 60.0

        dt_minutes = max(
            float(dt_minutes),
            0.0,
        )

        alpha = (
            1.0
            - np.exp(
                -ln2
                * dt_minutes
                / FAST_HALFLIFE_MIN
            )
        )

        slow[i] = (
            alpha * values[i]
            + (1.0 - alpha)
            * slow[i - 1]
        )

    fast = values - slow

    result = df.copy()

    result["_fast_x"] = fast[:, 0]
    result["_fast_y"] = fast[:, 1]
    result["_fast_z"] = fast[:, 2]
    result["_fast_clock"] = fast[:, 3]

    return result


def fast_feature_vector(
    history
):

    history = (
        history
        .sort_values("utc_time")
        .copy()
    )

    history = causal_decompose(
        history
    )

    fast = history[
        [
            "_fast_x",
            "_fast_y",
            "_fast_z",
            "_fast_clock",
        ]
    ].to_numpy(dtype=float)

    if len(fast) < 3:
        return None

    o0 = fast[-1]
    o1 = fast[-2]
    o2 = fast[-3]

    d0 = o0 - o1
    d1 = o1 - o2

    gap0 = (
        history["utc_time"].iloc[-1]
        - history["utc_time"].iloc[-2]
    ).total_seconds() / 60.0

    gap1 = (
        history["utc_time"].iloc[-2]
        - history["utc_time"].iloc[-3]
    ).total_seconds() / 60.0

    return np.concatenate([
        o0,
        o1,
        o2,
        d0,
        d1,
        [gap0, gap1],
    ])


def build_fast_training_data(
    df
):

    decomposed = causal_decompose(
        df
    )

    X = []
    y = []

    for i in range(
        3,
        len(decomposed)
    ):

        history = decomposed.iloc[:i]

        fast_features = (
            fast_feature_vector(
                history
            )
        )

        if fast_features is None:
            continue

        target = np.array([
            decomposed.loc[
                i,
                "_fast_x",
            ],
            decomposed.loc[
                i,
                "_fast_y",
            ],
            decomposed.loc[
                i,
                "_fast_z",
            ],
            decomposed.loc[
                i,
                "_fast_clock",
            ],
        ], dtype=float)

        X.append(
            fast_features
        )

        y.append(
            target
        )

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=float),
    )


def build_fast_validation_data(
    train_df,
    validation_df,
):

    X = []
    times = []

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

        # Only observations available BEFORE the
        # validation target are used as features.
        history = train_df[
            train_df["utc_time"]
            < row["utc_time"]
        ]

        if len(history) < 3:
            continue

        features = (
            fast_feature_vector(
                history
            )
        )

        if features is None:
            continue

        X.append(features)
        times.append(
            row["utc_time"]
        )

    return (
        np.asarray(
            X,
            dtype=float,
        ),
        pd.to_datetime(times),
    )


# ============================================================
# Main rolling validation
# ============================================================

def main():

    print(
        "\nCHAMPION + CAUSAL FAST RIDGE"
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
                df["utc_time"]
                < validation_day
            ].copy()

            # ------------------------------------------------
            # Champion training examples
            # ------------------------------------------------

            (
                train_sequences,
                train_horizons,
                train_regimes,
                y_train,
            ) = build_training_examples(
                train_df
            )

            (
                val_sequences,
                val_horizons,
                val_regimes,
                y_val,
            ) = build_validation_examples(
                train_df,
                validation_df
            )

            if len(train_sequences) < 5:
                continue

            if len(val_sequences) == 0:
                continue

            # ------------------------------------------------
            # Champion scaling
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

            horizon_scaler = (
                StandardScaler()
            )

            horizon_scaler.fit(
                train_horizons.reshape(
                    -1,
                    1,
                )
            )

            train_horizons_scaled = (
                horizon_scaler.transform(
                    train_horizons.reshape(
                        -1,
                        1,
                    )
                )
                .ravel()
            )

            val_horizons_scaled = (
                horizon_scaler.transform(
                    val_horizons.reshape(
                        -1,
                        1,
                    )
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
            # Train champion
            # ------------------------------------------------

            champion = train_champion(
                X_train,
                train_lengths,
                train_horizons_scaled,
                train_horizons,
                train_regimes,
                y_train_scaled,
            )

            champion.eval()

            with torch.no_grad():

                champion_scaled = (
                    champion(
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
                    .cpu()
                    .numpy()
                )

            champion_prediction = (
                y_scaler.inverse_transform(
                    champion_scaled
                )
            )

            # ------------------------------------------------
            # Fast Ridge
            # ------------------------------------------------

            fast_X_train, fast_y_train = (
                build_fast_training_data(
                    train_df
                )
            )

            fast_X_val, fast_val_times = (
                build_fast_validation_data(
                    train_df,
                    validation_df,
                )
            )

            if len(fast_X_val) == 0:
                continue

            fast_model = Ridge(
                alpha=10.0
            )

            fast_model.fit(
                fast_X_train,
                fast_y_train
            )

            fast_prediction = (
                fast_model.predict(
                    fast_X_val
                )
            )

            # ------------------------------------------------
            # Align champion validation rows with fast rows.
            # ------------------------------------------------

            val_time_series = pd.to_datetime(
                validation_df[
                    "utc_time"
                ]
                .sort_values()
                .to_numpy()
            )

            champion_indices = [
                i
                for i, t in enumerate(
                    val_time_series
                )
                if t in set(
                    fast_val_times
                )
            ]

            if len(champion_indices) != len(
                fast_val_times
            ):
                time_to_index = {
                    t: i
                    for i, t in enumerate(
                        val_time_series
                    )
                }

                champion_indices = [
                    time_to_index[t]
                    for t in fast_val_times
                    if t in time_to_index
                ]

            champion_aligned = (
                champion_prediction[
                    champion_indices
                ]
            )

            actual_aligned = (
                y_val[
                    champion_indices
                ]
            )

            # ------------------------------------------------
            # Lambda sweep
            # ------------------------------------------------

            fold_score = {
                "dataset": dataset_name,
                "validation_day":
                    validation_day.date(),
            }

            best_lambda = None
            best_w = -np.inf

            print(
                f"\nValidation day: "
                f"{validation_day.date()}"
            )

            print(
                f"Validation examples: "
                f"{len(actual_aligned)}"
            )

            print(
                "\nlambda sweep:"
            )

            for lam in LAMBDA_VALUES:

                prediction = (
                    champion_aligned
                    + lam * fast_prediction
                )

                residuals = (
                    actual_aligned
                    - prediction
                )

                results = (
                    evaluate_all_targets(
                        residuals
                    )
                )

                avg_w = (
                    results["average_W"]
                )

                print(
                    f"  lambda={lam:>4.2f} "
                    f"Average W="
                    f"{avg_w:.6f}"
                )

                fold_score[
                    f"W_lambda_{lam:.2f}"
                ] = avg_w

                if avg_w > best_w:

                    best_w = avg_w
                    best_lambda = lam

            fold_score[
                "best_lambda"
            ] = best_lambda

            fold_score[
                "best_W"
            ] = best_w

            # Champion W for direct comparison.
            champion_residuals = (
                actual_aligned
                - champion_aligned
            )

            champion_results = (
                evaluate_all_targets(
                    champion_residuals
                )
            )

            fold_score[
                "champion_W"
            ] = champion_results[
                "average_W"
            ]

            print(
                f"\nChampion W = "
                f"{fold_score['champion_W']:.6f}"
            )

            print(
                f"Best lambda = "
                f"{best_lambda:.2f}"
            )

            print(
                f"Best W = "
                f"{best_w:.6f}"
            )

            fold_results.append(
                fold_score
            )

    # ========================================================
    # Save results
    # ========================================================

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
        / "champion_plus_fast_fold_scores.csv",
        index=False,
    )

    print("\n")
    print("=" * 90)
    print(
        "CHAMPION + FAST RIDGE OVERALL"
    )
    print("=" * 90)

    champion_mean = (
        result_df[
            "champion_W"
        ].mean()
    )

    best_mean = (
        result_df[
            "best_W"
        ].mean()
    )

    print(
        f"Champion mean W: "
        f"{champion_mean:.6f}"
    )

    print(
        f"Best-lambda mean W: "
        f"{best_mean:.6f}"
    )

    print(
        f"Delta W: "
        f"{best_mean - champion_mean:+.6f}"
    )

    print(
        f"Champion median W: "
        f"{result_df['champion_W'].median():.6f}"
    )

    print(
        f"Best-lambda median W: "
        f"{result_df['best_W'].median():.6f}"
    )

    print("\nSaved:")
    print(
        "results/models/"
        "champion_plus_fast_fold_scores.csv"
    )


if __name__ == "__main__":
    main()
