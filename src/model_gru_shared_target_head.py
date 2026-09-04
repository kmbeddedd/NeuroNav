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

torch.manual_seed(SEED)
np.random.seed(SEED)


class SharedTargetHeadGRU(nn.Module):

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

        combined_size = (
            hidden_size + 2 + 4
        )

        # Shared representation.
        self.shared = nn.Sequential(
            nn.Linear(
                combined_size,
                32,
            ),
            nn.ReLU(),
        )

        # Shared output captures the common orbital/clock
        # structure already learned by the champion.
        self.shared_head = nn.Linear(
            32,
            4,
        )

        # Small target-specific residual corrections.
        self.target_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(32, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                )
                for _ in range(4)
            ]
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

        regime_repr = self.horizon_embedding(
            horizon_regime
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

        shared_repr = self.shared(
            combined
        )

        shared_output = self.shared_head(
            shared_repr
        )

        residuals = torch.cat(
            [
                head(shared_repr)
                for head in self.target_heads
            ],
            dim=1,
        )

        return (
            shared_output
            + residuals
        )


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
                "utc_time",
                *ERROR_COLS,
            ]
        )
        .sort_values("utc_time")
        .reset_index(drop=True)
    )


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
        day_minutes
        / (24.0 * 60.0)
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
        (
            target_time
            - history["utc_time"]
        )
        .dt.total_seconds()
        .to_numpy()
        / 60.0
    )

    gaps = (
        history["utc_time"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
        .fillna(0.0)
        .to_numpy()
    )

    target_phase = np.column_stack(
        [
            np.full(
                len(values),
                target_sin,
            ),
            np.full(
                len(values),
                target_cos,
            ),
        ]
    )

    sequence = np.column_stack(
        [
            values,
            gaps,
            time_to_target,
            target_phase,
        ]
    )

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


def build_training_examples(df):

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
        len(df),
    ):

        target_time = df.loc[
            i,
            "utc_time",
        ]

        result = make_sequence(
            df.iloc[:i],
            target_time,
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
            row[ERROR_COLS]
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


def pad_sequences(
    sequences,
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
    horizons,
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


def train_model(
    X,
    lengths,
    horizons_scaled,
    horizons_raw,
    regimes,
    y,
):

    model = SharedTargetHeadGRU(
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

    weights = torch.tensor(
        horizon_weights(
            horizons_raw
        ),
        dtype=torch.float32,
        device=DEVICE,
    ).reshape(-1, 1)

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
        device=DEVICE,
    )

    model.train()

    for _ in range(EPOCHS):

        optimizer.zero_grad()

        predictions = model(
            X_tensor,
            lengths_tensor,
            horizon_tensor,
            regime_tensor,
        )

        loss = (
            (
                predictions
                - y_tensor
            ) ** 2
            * weights
        ).mean()

        loss.backward()
        optimizer.step()

    return model


def main():

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 90)
    print(
        "SHARED + TARGET-SPECIFIC RESIDUAL HEAD"
    )
    print("=" * 90)

    fold_results = []

    validation_days = {
        "GEO": [
            pd.Timestamp("2025-09-02"),
            pd.Timestamp("2025-09-03"),
            pd.Timestamp("2025-09-04"),
            pd.Timestamp("2025-09-05"),
            pd.Timestamp("2025-09-06"),
            pd.Timestamp("2025-09-07"),
        ],
        "MEO": [
            pd.Timestamp("2025-09-02"),
            pd.Timestamp("2025-09-03"),
            pd.Timestamp("2025-09-05"),
            pd.Timestamp("2025-09-06"),
            pd.Timestamp("2025-09-07"),
        ],
        "MEO2": [
            pd.Timestamp("2025-09-04"),
            pd.Timestamp("2025-09-06"),
            pd.Timestamp("2025-09-07"),
            pd.Timestamp("2025-09-08"),
            pd.Timestamp("2025-09-09"),
        ],
    }

    for dataset_name, filename in DATASETS.items():

        print(
            f"\n{'=' * 90}\n"
            f"{dataset_name}\n"
            f"{'=' * 90}"
        )

        df = load_data(
            DATA_DIR / filename
        )

        for validation_day in validation_days[
            dataset_name
        ]:

            train_df = df[
                df["utc_time"]
                < validation_day
            ].copy()

            val_df = df[
                df["utc_time"].dt.date
                == validation_day.date()
            ].copy()

            if len(val_df) == 0:
                continue

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
                val_df
            )

            if (
                len(train_sequences) == 0
                or len(val_sequences) == 0
            ):
                continue

            x_scaler = StandardScaler()

            x_scaler.fit(
                np.concatenate(
                    train_sequences,
                    axis=0,
                )
            )

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
                train_horizons.reshape(
                    -1,
                    1,
                )
            )

            train_h_scaled = (
                horizon_scaler
                .transform(
                    train_horizons.reshape(
                        -1,
                        1,
                    )
                )
                .ravel()
            )

            val_h_scaled = (
                horizon_scaler
                .transform(
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

            torch.manual_seed(SEED)
            np.random.seed(SEED)

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
                        val_h_scaled.reshape(
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

            predictions = (
                y_scaler.inverse_transform(
                    pred_scaled
                    .cpu()
                    .numpy()
                )
            )

            residuals = (
                y_val
                - predictions
            )

            metrics = (
                evaluate_all_targets(
                    residuals
                )
            )

            average_W = float(
                np.mean(
                    [
                        metrics[col]["W"]
                        for col in ERROR_COLS
                    ]
                )
            )

            fold_results.append(
                {
                    "dataset": dataset_name,
                    "validation_day":
                        validation_day.date(),
                    "average_W":
                        average_W,
                }
            )

            print(
                f"{validation_day.date()} "
                f"Average W = "
                f"{average_W:.6f}"
            )

    result_df = pd.DataFrame(
        fold_results
    )

    output_dir = (
        DATA_DIR.parent
        / "results"
        / "models"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        output_dir
        / "shared_target_head_fold_scores.csv",
        index=False,
    )

    print("\n" + "=" * 90)
    print(
        f"Mean fold Average W: "
        f"{result_df['average_W'].mean():.6f}"
    )
    print(
        f"Median fold Average W: "
        f"{result_df['average_W'].median():.6f}"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
