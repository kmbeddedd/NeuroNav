import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.preprocessing import StandardScaler

from evaluation import ERROR_COLS


DATA_DIR = Path("data")

TRAIN_FILE = DATA_DIR / "DATA_GEO_Train.csv"
TEST_FILE = DATA_DIR / "DATA_GEO_Test.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
HISTORY_DAYS = 7
MIN_HISTORY = 2
EPOCHS = 300

torch.manual_seed(SEED)
np.random.seed(SEED)


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
            nn.Linear(hidden_size + 2 + 4, 32),
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

        _, hidden = self.gru(packed)

        history_repr = hidden[-1]

        regime_repr = self.horizon_embedding(
            horizon_regime
        )

        horizon_norm = horizon / 1440.0

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


def load_data(path):

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
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
            subset=["utc_time"] + ERROR_COLS
        )
        .sort_values("utc_time")
        .reset_index(drop=True)
    )


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
        df[
            ["utc_time"]
        ]
        .drop_duplicates()
        .sort_values("utc_time")
        .reset_index(drop=True)
    )


def make_sequence(history, target_time):

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
        np.full(len(values), target_sin),
        np.full(len(values), target_cos),
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


def build_training_examples(df):

    sequences = []
    horizons = []
    horizon_regimes = []
    targets = []

    df = (
        df
        .sort_values("utc_time")
        .reset_index(drop=True)
    )

    for i in range(MIN_HISTORY, len(df)):

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

        sequence, horizon, horizon_regime = result

        sequences.append(sequence)
        horizons.append(horizon)
        horizon_regimes.append(horizon_regime)

        targets.append(
            df.loc[
                i,
                ERROR_COLS
            ].astype(float).to_numpy()
        )

    return (
        sequences,
        np.asarray(
            horizons,
            dtype=float
        ),
        np.asarray(
            horizon_regimes,
            dtype=np.int64
        ),
        np.asarray(
            targets,
            dtype=float
        ),
    )


def build_test_examples(
    train_df,
    test_times,
):

    sequences = []
    horizons = []
    horizon_regimes = []
    target_times = []

    for target_time in test_times:

        result = make_sequence(
            train_df,
            target_time,
        )

        if result is None:
            continue

        sequence, horizon, horizon_regime = result

        sequences.append(sequence)
        horizons.append(horizon)
        horizon_regimes.append(horizon_regime)
        target_times.append(
            target_time
        )

    return (
        sequences,
        np.asarray(
            horizons,
            dtype=float
        ),
        np.asarray(
            horizon_regimes,
            dtype=np.int64
        ),
        pd.to_datetime(
            target_times
        ),
    )


def pad_sequences(sequences):

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
            feature_dim
        ),
        dtype=np.float32,
    )

    lengths = np.zeros(
        n,
        dtype=np.int64,
    )

    for i, seq in enumerate(sequences):

        length = len(seq)

        X[i, :length] = seq
        lengths[i] = length

    return X, lengths


def horizon_weights(horizons):

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
    horizon_regimes,
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

    horizon_regime_tensor = torch.tensor(
        horizon_regimes,
        dtype=torch.long,
        device=DEVICE,
    )

    weights = torch.tensor(
        horizon_weights(horizons_raw),
        dtype=torch.float32,
        device=DEVICE,
    ).reshape(-1, 1)

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
        device=DEVICE,
    )

    model.train()

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        predictions = model(
            X_tensor,
            lengths_tensor,
            horizon_tensor,
            horizon_regime_tensor,
        )

        loss = (
            (predictions - y_tensor) ** 2
            * weights
        ).mean()

        loss.backward()
        optimizer.step()

    return model


def main():

    print("=" * 90)
    print("GEO CHAMPION INFERENCE")
    print("=" * 90)
    print(f"Device: {DEVICE}")

    train_df = load_data(
        TRAIN_FILE
    )

    test_df = load_test_times(
        TEST_FILE
    )

    print(
        f"Training rows: {len(train_df)}"
    )

    print(
        f"Test timestamps: {len(test_df)}"
    )

    train_sequences, train_horizons, train_horizon_regimes, y_train = (
        build_training_examples(
            train_df
        )
    )

    test_sequences, test_horizons, test_horizon_regimes, test_times = (
        build_test_examples(
            train_df,
            test_df["utc_time"].tolist()
        )
    )

    if len(test_sequences) == 0:
        raise RuntimeError(
            "No valid test sequences were created."
        )

    print(
        f"Training examples: {len(train_sequences)}"
    )

    print(
        f"Usable test examples: {len(test_sequences)}"
    )

    if len(test_sequences) != len(test_df):
        print(
            "WARNING: Some test timestamps "
            "could not be predicted."
        )

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

    print(
        f"Test horizon: "
        f"{test_horizons.min():.1f}"
        f" -> "
        f"{test_horizons.max():.1f} min"
    )

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
                test_horizons_scaled.reshape(-1, 1),
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

    output = pd.DataFrame(
        {
            "utc_time": test_times,
            "pred_x_error (m)": predictions[:, 0],
            "pred_y_error (m)": predictions[:, 1],
            "pred_z_error (m)": predictions[:, 2],
            "pred_satclockerror (m)": predictions[:, 3],
            "horizon_min": test_horizons,
            "horizon_regime": test_horizon_regimes,
        }
    )

    output_dir = Path(
        "results/predictions"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "GEO_champion_predictions.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print("\nPrediction summary:")
    print(output.head(10).to_string(index=False))

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()