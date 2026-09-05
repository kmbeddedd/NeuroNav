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


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x, lengths):

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.gru(packed)

        return self.head(hidden[-1])


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
        df.drop_duplicates()
        .dropna(subset=["utc_time"] + ERROR_COLS)
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

    target_time = pd.Timestamp(target_time)

    history_start = (
        target_time
        - pd.Timedelta(days=HISTORY_DAYS)
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

    times = (
        history["utc_time"]
        .astype("int64")
        .to_numpy()
    )

    target_ns = target_time.value

    time_to_target = (
        (target_ns - times)
        / 60_000_000_000
    ).astype(float)

    gaps = (
        np.diff(times)
        / 60_000_000_000
    ).astype(float)

    gaps = np.concatenate([
        [0.0],
        gaps,
    ])

    # First differences of the four error variables.
    differences = np.zeros_like(values)

    if len(values) > 1:
        differences[1:] = np.diff(
            values,
            axis=0
        )

    sequence = np.column_stack([
        values,
        differences,
        gaps,
        time_to_target,
    ])

    return sequence.astype(float)


def build_training_examples(train_df):

    sequences = []
    targets = []

    train_df = (
        train_df
        .sort_values("utc_time")
        .reset_index(drop=True)
    )

    for i in range(MIN_HISTORY, len(train_df)):

        target_time = train_df.loc[
            i,
            "utc_time"
        ]

        history = train_df.iloc[:i]

        sequence = make_sequence(
            history,
            target_time,
        )

        if sequence is None:
            continue

        sequences.append(sequence)

        targets.append(
            train_df.loc[
                i,
                ERROR_COLS
            ].astype(float).to_numpy()
        )

    return sequences, np.asarray(
        targets,
        dtype=float,
    )


def build_validation_examples(
    train_df,
    validation_df,
):

    sequences = []
    targets = []

    train_df = (
        train_df
        .sort_values("utc_time")
        .reset_index(drop=True)
    )

    validation_df = (
        validation_df
        .sort_values("utc_time")
        .reset_index(drop=True)
    )

    for _, row in validation_df.iterrows():

        target_time = row["utc_time"]

        sequence = make_sequence(
            train_df,
            target_time,
        )

        if sequence is None:
            continue

        sequences.append(sequence)

        targets.append(
            row[ERROR_COLS]
            .astype(float)
            .to_numpy()
        )

    return sequences, np.asarray(
        targets,
        dtype=float,
    )


def pad_sequences(sequences):

    max_len = max(
        len(seq)
        for seq in sequences
    )

    n = len(sequences)
    feature_dim = sequences[0].shape[1]

    X = np.zeros(
        (n, max_len, feature_dim),
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


def train_model(
    X,
    lengths,
    y,
):

    model = GRUModel(
        input_size=X.shape[2],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = nn.MSELoss()

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
        )

        loss = loss_fn(
            predictions,
            y_tensor,
        )

        loss.backward()

        optimizer.step()

    return model


def main():

    print("\nENHANCED GRU — TEMPORAL DIFFERENCES")
    print("=" * 90)
    print(f"Device: {DEVICE}")

    fold_results = []

    for dataset_name, filename in DATASETS.items():

        df = load_data(
            DATA_DIR / filename
        )

        days = sorted(
            df["utc_time"]
            .dt.normalize()
            .unique()
        )

        print(f"\n{dataset_name}")
        print("-" * 90)

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

            train_sequences, y_train = (
                build_training_examples(
                    train_df
                )
            )

            val_sequences, y_val = (
                build_validation_examples(
                    train_df,
                    validation_df,
                )
            )

            if len(train_sequences) < 5:
                continue

            if len(val_sequences) == 0:
                continue

            # Fit the input scaler on training data only.
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

            val_sequences = [
                x_scaler.transform(seq)
                for seq in val_sequences
            ]

            # Fit target scaler on training targets only.
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

            model = train_model(
                X_train,
                train_lengths,
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
                ).cpu().numpy()

            predictions = (
                y_scaler.inverse_transform(
                    predictions_scaled
                )
            )

            residuals = (
                y_val - predictions
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
        exist_ok=True
    )

    result_df = pd.DataFrame(
        fold_results
    )

    result_df.to_csv(
        output_dir /
        "enhanced_gru_fold_scores.csv",
        index=False,
    )

    print("\n")
    print("=" * 90)
    print("ENHANCED GRU OVERALL")
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
        "enhanced_gru_fold_scores.csv"
    )


if __name__ == "__main__":
    main()
