import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from evaluation import evaluate_all_targets, ERROR_COLS


# ============================================================
# CONFIG
# ============================================================

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

VAR_ALPHA = 1000.0

LAMBDAS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
]

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# CHAMPION MODEL
# Exact architecture from model_gru_horizon_regime.py
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


# ============================================================
# DATA
# ============================================================

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


# ============================================================
# CHAMPION SEQUENCE CONSTRUCTION
# ============================================================

def make_sequence(history, target_time):

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

    day_fraction = day_minutes / (24.0 * 60.0)

    target_sin = np.sin(
        2.0 * np.pi * day_fraction
    )

    target_cos = np.cos(
        2.0 * np.pi * day_fraction
    )

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
    targets = []
    horizon_regimes = []

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
        np.asarray(horizons, dtype=float),
        np.asarray(horizon_regimes, dtype=np.int64),
        np.asarray(targets, dtype=float),
    )


def build_validation_examples(
    train_df,
    validation_df,
):

    sequences = []
    horizons = []
    targets = []
    horizon_regimes = []

    for _, row in (
        validation_df
        .sort_values("utc_time")
        .iterrows()
    ):

        target_time = row["utc_time"]

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

        targets.append(
            row[ERROR_COLS]
            .astype(float)
            .to_numpy()
        )

    return (
        sequences,
        np.asarray(horizons, dtype=float),
        np.asarray(horizon_regimes, dtype=np.int64),
        np.asarray(targets, dtype=float),
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


# ============================================================
# CHAMPION TRAINING
# ============================================================

def train_gru(
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

    for _ in range(EPOCHS):

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


# ============================================================
# DYNAMICAL VAR EXPERT
# ============================================================

def build_var_examples(indices, E, T):

    X = []
    Y = []

    for i in indices:

        if i < 2:
            continue

        dt = (
            T[i] - T[i - 1]
        ) / np.timedelta64(1, "m")

        dt = float(dt)

        if dt <= 0:
            continue

        current = E[i - 1]
        previous = E[i - 2]
        delta = current - previous

        features = np.concatenate([
            current,
            delta,
            [dt],
            [1.0 / dt],
        ])

        X.append(features)
        Y.append(E[i])

    if not X:
        return (
            np.empty((0, 10)),
            np.empty((0, 4)),
        )

    return (
        np.asarray(X, dtype=float),
        np.asarray(Y, dtype=float),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nGRU + REGULARIZED DYNAMICAL EXPERT BLEND")
    print("=" * 100)
    print(f"Device:      {DEVICE}")
    print(f"VAR alpha:   {VAR_ALPHA}")
    print(f"Lambdas:     {LAMBDAS}")

    # Store validation residuals for every lambda.
    blend_residuals = {
        dataset: {
            lam: []
            for lam in LAMBDAS
        }
        for dataset in DATASETS
    }

    for dataset_name, filename in DATASETS.items():

        df = load_data(
            DATA_DIR / filename
        )

        days = sorted(
            df["utc_time"]
            .dt.normalize()
            .unique()
        )

        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}")
        print("=" * 100)

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

            print(
                f"\nValidation day: "
                f"{validation_day.date()}"
            )

            # ----------------------------------------------------
            # CHAMPION DATA
            # ----------------------------------------------------

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

            gru = train_gru(
                X_train,
                train_lengths,
                train_horizons_scaled,
                train_horizons,
                train_horizon_regimes,
                y_train_scaled,
            )

            gru.eval()

            with torch.no_grad():

                gru_scaled = gru(
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

            gru_pred = y_scaler.inverse_transform(
                gru_scaled
            )

            # ----------------------------------------------------
            # VAR DATA
            # ----------------------------------------------------

            E = train_df[ERROR_COLS].astype(float).to_numpy()
            T = train_df["utc_time"].to_numpy()

            train_indices = np.arange(len(train_df))

            X_var_train, y_var_train = (
                build_var_examples(
                    train_indices,
                    E,
                    T,
                )
            )

            # Validation examples correspond to each target
            # in validation_df and use the final observed state
            # before the target day.
            #
            # For validation days, the VAR makes one-step-style
            # predictions from the chronological training history.
            #
            # The champion predicts all validation timestamps from
            # train_df, so for fairness we use the same available
            # train_df state for each validation target.
            #
            # For irregular validation targets, feature construction
            # is target-specific.
            X_var_val = []
            y_var_val = []

            if len(train_df) >= 2:

                hist_E = train_df[
                    ERROR_COLS
                ].astype(float).to_numpy()

                hist_T = train_df[
                    "utc_time"
                ].to_numpy()

                current = hist_E[-1]
                previous = hist_E[-2]

                for _, row in (
                    validation_df
                    .sort_values("utc_time")
                    .iterrows()
                ):

                    target_time = row["utc_time"]

                    dt = (
                        target_time
                        - pd.Timestamp(hist_T[-1])
                    ).total_seconds() / 60.0

                    if dt <= 0:
                        continue

                    delta = current - previous

                    X_var_val.append(
                        np.concatenate([
                            current,
                            delta,
                            [dt],
                            [1.0 / dt],
                        ])
                    )

                    y_var_val.append(
                        row[
                            ERROR_COLS
                        ].astype(float).to_numpy()
                    )

            X_var_val = np.asarray(
                X_var_val,
                dtype=float,
            )

            y_var_val = np.asarray(
                y_var_val,
                dtype=float,
            )

            if len(X_var_train) == 0 or len(X_var_val) == 0:
                continue

            var_scaler = StandardScaler()

            X_var_train_scaled = (
                var_scaler.fit_transform(
                    X_var_train
                )
            )

            X_var_val_scaled = (
                var_scaler.transform(
                    X_var_val
                )
            )

            var = Ridge(
                alpha=VAR_ALPHA
            )

            var.fit(
                X_var_train_scaled,
                y_var_train,
            )

            var_pred = var.predict(
                X_var_val_scaled
            )

            # ----------------------------------------------------
            # ALIGNMENT CHECK
            # ----------------------------------------------------

            n = min(
                len(y_val),
                len(var_pred),
            )

            if n != len(y_val):
                print(
                    "WARNING: validation alignment "
                    f"GRU={len(y_val)} VAR={len(var_pred)}"
                )

            gru_pred = gru_pred[:n]
            var_pred = var_pred[:n]
            y_val = y_val[:n]

            # ----------------------------------------------------
            # BLEND
            # ----------------------------------------------------

            print(
                f"Points scored: {n}"
            )

            for lam in LAMBDAS:

                prediction = (
                    (1.0 - lam) * gru_pred
                    + lam * var_pred
                )

                residual = (
                    y_val - prediction
                )

                blend_residuals[
                    dataset_name
                ][lam].append(
                    residual
                )

                scores = evaluate_all_targets(
                    residual
                )

                print(
                    f"  lambda={lam:>4.2f}  "
                    f"{scores}"
                )

    # ============================================================
    # GLOBAL SCORE
    #
    # IMPORTANT:
    # Match the champion aggregation:
    # evaluate each validation fold separately, then average
    # the fold-level average_W values.
    # ============================================================

    print("\n" + "=" * 100)
    print("GLOBAL BLEND RESULTS")
    print("=" * 100)

    global_scores = []

    for lam in LAMBDAS:

        fold_average_W = []

        for dataset_name in DATASETS:

            for residual in blend_residuals[
                dataset_name
            ][lam]:

                scores = evaluate_all_targets(
                    residual
                )

                fold_average_W.append(
                    scores["average_W"]
                )

        if not fold_average_W:
            continue

        mean_W = float(
            np.mean(fold_average_W)
        )

        std_W = float(
            np.std(fold_average_W)
        )

        print(
            f"\\nlambda={lam:.2f}  "
            f"mean_fold_W={mean_W:.6f}  "
            f"std_fold_W={std_W:.6f}  "
            f"folds={len(fold_average_W)}"
        )

        global_scores.append(
            (
                lam,
                mean_W,
            )
        )

    print("\\n" + "=" * 100)
    print("LAMBDA RANKING")
    print("=" * 100)

    for lam, score in sorted(
        global_scores,
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"lambda={lam:.2f}  "
            f"mean_fold_W={score:.6f}"
        )

    print("\n" + "=" * 100)
    print("END")
    print("=" * 100)


if __name__ == "__main__":
    main()
