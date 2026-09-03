"""Train small BiLSTM models for the supplied GNSS error datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATASETS = {
    "GEO": ("DATA_GEO_Train.csv", "DATA_GEO_Test.csv"),
    "MEO": ("DATA_MEO_Train.csv", "DATA_MEO_Test.csv"),
    "MEO2": ("DATA_MEO_Train2.csv", "DATA_MEO_Test2.csv"),
}
TARGETS = ["x_error (m)", "y_error (m)", "z_error (m)", "satclockerror (m)"]


class BiLSTM(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.output(sequence[:, -1, :])


def read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [" ".join(column.split()) for column in frame.columns]
    frame["utc_time"] = pd.to_datetime(frame["utc_time"])
    return frame


def split_sequences(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Split concatenated satellites whenever their timestamp resets."""
    reset = frame["utc_time"].diff().dt.total_seconds().fillna(1) <= 0
    starts = np.flatnonzero(reset.to_numpy())
    boundaries = np.r_[0, starts, len(frame)]
    boundaries = np.unique(boundaries)
    return [frame.iloc[a:b].reset_index(drop=True) for a, b in zip(boundaries[:-1], boundaries[1:])]


def window_samples(sequences: list[np.ndarray], window: int) -> tuple[np.ndarray, np.ndarray]:
    inputs, labels = [], []
    for sequence in sequences:
        for index in range(window, len(sequence)):
            inputs.append(sequence[index - window : index])
            labels.append(sequence[index])
    return np.asarray(inputs, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def recursive_forecast(
    model: nn.Module,
    history: np.ndarray,
    steps: int,
    window: int,
    device: torch.device,
) -> np.ndarray:
    values = list(history[-window:].copy())
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(np.asarray(values[-window:]), dtype=torch.float32, device=device).unsqueeze(0)
            values.append(model(x).squeeze(0).cpu().numpy())
    return np.asarray(values[-steps:])


def train_dataset(
    name: str,
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    epochs: int,
    window: int,
    seed: int,
    device: torch.device,
) -> dict:
    train_frame = read_csv(train_path)
    test_frame = read_csv(test_path)
    train_parts = split_sequences(train_frame)
    test_parts = split_sequences(test_frame)
    if len(train_parts) != len(test_parts):
        raise ValueError(f"{name}: found {len(train_parts)} train and {len(test_parts)} test sequences")

    effective_window = min(window, min(len(part) for part in train_parts) - 1)
    scaler = StandardScaler().fit(train_frame[TARGETS].to_numpy())
    scaled_train = [scaler.transform(part[TARGETS].to_numpy()) for part in train_parts]
    x_train, y_train = window_samples(scaled_train, effective_window)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=min(16, len(x_train)),
        shuffle=True,
        generator=generator,
    )
    model = BiLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        total = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(inputs)
        final_loss = total / len(x_train)

    prediction_parts = []
    for train_sequence, test_part in zip(scaled_train, test_parts):
        scaled_prediction = recursive_forecast(
            model, train_sequence, len(test_part), effective_window, device
        )
        prediction_parts.append(scaler.inverse_transform(scaled_prediction))
    predictions = np.vstack(prediction_parts)
    truth = test_frame[TARGETS].to_numpy()

    result = test_frame.copy()
    for index, target in enumerate(TARGETS):
        result[f"predicted_{target}"] = predictions[:, index]
    result.to_csv(output_dir / f"{name}_predictions.csv", index=False)

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "window": effective_window,
        "targets": TARGETS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    torch.save(checkpoint, output_dir / f"{name}_bilstm.pt")

    return {
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "sequences": len(train_parts),
        "window": effective_window,
        "final_training_mse_scaled": final_loss,
        "overall_mae_m": float(mean_absolute_error(truth, predictions)),
        "overall_rmse_m": float(mean_squared_error(truth, predictions) ** 0.5),
        "mae_by_target_m": {
            target: float(mean_absolute_error(truth[:, i], predictions[:, i]))
            for i, target in enumerate(TARGETS)
        },
        "rmse_by_target_m": {
            target: float(mean_squared_error(truth[:, i], predictions[:, i]) ** 0.5)
            for i, target in enumerate(TARGETS)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Data_PS-08"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    for name, (train_file, test_file) in DATASETS.items():
        metrics[name] = train_dataset(
            name=name,
            train_path=args.data_dir / train_file,
            test_path=args.data_dir / test_file,
            output_dir=args.output_dir,
            epochs=args.epochs,
            window=args.window,
            seed=args.seed,
            device=device,
        )
        print(
            f"{name}: MAE={metrics[name]['overall_mae_m']:.4f} m, "
            f"RMSE={metrics[name]['overall_rmse_m']:.4f} m"
        )

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved models, predictions, and metrics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
