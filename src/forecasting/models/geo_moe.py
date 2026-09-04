"""GEO Gated Mixture-of-Experts (MoE) Core Forecaster (Promoted Architecture).

Winner of the official Day-8 Benchmark.
Architecture:
- Bidirectional GRU history encoder (24h physical history)
- Query projection head
- Regime gating head predicting excursion probability p_gate in [0, 1]
- Normal expert head + Excursion expert head
- Final delta = (1 - p_gate) * normal_delta + p_gate * excursion_delta
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.forecasting.base import ForecastModel, ModelMetadata
from src.forecasting.models.harmonic_ridge import extract_harmonic_time_features, HarmonicRidgeModel


class GEOGatedMoENetwork(nn.Module):
    """PyTorch network for Causal Residual Gated Mixture-of-Experts."""

    def __init__(self, history_dim: int = 16, query_dim: int = 13, hidden_dim: int = 24):
        super().__init__()
        enc_out = hidden_dim * 2  # Bidirectional GRU
        self.gru = nn.GRU(history_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(enc_out)
        self.dropout = nn.Dropout(0.1)
        self.query_proj = nn.Sequential(
            nn.Linear(query_dim, 24),
            nn.SiLU(),
            nn.Linear(24, 24),
        )
        fused = enc_out + 24

        # Gate head
        self.gate_head = nn.Sequential(
            nn.Linear(fused, 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )

        # Normal regime expert
        self.normal_head = nn.Sequential(
            nn.Linear(fused, 32),
            nn.SiLU(),
            nn.Linear(32, 4),
        )

        # Excursion regime expert (wider capacity)
        self.excursion_head = nn.Sequential(
            nn.Linear(fused, 48),
            nn.SiLU(),
            nn.Linear(48, 32),
            nn.SiLU(),
            nn.Linear(32, 4),
        )

    def forward(
        self,
        history: torch.Tensor,
        query: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: delta_pred (B, 4), gate_logit (B,), p_gate (B, 1)"""
        out, _ = self.gru(history)
        h_enc = self.dropout(self.ln(out[:, -1]))
        h_q = self.query_proj(query)
        fused = torch.cat((h_enc, h_q), dim=-1)

        gate_logit = self.gate_head(fused).squeeze(-1)
        p_gate = torch.sigmoid(gate_logit).unsqueeze(-1)

        normal_delta = self.normal_head(fused)
        excursion_delta = self.excursion_head(fused)

        delta_pred = (1.0 - p_gate) * normal_delta + p_gate * excursion_delta
        return delta_pred, gate_logit, p_gate


class GEOGatedMoEModel(ForecastModel):
    """Core adapter for GEO Gated Mixture-of-Experts residual forecaster."""

    def __init__(
        self,
        name: str = "GEO Gated MoE",
        history_dim: int = 16,
        query_dim: int = 13,
        hidden_dim: int = 24,
        max_epochs: int = 60,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        version: str = "1.0.0",
        **kwargs,
    ):
        super().__init__(name=name, model_type="geo_moe", version=version)
        self.orbit_class = kwargs.get("orbit_class", "GEO")
        self.history_dim = history_dim
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device(device)
        self.target_cols = ["x_error_m", "y_error_m", "z_error_m", "clock_error_m"]

        self.baseline_model: Optional[HarmonicRidgeModel] = None
        self.network: Optional[GEOGatedMoENetwork] = None
        self.origin: Optional[pd.Timestamp] = None
        self.center_d: np.ndarray = np.zeros(4, dtype=np.float64)
        self.scale_d: np.ndarray = np.ones(4, dtype=np.float64)
        self.detector: Dict[str, float] = {}

    def _extract_history_features(
        self,
        sub_df: pd.DataFrame,
        query_time: pd.Timestamp,
        base_vals: np.ndarray,
        max_len: int = 24,
    ) -> np.ndarray:
        """Constructs physical step features for the GRU history window."""
        vals = sub_df[self.target_cols].to_numpy(dtype=np.float64)
        deltas = (vals - base_vals - self.center_d) / self.scale_d

        times = pd.to_datetime(sub_df["utc_time"])
        time_secs = times.to_numpy(dtype="datetime64[ns]").astype(np.int64) / 1e9
        query_sec = pd.Timestamp(query_time).timestamp()

        dt_hours = np.zeros(len(times), dtype=float)
        dt_hours[1:] = (time_secs[1:] - time_secs[:-1]) / 3600.0
        dt_hours[0] = dt_hours[1] if len(dt_hours) > 1 else 0.25

        age_hours = (query_sec - time_secs) / 3600.0

        phase = 2.0 * np.pi * (
            times.dt.hour * 3600.0 + times.dt.minute * 60.0 + times.dt.second
        ).to_numpy(dtype=float) / 86400.0

        diff_vals = np.zeros_like(vals)
        diff_vals[1:] = vals[1:] - vals[:-1]
        diff_norm = diff_vals / self.scale_d

        norm_3d = np.sqrt(np.sum(vals[:, :3] ** 2, axis=1, keepdims=True)) / 20.0

        step_features = np.column_stack([
            deltas,
            vals / 20.0,
            diff_norm,
            norm_3d,
            (dt_hours / 2.0).reshape(-1, 1),
            (age_hours / 24.0).reshape(-1, 1),
            np.sin(phase).reshape(-1, 1),
            np.cos(phase).reshape(-1, 1),
        ]).astype(np.float32)

        # Pad to max_len if shorter
        if len(step_features) < max_len:
            padding = np.repeat(step_features[[0]], max_len - len(step_features), axis=0)
            step_features = np.vstack([padding, step_features])
        elif len(step_features) > max_len:
            step_features = step_features[-max_len:]

        return step_features

    def _extract_query_features(
        self,
        query_time: pd.Timestamp,
        history_end: pd.Timestamp,
        recent_norm_rms: float,
    ) -> np.ndarray:
        lead_hours = (query_time - history_end).total_seconds() / 3600.0
        elapsed_days = (query_time - self.origin).total_seconds() / 86400.0
        phase = 2.0 * np.pi * (
            query_time.hour * 3600.0 + query_time.minute * 60.0 + query_time.second
        ) / 86400.0

        harmonics = []
        for k in (1, 2, 3, 4):
            harmonics.extend([math.sin(k * phase), math.cos(k * phase)])

        # Logistic regime probability from recent RMS
        x0 = self.detector.get("x0", 1.0)
        scale = max(self.detector.get("scale", 1.0), 0.5)
        p_regime = 1.0 / (1.0 + math.exp(-((recent_norm_rms - x0) / scale)))

        query_arr = [
            lead_hours / 24.0,
            elapsed_days / 7.0,
            p_regime,
            recent_norm_rms / 20.0,
            *harmonics,
        ]
        return np.asarray(query_arr, dtype=np.float32)

    def fit(self, train_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> "GEOGatedMoEModel":
        clean = train_df.dropna(subset=["utc_time", *self.target_cols]).sort_values("utc_time").reset_index(drop=True)
        self.origin = clean["utc_time"].iloc[0].floor("D")

        # 1. Fit baseline Harmonic Ridge model
        self.baseline_model = HarmonicRidgeModel()
        self.baseline_model.fit(clean)
        base_preds = self.baseline_model.predict(clean, clean["utc_time"])

        # 2. Compute residual scaling stats
        vals = clean[self.target_cols].to_numpy(dtype=np.float64)
        raw_deltas = vals - base_preds
        self.center_d = np.median(raw_deltas, axis=0)
        q25, q75 = np.percentile(raw_deltas, [25, 75], axis=0)
        self.scale_d = np.maximum((q75 - q25) / 1.349, 0.01)

        # 3. Fit regime detector on 3D error norms
        orbit_norms = np.sqrt(np.sum(vals[:, :3] ** 2, axis=1))
        s_norm = pd.Series(orbit_norms)
        roll_rms = s_norm.rolling(4, min_periods=1).apply(lambda w: float(np.sqrt(np.mean(w ** 2)))).to_numpy()
        x0 = float(np.quantile(roll_rms, 0.75))
        p90 = float(np.quantile(roll_rms, 0.90))
        self.detector = {"x0": x0, "scale": max((p90 - x0) / 2.0, 0.5)}

        # 4. Generate training samples
        hists, queries, targets, regimes = [], [], [], []
        n_rows = len(clean)
        for i in range(4, n_rows):
            query_time = clean["utc_time"].iloc[i]
            # Use history strictly before query_time
            hist_sub = clean.iloc[:i]
            base_sub = base_preds[:i]

            hist_feat = self._extract_history_features(hist_sub, query_time, base_sub)
            recent_rms = float(np.sqrt(np.mean(orbit_norms[max(0, i - 4):i] ** 2)))
            q_feat = self._extract_query_features(query_time, hist_sub["utc_time"].iloc[-1], recent_rms)

            true_delta = (vals[i] - base_preds[i] - self.center_d) / self.scale_d
            t_norm = orbit_norms[i]
            t_regime = 1.0 / (1.0 + math.exp(-((t_norm - x0) / max(self.detector["scale"], 0.5))))

            hists.append(hist_feat)
            queries.append(q_feat)
            targets.append(true_delta)
            regimes.append(t_regime)

        if not hists:
            raise ValueError("Insufficient rows to construct training examples for GEO Gated MoE")

        hist_tensor = torch.as_tensor(np.asarray(hists), dtype=torch.float32, device=self.device)
        query_tensor = torch.as_tensor(np.asarray(queries), dtype=torch.float32, device=self.device)
        target_tensor = torch.as_tensor(np.asarray(targets), dtype=torch.float32, device=self.device)
        regime_tensor = torch.as_tensor(np.asarray(regimes), dtype=torch.float32, device=self.device)

        # 5. Initialize and train network
        self.history_dim = int(hist_tensor.shape[-1])
        self.query_dim = int(query_tensor.shape[-1])
        self.network = GEOGatedMoENetwork(
            history_dim=self.history_dim,
            query_dim=self.query_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        huber = nn.SmoothL1Loss(beta=1.0)
        bce = nn.BCEWithLogitsLoss()

        self.network.train()
        for _ in range(self.max_epochs):
            optimizer.zero_grad()
            delta_pred, gate_logit, _ = self.network(hist_tensor, query_tensor)
            loss_delta = huber(delta_pred, target_tensor)
            loss_gate = bce(gate_logit, regime_tensor)
            loss = loss_delta + 0.1 * loss_gate
            loss.backward()
            optimizer.step()

        self.network.eval()
        self.is_fitted = True
        return self

    def predict(
        self,
        history_df: pd.DataFrame,
        horizon_epochs: Union[int, pd.DatetimeIndex, Sequence[pd.Timestamp]],
    ) -> np.ndarray:
        if not self.is_fitted or self.network is None or self.baseline_model is None:
            raise ValueError("GEOGatedMoEModel must be fitted before predict()")

        clean_hist = history_df.dropna(subset=["utc_time", *self.target_cols]).sort_values("utc_time").reset_index(drop=True)
        if clean_hist.empty:
            raise ValueError("history_df must contain valid historical observations")

        if isinstance(horizon_epochs, int):
            last_time = pd.to_datetime(clean_hist["utc_time"].iloc[-1])
            step_interval = pd.Timedelta(minutes=15)
            forecast_times = pd.date_range(start=last_time + step_interval, periods=horizon_epochs, freq=step_interval)
        else:
            forecast_times = pd.DatetimeIndex(horizon_epochs)

        # Baseline predictions
        base_preds = self.baseline_model.predict(clean_hist, forecast_times)
        hist_base = self.baseline_model.predict(clean_hist, clean_hist["utc_time"])

        vals_hist = clean_hist[self.target_cols].to_numpy(dtype=np.float64)
        orbit_norms = np.sqrt(np.sum(vals_hist[:, :3] ** 2, axis=1))
        recent_rms = float(np.sqrt(np.mean(orbit_norms[-4:] ** 2))) if len(orbit_norms) else 1.0

        hist_end = clean_hist["utc_time"].iloc[-1]
        preds = []

        self.network.eval()
        with torch.no_grad():
            for idx, q_time in enumerate(forecast_times):
                h_feat = self._extract_history_features(clean_hist, q_time, hist_base)
                q_feat = self._extract_query_features(q_time, hist_end, recent_rms)

                h_t = torch.as_tensor(h_feat[None, ...], dtype=torch.float32, device=self.device)
                q_t = torch.as_tensor(q_feat[None, ...], dtype=torch.float32, device=self.device)

                delta_pred, _, _ = self.network(h_t, q_t)
                scaled_delta = delta_pred.cpu().numpy()[0] * self.scale_d + self.center_d
                total_pred = base_preds[idx] + scaled_delta
                preds.append(total_pred)

        return np.asarray(preds, dtype=np.float64)

    def save(self, path: Union[str, Path]) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "model_type": self.model_type,
            "version": self.version,
            "history_dim": self.history_dim,
            "query_dim": self.query_dim,
            "hidden_dim": self.hidden_dim,
            "max_epochs": self.max_epochs,
            "lr": self.lr,
            "origin": self.origin.isoformat() if self.origin else None,
            "center_d": self.center_d.tolist(),
            "scale_d": self.scale_d.tolist(),
            "detector": self.detector,
            "baseline_model": self.baseline_model,
            "network_state_dict": self.network.state_dict() if self.network else None,
            "target_cols": self.target_cols,
        }
        torch.save(payload, out_path)
        return out_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GEOGatedMoEModel":
        p = Path(path)
        payload = torch.load(p, map_location="cpu", weights_only=False)
        state_dict = payload.get("network_state_dict")
        h_dim = payload.get("history_dim")
        q_dim = payload.get("query_dim")
        if state_dict:
            if "gru.weight_ih_l0" in state_dict:
                h_dim = int(state_dict["gru.weight_ih_l0"].shape[1])
            if "query_proj.0.weight" in state_dict:
                q_dim = int(state_dict["query_proj.0.weight"].shape[1])

        model = cls(
            name=payload.get("name", "GEO Gated MoE"),
            history_dim=h_dim if h_dim is not None else 17,
            query_dim=q_dim if q_dim is not None else 12,
            hidden_dim=payload.get("hidden_dim", 24),
            version=payload.get("version", "1.0.0"),
        )
        model.baseline_model = payload["baseline_model"]
        model.center_d = np.asarray(payload["center_d"], dtype=np.float64)
        model.scale_d = np.asarray(payload["scale_d"], dtype=np.float64)
        model.detector = payload["detector"]
        if payload.get("origin"):
            model.origin = pd.Timestamp(payload["origin"])

        if payload.get("network_state_dict"):
            model.network = GEOGatedMoENetwork(
                history_dim=model.history_dim,
                query_dim=model.query_dim,
                hidden_dim=model.hidden_dim,
            )
            model.network.load_state_dict(payload["network_state_dict"])
            model.network.eval()

        model.is_fitted = True
        return model

    def get_metadata(self) -> ModelMetadata:
        n_params = sum(p.numel() for p in self.network.parameters()) if self.network else 10000
        return ModelMetadata(
            name=self.name,
            model_type=self.model_type,
            version=self.version,
            architecture="Causal Gated Mixture-of-Experts with GRU Regime Encoder",
            parameters={"hidden_dim": self.hidden_dim, "max_epochs": self.max_epochs},
            lookback_steps=24,
            forecast_horizon=96,
            features=["physical_deltas", "orbit_norms", "phase_harmonics", "regime_prob"],
            target_representation="ECEF",
            supports_uncertainty=False,
            trainable=True,
            description="Benchmark-winning gated mixture-of-experts model interpolating between normal and excursion heads.",
            parameter_count=n_params,
        )
