import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import SEQ_LEN, FORECAST_HORIZON, TARGET_COLS_4

def compute_prn_embedding_dim(num_satellites: int) -> int:
    return int(math.ceil(1.6 * num_satellites ** 0.52))

class Time2Vec(nn.Module):

    def __init__(self, in_features: int=1, out_features: int=8):
        super().__init__()
        self.out_features = out_features
        self.linear_w = nn.Parameter(torch.randn(in_features, 1))
        self.linear_b = nn.Parameter(torch.randn(1))
        self.periodic_w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.periodic_b = nn.Parameter(torch.randn(out_features - 1))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        linear_part = torch.matmul(tau, self.linear_w) + self.linear_b
        periodic_part = torch.sin(torch.matmul(tau, self.periodic_w) + self.periodic_b)
        return torch.cat([linear_part, periodic_part], dim=-1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class RevIN(nn.Module):

    def __init__(self, num_features: int, eps: float=1e-05, affine: bool=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x_norm = (x - self.mean) / self.stdev
            if self.affine:
                x_norm = x_norm * self.affine_weight + self.affine_bias
            return x_norm
        elif mode == 'denorm':
            x_denorm = x
            if self.affine:
                safe_weight = torch.where(self.affine_weight.abs() < self.eps, self.affine_weight.sign() * self.eps + (self.affine_weight == 0) * self.eps, self.affine_weight)
                x_denorm = (x_denorm - self.affine_bias) / safe_weight
            x_denorm = x_denorm * self.stdev + self.mean
            return x_denorm
        elif mode == 'denorm_sigma':
            if self.affine:
                affine_scale = self.affine_weight.abs().clamp_min(self.eps)
                x = x / affine_scale
            return x * self.stdev
        else:
            raise NotImplementedError(f"RevIN mode '{mode}' is not supported.")

class BiLSTMGRUMHSABackbone(nn.Module):

    def __init__(self, num_features: int, num_satellites: int, d_model: int=64, bilstm_units: int=48, gru_units: int=48, nhead: int=4, num_layers: int=1, num_orbit_classes: int=0, orbit_embedding_dim: int=4, dropout: float=0.1):
        super().__init__()
        self.d_model = d_model
        self.de = compute_prn_embedding_dim(num_satellites)
        self.sat_embedding = nn.Embedding(num_satellites, self.de)
        self.orbit_embedding = nn.Embedding(num_orbit_classes, orbit_embedding_dim) if num_orbit_classes > 0 else None
        self.orbit_embedding_dim = orbit_embedding_dim if self.orbit_embedding is not None else 0
        self.t2v = Time2Vec(in_features=1, out_features=8)
        raw_dim = num_features + self.de + self.orbit_embedding_dim + 8
        self.input_proj = nn.Linear(raw_dim, d_model)
        self.bilstm = nn.LSTM(input_size=d_model, hidden_size=bilstm_units, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=bilstm_units * 2, hidden_size=gru_units, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.mhsa = nn.MultiheadAttention(embed_dim=gru_units, num_heads=nhead, batch_first=True, dropout=dropout)
        self.mhsa_norm = nn.LayerNorm(gru_units)
        self.mhsa_ffn = nn.Sequential(nn.Linear(gru_units, gru_units * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(gru_units * 4, gru_units))
        self.mhsa_ffn_norm = nn.LayerNorm(gru_units)
        self.attention_layers = nn.ModuleList()
        for _ in range(max(1, num_layers) - 1):
            self.attention_layers.append(nn.ModuleDict({'attention': nn.MultiheadAttention(embed_dim=gru_units, num_heads=nhead, batch_first=True, dropout=dropout), 'norm1': nn.LayerNorm(gru_units), 'ffn': nn.Sequential(nn.Linear(gru_units, gru_units * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(gru_units * 4, gru_units)), 'norm2': nn.LayerNorm(gru_units)}))
        self.context_dim = gru_units * 2 + self.de + self.orbit_embedding_dim
        self.seq_feature_dim = gru_units

    def forward(self, x: torch.Tensor, sat_ids: torch.Tensor, orbit_class_ids: torch.Tensor | None=None) -> tuple:
        B, seq_len, _ = x.shape
        tau = torch.linspace(0, 1, seq_len, device=x.device).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1)
        t2v_feats = self.t2v(tau)
        e_prn = self.sat_embedding(sat_ids)
        e_prn_expanded = e_prn.unsqueeze(1).repeat(1, seq_len, 1)
        embeddings = [e_prn_expanded]
        context_embeddings = [e_prn]
        if self.orbit_embedding is not None:
            if orbit_class_ids is None:
                raise ValueError('orbit_class_ids are required when orbit conditioning is enabled')
            e_orbit = self.orbit_embedding(orbit_class_ids)
            embeddings.append(e_orbit.unsqueeze(1).repeat(1, seq_len, 1))
            context_embeddings.append(e_orbit)
        combined_input = torch.cat([x, t2v_feats, *embeddings], dim=-1)
        h = self.input_proj(combined_input)
        bilstm_out, _ = self.bilstm(h)
        bilstm_out = self.dropout1(bilstm_out)
        gru_seq, h_n = self.gru(bilstm_out)
        gru_seq = self.dropout2(gru_seq)
        h_gru = h_n[-1]
        mhsa_out, _ = self.mhsa(gru_seq, gru_seq, gru_seq)
        mhsa_seq = self.mhsa_norm(gru_seq + mhsa_out)
        mhsa_seq = self.mhsa_ffn_norm(mhsa_seq + self.mhsa_ffn(mhsa_seq))
        for block in self.attention_layers:
            attention_out, _ = block['attention'](mhsa_seq, mhsa_seq, mhsa_seq)
            mhsa_seq = block['norm1'](mhsa_seq + attention_out)
            mhsa_seq = block['norm2'](mhsa_seq + block['ffn'](mhsa_seq))
        mhsa_pooled = torch.mean(mhsa_seq, dim=1)
        context = torch.cat([h_gru, mhsa_pooled, *context_embeddings], dim=-1)
        return (mhsa_seq, context)

class ProbabilisticGaussianHead(nn.Module):

    def __init__(self, seq_feature_dim: int, context_dim: int, seq_len: int=SEQ_LEN, forecast_horizon: int=FORECAST_HORIZON, output_dim: int=len(TARGET_COLS_4), separate_orbit_clock_heads: bool=True):
        super().__init__()
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        self.separate_orbit_clock_heads = bool(separate_orbit_clock_heads and output_dim == 4)
        self.temporal_map = nn.Linear(seq_len, forecast_horizon)
        self.context_proj = nn.Linear(context_dim, seq_feature_dim)
        self.feat_norm = nn.LayerNorm(seq_feature_dim)

        def projection(outputs: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(seq_feature_dim, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, outputs * 2))
        if self.separate_orbit_clock_heads:
            self.orbit_proj = projection(3)
            self.clock_proj = projection(1)
            self.proj_net = None
        else:
            self.orbit_proj = None
            self.clock_proj = None
            self.proj_net = projection(output_dim)

    def forward(self, mhsa_seq: torch.Tensor, context: torch.Tensor) -> tuple:
        c_mod = self.context_proj(context).unsqueeze(1)
        h_seq = mhsa_seq + c_mod
        h_seq_t = h_seq.transpose(1, 2)
        h_proj_t = self.temporal_map(h_seq_t)
        h_future = h_proj_t.transpose(1, 2)
        h_norm = self.feat_norm(h_future)
        if self.separate_orbit_clock_heads:
            orbit = self.orbit_proj(h_norm)
            clock = self.clock_proj(h_norm)
            orbit_mu, orbit_raw_scale = orbit.chunk(2, dim=-1)
            clock_mu, clock_raw_scale = clock.chunk(2, dim=-1)
            mu_delta = torch.cat((orbit_mu, clock_mu), dim=-1)
            raw_scale = torch.cat((orbit_raw_scale, clock_raw_scale), dim=-1)
            sigma = F.softplus(raw_scale) + 0.0001
            return (mu_delta, sigma)
        out = self.proj_net(h_norm)
        mu_delta = out[:, :, :self.output_dim]
        sigma = F.softplus(out[:, :, self.output_dim:]) + 0.0001
        return (mu_delta, sigma)

class AnomalySpikeBCEHead(nn.Module):

    def __init__(self, context_dim: int, forecast_horizon: int=FORECAST_HORIZON):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.net = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, forecast_horizon))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)

class GNSSForecaster(nn.Module):

    def __init__(self, num_features: int, num_satellites: int, d_model: int=64, bilstm_units: int=48, gru_units: int=48, nhead: int=4, num_layers: int=1, seq_len: int=SEQ_LEN, forecast_horizon: int=FORECAST_HORIZON, output_dim: int=len(TARGET_COLS_4), target_feature_indices: tuple[int, ...] | None=None, use_revin: bool=True, enable_event_head: bool=False, separate_orbit_clock_heads: bool=False, num_orbit_classes: int=0, orbit_class_by_satellite: tuple[int, ...] | None=None, orbit_embedding_dim: int=4, dropout: float=0.1, **legacy_options):
        super().__init__()
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        self.target_feature_indices = tuple(range(output_dim)) if target_feature_indices is None else tuple(target_feature_indices)
        if len(self.target_feature_indices) != output_dim:
            raise ValueError('target_feature_indices must have one index per output')
        if min(self.target_feature_indices) < 0 or max(self.target_feature_indices) >= num_features:
            raise ValueError('target_feature_indices contains an out-of-range feature index')
        self.use_revin = use_revin
        self.enable_event_head = enable_event_head
        legacy_use_decomposition = legacy_options.pop('use_decomposition', False)
        legacy_options.pop('decomposition_degree', None)
        legacy_options.pop('decomposition_harmonics', None)
        if legacy_options:
            unexpected = ', '.join(sorted(legacy_options))
            raise TypeError(f'unexpected model configuration option(s): {unexpected}')
        if legacy_use_decomposition:
            raise ValueError('Polynomial/harmonic decomposition was removed after it failed the held-out ablation; retrain with the current configuration.')
        if num_orbit_classes > 0:
            if orbit_class_by_satellite is None or len(orbit_class_by_satellite) != num_satellites:
                raise ValueError('orbit_class_by_satellite must map every satellite')
            mapping = torch.as_tensor(orbit_class_by_satellite, dtype=torch.long)
        else:
            mapping = torch.empty(0, dtype=torch.long)
        self.register_buffer('orbit_class_by_satellite', mapping, persistent=False)
        if self.use_revin:
            self.revin = RevIN(num_features=output_dim, affine=True)
        self.backbone = BiLSTMGRUMHSABackbone(num_features=num_features, num_satellites=num_satellites, d_model=d_model, bilstm_units=bilstm_units, gru_units=gru_units, nhead=nhead, num_layers=num_layers, num_orbit_classes=num_orbit_classes, orbit_embedding_dim=orbit_embedding_dim, dropout=dropout)
        context_dim = self.backbone.context_dim
        seq_feature_dim = self.backbone.seq_feature_dim
        self.prob_head = ProbabilisticGaussianHead(seq_feature_dim=seq_feature_dim, context_dim=context_dim, seq_len=seq_len, forecast_horizon=forecast_horizon, output_dim=output_dim, separate_orbit_clock_heads=separate_orbit_clock_heads)
        self.spike_head = AnomalySpikeBCEHead(context_dim=context_dim, forecast_horizon=forecast_horizon) if enable_event_head else None

    def forward(self, x: torch.Tensor, sat_ids: torch.Tensor) -> tuple:
        target_indices = list(self.target_feature_indices)
        target_history = x[:, :, target_indices]
        x_processed = x
        if self.use_revin:
            x_processed = x.clone()
            x_norm = self.revin(target_history, mode='norm')
            x_processed[:, :, target_indices] = x_norm
        orbit_ids = self.orbit_class_by_satellite[sat_ids] if self.orbit_class_by_satellite.numel() else None
        mhsa_seq, context = self.backbone(x_processed, sat_ids, orbit_ids)
        spike_probs = self.spike_head(context) if self.spike_head is not None else context.new_zeros((context.shape[0], self.forecast_horizon))
        mu_delta, sigma = self.prob_head(mhsa_seq, context)
        last_obs = x_processed[:, -1:, target_indices]
        mu = last_obs + mu_delta
        if self.use_revin:
            mu = self.revin(mu, mode='denorm')
            sigma = self.revin(sigma, mode='denorm_sigma')
        return (mu, sigma, spike_probs, context)

# Backward-compatible alias
GNSSHybridForecaster = GNSSForecaster
