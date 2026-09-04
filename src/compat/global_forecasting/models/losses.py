import torch
import torch.nn.functional as F

def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    weights = torch.broadcast_to(mask.to(dtype=values.dtype), values.shape)
    denominator = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denominator

def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
    sigma = sigma.clamp_min(1e-05)
    loss = torch.log(sigma) + (targets - mu) ** 2 / (2 * sigma.square())
    return _masked_mean(loss, mask)

def student_t_nll_loss(mu: torch.Tensor, scale: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None=None, degrees_of_freedom: float=3.0) -> torch.Tensor:
    if degrees_of_freedom <= 2.0:
        raise ValueError('degrees_of_freedom must be greater than 2')
    scale = scale.clamp_min(1e-05)
    df = torch.as_tensor(degrees_of_freedom, dtype=mu.dtype, device=mu.device)
    z2 = ((targets - mu) / scale).square()
    nll = torch.log(scale) + 0.5 * torch.log(df * torch.pi) + torch.lgamma(0.5 * df) - torch.lgamma(0.5 * (df + 1.0)) + 0.5 * (df + 1.0) * torch.log1p(z2 / df)
    return _masked_mean(nll, mask)

def spike_bce_loss(spike_logits: torch.Tensor, spike_targets: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(spike_logits, spike_targets, reduction='none')
    return _masked_mean(loss, mask)

def smoothness_loss(predictions: torch.Tensor, targets: torch.Tensor | None=None, mask: torch.Tensor | None=None) -> torch.Tensor:
    diffs = predictions[:, 1:, :] - predictions[:, :-1, :]
    if targets is not None:
        diffs = diffs - (targets[:, 1:, :] - targets[:, :-1, :])
    derivative_mask = None
    if mask is not None:
        derivative_mask = mask[:, 1:, :] * mask[:, :-1, :]
    return _masked_mean(diffs.square(), derivative_mask)

def multiscale_smoothness_loss(predictions: torch.Tensor, targets: torch.Tensor | None=None, mask: torch.Tensor | None=None, scales: tuple[int, ...]=(1, 2, 4, 8)) -> torch.Tensor:
    terms = []
    for scale in scales:
        if scale >= predictions.shape[1]:
            continue
        diffs = predictions[:, scale:, :] - predictions[:, :-scale, :]
        if targets is not None:
            diffs = diffs - (targets[:, scale:, :] - targets[:, :-scale, :])
        scale_mask = None
        if mask is not None:
            scale_mask = mask[:, scale:, :] * mask[:, :-scale, :]
        terms.append(_masked_mean(diffs.square(), scale_mask))
    return torch.stack(terms).mean() if terms else predictions.new_zeros(())

def frequency_consistency_loss(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
    if mask is not None:
        predictions = torch.where(mask.bool(), predictions, targets)
    pred_fft = torch.fft.rfft(predictions.float(), dim=1)
    target_fft = torch.fft.rfft(targets.float(), dim=1)
    return torch.mean((torch.abs(pred_fft) - torch.abs(target_fft)) ** 2)

def clock_drift_acceleration_loss(predictions: torch.Tensor, targets: torch.Tensor | None=None, mask: torch.Tensor | None=None) -> torch.Tensor:
    clock = predictions[:, :, 3]
    velocity = clock[:, 1:] - clock[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    if targets is not None:
        target_velocity = targets[:, 1:, 3] - targets[:, :-1, 3]
        acceleration = acceleration - (target_velocity[:, 1:] - target_velocity[:, :-1])
    acceleration_mask = None
    if mask is not None:
        clock_mask = mask[:, :, 3]
        acceleration_mask = clock_mask[:, 2:] * clock_mask[:, 1:-1] * clock_mask[:, :-2]
    return _masked_mean(acceleration.square(), acceleration_mask)

def soft_dtw_cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x ** 2).sum(dim=-1, keepdim=True)
    y_norm = (y ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y.transpose(1, 2))
    return F.relu(dist)

def soft_dtw_loss(x: torch.Tensor, y: torch.Tensor, gamma: float=0.1, pool_size: int=4) -> torch.Tensor:
    if pool_size > 1 and x.shape[1] >= pool_size * 2:
        x_in = F.avg_pool1d(x.transpose(1, 2), kernel_size=pool_size).transpose(1, 2)
        y_in = F.avg_pool1d(y.transpose(1, 2), kernel_size=pool_size).transpose(1, 2)
    else:
        x_in, y_in = (x, y)
    B, N, D = x_in.shape
    D_mat = soft_dtw_cost_matrix(x_in, y_in)
    large_val = 1000000.0
    R = torch.full((B, N + 1, N + 1), large_val, device=x.device, dtype=x.dtype)
    R[:, 0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            r0 = R[:, i - 1, j]
            r1 = R[:, i, j - 1]
            r2 = R[:, i - 1, j - 1]
            stacked = torch.stack([r0, r1, r2], dim=-1)
            softmin = -gamma * torch.logsumexp(-stacked / gamma, dim=-1)
            R[:, i, j] = D_mat[:, i - 1, j - 1] + softmin
    return R[:, N, N].mean()

def dilate_loss(predictions: torch.Tensor, targets: torch.Tensor, alpha: float=0.6, gamma: float=0.1) -> torch.Tensor:
    B, N, D = predictions.shape
    shape_loss = soft_dtw_loss(predictions, targets, gamma=gamma)
    i_idx = torch.arange(N, device=predictions.device, dtype=torch.float32).view(1, N, 1)
    j_idx = torch.arange(N, device=predictions.device, dtype=torch.float32).view(1, 1, N)
    omega = (i_idx - j_idx) ** 2 / N ** 2
    d_mat = soft_dtw_cost_matrix(predictions, targets)
    temporal_loss = (d_mat * omega).mean()
    return alpha * shape_loss + (1.0 - alpha) * temporal_loss

def composite_transformer_loss(mu: torch.Tensor, sigma: torch.Tensor, spike_probs: torch.Tensor | None, targets: torch.Tensor, spike_targets: torch.Tensor | None, target_mask: torch.Tensor | None=None, distribution: str='student_t', degrees_of_freedom: float=3.0, lambda_smooth: float=0.0, lambda_multi: float=0.0, lambda_freq: float=0.0, lambda_clock: float=0.0, lambda_spike: float=0.0, lambda_sigma: float=0.0, lambda_dilate: float=0.0, orbit_loss_weight: float=1.0, clock_loss_weight: float=1.0) -> torch.Tensor:
    if orbit_loss_weight < 0 or clock_loss_weight < 0 or orbit_loss_weight + clock_loss_weight <= 0:
        raise ValueError('orbit/clock loss weights must be non-negative with a positive sum')
    if distribution not in {'student_t', 'gaussian'}:
        raise ValueError(f'Unknown predictive distribution: {distribution}')

    def target_nll(target_slice: slice) -> torch.Tensor:
        sliced_mask = target_mask[..., target_slice] if target_mask is not None else None
        if distribution == 'student_t':
            return student_t_nll_loss(mu[..., target_slice], sigma[..., target_slice], targets[..., target_slice], sliced_mask, degrees_of_freedom)
        return gaussian_nll_loss(mu[..., target_slice], sigma[..., target_slice], targets[..., target_slice], sliced_mask)
    if mu.shape[-1] == 4:
        nll = (orbit_loss_weight * target_nll(slice(0, 3)) + clock_loss_weight * target_nll(slice(3, 4))) / (orbit_loss_weight + clock_loss_weight)
    else:
        nll = target_nll(slice(None))
    zero = mu.new_zeros(())
    spike = zero
    if lambda_spike > 0 and spike_probs is not None and (spike_targets is not None):
        spike_mask = target_mask.any(dim=-1) if target_mask is not None else None
        spike = spike_bce_loss(spike_probs, spike_targets, spike_mask)
    smooth = smoothness_loss(mu, targets, target_mask) if lambda_smooth > 0 else zero
    multi = multiscale_smoothness_loss(mu, targets, target_mask) if lambda_multi > 0 else zero
    freq = frequency_consistency_loss(mu, targets, target_mask) if lambda_freq > 0 else zero
    clock = clock_drift_acceleration_loss(mu, targets, target_mask) if lambda_clock > 0 else zero
    sigma_penalty = _masked_mean(torch.log(sigma.clamp_min(1e-05)).square(), target_mask)
    phase_loss = zero
    if lambda_dilate > 0:
        if target_mask is None:
            phase_loss = dilate_loss(mu, targets)
        else:
            fully_observed = target_mask.bool().all(dim=-1).all(dim=-1)
            if fully_observed.any():
                phase_loss = dilate_loss(mu[fully_observed], targets[fully_observed])
    total = nll + lambda_spike * spike + lambda_smooth * smooth + lambda_multi * multi + lambda_freq * freq + lambda_clock * clock + lambda_sigma * sigma_penalty + lambda_dilate * phase_loss
    return total

def diffusion_mse_loss(predicted_noise: torch.Tensor, true_noise: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
    return _masked_mean((predicted_noise - true_noise).square(), mask)
