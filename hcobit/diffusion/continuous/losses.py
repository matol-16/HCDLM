"""CoBit's training objective: EDM-weighted squared error of the bitwise clean-data
estimate D(x_sigma, sigma) = sigmoid(logits) (equation 2), with H-CoBit's per-modality
loss weight lambda_2 (equation 5)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def _sigma_weight(cfg, sigma: torch.Tensor, ndim: int) -> torch.Tensor:
    """EDM weight (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2; sigma is [B] or per-position [B,S]."""
    sigma2 = sigma.to(torch.float32) ** 2
    if sigma2.dim() >= 2:
        while sigma2.dim() < ndim:
            sigma2 = sigma2.unsqueeze(-1)
    else:
        sigma2 = sigma2.view(-1, *([1] * (ndim - 1)))
    weighting = str(getattr(cfg.train, "loss_weighting", "edm")).lower()
    if weighting != "edm":
        raise ValueError(f"unsupported loss_weighting={weighting!r}")
    sd2 = float(cfg.diffusion.continuous.sigma_data) ** 2
    return (sigma2 + sd2) / (sigma2 * sd2)


def _mask_3d(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return (mask.unsqueeze(-1) if mask.dim() == 2 else mask).to(dtype=ref.dtype, device=ref.device)


def _weighted_mean(weight, per_pos, mask):
    """Plain mean over all (masked) bits: token and cluster bits pooled together."""
    if mask is not None:
        m = _mask_3d(mask, per_pos)
        return (weight * per_pos * m).sum() / m.sum().clamp_min(1.0)
    return (weight * per_pos).mean()


def _grouped_weighted_mean(weight, per_pos, mask, group_norm: Dict[str, Any]):
    """L_token + lambda_2 * L_cluster, each block averaged over its own (masked) bits."""
    wl = weight * per_pos
    info_m = _mask_3d(group_norm["info_mask"], per_pos)
    cluster_m = _mask_3d(group_norm["cluster_mask"], per_pos)
    if mask is not None:
        base = _mask_3d(mask, per_pos)
        info_m, cluster_m = info_m * base, cluster_m * base
    info_m, cluster_m = info_m.expand_as(wl), cluster_m.expand_as(wl)
    l_info = (wl * info_m).sum() / info_m.sum().clamp_min(1.0)
    l_cluster = (wl * cluster_m).sum() / cluster_m.sum().clamp_min(1.0)
    return l_info + float(group_norm["alpha"]) * l_cluster


def binary_score_interpolation_loss(
    logits: torch.Tensor,
    x0: torch.Tensor,
    sigma: torch.Tensor,
    cfg,
    return_entropy_metric: bool = False,
    mask: Optional[torch.Tensor] = None,
    group_norm: Optional[Dict[str, Any]] = None,
):
    """Loss (and optionally the per-example squared error used to fit the noise schedule).

    ``group_norm`` (info_mask, cluster_mask, alpha) switches the reduction to
    L_token + alpha * L_cluster; the returned entropy metric is always pooled.
    """
    if str(getattr(cfg.train, "loss_type", "binary_sm")).lower() != "binary_sm":
        raise ValueError("only loss_type='binary_sm' is supported")
    if logits.dim() == 2:
        logits = logits.unsqueeze(-1)
    b = logits.size(0)
    target = x0.view_as(logits).to(torch.float32)
    weight = _sigma_weight(cfg, sigma, ndim=3)
    probs = torch.sigmoid(logits.to(torch.float32))
    per_pos = F.mse_loss(probs, target, reduction="none")
    if group_norm is not None:
        mean_loss = _grouped_weighted_mean(weight, per_pos, mask, group_norm)
    else:
        mean_loss = _weighted_mean(weight, per_pos, mask)
    if not return_entropy_metric:
        return mean_loss

    sq_err = (probs - target) ** 2
    if mask is not None:
        m = _mask_3d(mask, sq_err)
        entropy_metric = (sq_err * m).view(b, -1).sum(dim=1) / m.view(b, -1).sum(dim=1).clamp_min(1.0)
    else:
        entropy_metric = sq_err.view(b, -1).mean(dim=1)
    return mean_loss, entropy_metric
