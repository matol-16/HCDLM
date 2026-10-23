"""CoBit's matched-filter residual output: logits = network output + (x_t - c) / sigma^2."""
from __future__ import annotations

from typing import Optional

import torch


def _model_logits_continuous(
    model,
    cfg,
    x_t: torch.Tensor,
    sigma: torch.Tensor,
    x0_hat: Optional[torch.Tensor],
    sigma_pos: Optional[torch.Tensor] = None,
    sigma_pos_mf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Bit logits [B, S] of the denoiser at (x_t, sigma), self-conditioned on x0_hat.

    The analytic matched-filter term scale * (x_t - center) / sigma^2 (clipped) is added
    to the network output outside the compiled model. Under the per-modality schedule,
    ``sigma_pos`` ([B, S]) is the per-position sigma for the input scaling c_in, and
    ``sigma_pos_mf`` ([B, S]) the per-position sigma of the matched filter. The scalar
    ``sigma`` (token-bit sigma) always drives the time embedding.
    """
    mode = str(getattr(cfg.model, "continuous_logit_scaling", "none")).lower()
    if mode not in {"matched_filter", "matched_filter_residual"}:
        raise ValueError(f"unsupported continuous_logit_scaling={mode!r}")

    logits = model(x_t, sigma, x0_hat, **({"sigma_pos": sigma_pos} if sigma_pos is not None else {}))
    if logits.dim() == 3 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    if logits.dim() != 2:
        raise ValueError(f"expected bit logits [B,S] or [B,S,1], got {tuple(logits.shape)}")

    center = float(getattr(cfg.model, "matched_filter_center", cfg.diffusion.continuous.data_center))
    scale = float(getattr(cfg.model, "matched_filter_scale", 1.0))
    clip = getattr(cfg.model, "matched_filter_clip", None)
    x_f32 = x_t.to(torch.float32)
    if sigma_pos_mf is not None:
        if sigma_pos_mf.shape != x_f32.shape:
            raise ValueError(f"sigma_pos_mf {tuple(sigma_pos_mf.shape)} must match x_t {tuple(x_t.shape)}")
        sigma2 = sigma_pos_mf.to(torch.float32).square().clamp_min(1e-12)
    else:
        sigma2 = sigma.to(torch.float32).square().view(-1, 1).clamp_min(1e-12)
    mf = scale * (x_f32 - center) / sigma2
    if clip is not None:
        mf = mf.clamp(-float(clip), float(clip))
    return logits + mf.to(dtype=logits.dtype)
