# evaluation/sigma_loss_profile.py
"""
Post-training estimation of the *shape of the denoising loss with respect to σ*.

This mirrors the online buffer mechanism used during training
(``trainers/trainer.py`` + ``utils/schedule_controller.py``): there, the trainer
stores (σ, entropy-metric) pairs throughout training and periodically estimates
the entropy-rate profile / CDF, which ``EntropySchedulePlotCallback`` plots every
K epochs.

Here we reproduce the same construction *after* training by sweeping a held-out
(test) split: for each example we draw σ, corrupt x0, run the model, and record
the per-example entropy metric (the squared-error denoising proxy
``||sigmoid(logits) - x0||^2`` averaged over the relevant bit positions — exactly
the ``return_entropy_metric`` output of
``diffusion.continuous.losses.binary_score_interpolation_loss``).

When **cluster-bit (semantic) guidance** is enabled, every token's bitstream is
``[info | cluster]``. We additionally record the entropy metric restricted to the
information bits and to the trailing cluster ("additional semantic") bits
separately, so their σ-profiles / CDFs can be compared (do the extra semantic
bits have a different loss-vs-σ shape than the information bits?).

The entropy CDF/PDF is built with the *same* recipe as
``EntropyScheduleController.recompute_entropy_from_buffer``:

    rate_samples  = metric / σ²
    mean_rate(bin)= mean(rate_samples in log-σ bin)
    q(bin)        ∝ g(σ_mid; c, n) · mean_rate(bin)^power
    g(σ)          = σⁿ / (σⁿ + cⁿ)        (Gabriel regularizer)

so the post-hoc profile is directly comparable to the one produced during
training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast

from diffusion.continuous.logit_postprocess import _model_logits_continuous


# -----------------------------------------------------------------------------
# Bit-group masks (info vs cluster) — shared with the dual entropy scheduler.
# -----------------------------------------------------------------------------
from utils.bit_groups import bit_group_masks  # noqa: E402  (re-exported for callers)


def _generalized_regularizer(sigmas: torch.Tensor, c: float, n: float) -> torch.Tensor:
    """Gabriel gate g(σ)=σⁿ/(σⁿ+cⁿ); identical to schedule_controller."""
    c = float(max(c, 1e-12))
    n = float(n)
    x = (sigmas.clamp_min(1e-12) / c).pow(n)
    return x / (1.0 + x)


# -----------------------------------------------------------------------------
# (σ, loss) pair collection over a loader
# -----------------------------------------------------------------------------
@dataclass
class SigmaLossPairs:
    """Raw (σ, entropy-metric) pairs collected over a split, per bit group.

    ``groups`` maps a group name ("all", "info", "cluster") to a dict with CPU
    float tensors ``{"sigma": [N], "metric": [N]}``. "info"/"cluster" are present
    only under cluster-bit guidance.
    """

    groups: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    num_examples: int = 0
    sigma_sampling: str = "log-uniform"
    mc_samples: int = 1


@torch.no_grad()
def collect_sigma_loss_pairs(
    *,
    model: torch.nn.Module,
    cfg,
    data_loader,
    device: torch.device,
    sigma_sampling: str = "log-uniform",
    mc_samples: int = 1,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    max_batches: Optional[int] = None,
    progress: bool = True,
) -> SigmaLossPairs:
    """Sweep ``data_loader`` collecting (σ, entropy-metric) pairs.

    The forward path mirrors the unconditional binary training step
    (``Trainer._step_continuous``): the same self-conditioning replication and the
    same ``loss_fn(..., return_entropy_metric=True)`` entropy metric. With cluster
    guidance the metric is additionally computed restricted to the info and
    cluster bit blocks via per-position masks.
    """
    if data_loader is None:
        raise ValueError("data_loader is None")

    repr_mode = str(getattr(cfg.data, "representation", "binary")).lower()
    if repr_mode != "binary":
        raise ValueError(
            f"sigma_loss_profile currently supports representation='binary', got {repr_mode!r}"
        )

    from diffusion.continuous.processes import ContinuousForwardProcess
    from diffusion.continuous.losses import binary_score_interpolation_loss

    proc = ContinuousForwardProcess(cfg)
    sigma_sampling = str(sigma_sampling).lower().strip()

    masks = bit_group_masks(cfg, device)
    group_names = ["all"] + (["info", "cluster"] if masks is not None else [])

    sc_enabled = bool(getattr(cfg.model, "self_condition", False))
    p_sc = float(getattr(cfg.train, "self_condition_prob", 0.5))

    p_mean = float(getattr(cfg.diffusion.continuous, "p_mean", -1.2))
    p_std = float(getattr(cfg.diffusion.continuous, "p_std", 1.2))

    buffers: Dict[str, Dict[str, List[torch.Tensor]]] = {
        g: {"sigma": [], "metric": []} for g in group_names
    }
    n_examples = 0
    mc = max(1, int(mc_samples))

    model.eval()

    iterator = data_loader
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="sigma_loss_profile", leave=True)
        except Exception:
            pass

    for bi, batch in enumerate(iterator):
        if max_batches is not None and bi >= int(max_batches):
            break

        x0 = batch[0] if isinstance(batch, (tuple, list)) else batch
        B = int(x0.size(0))
        if B <= 0:
            continue

        x0_clean = x0.to(device, non_blocking=True).view(B, -1).to(torch.float32)  # [B,S]
        S = int(x0_clean.size(1))
        n_examples += B

        for _ in range(mc):
            if sigma_sampling in {"log-normal", "lognormal", "edm"}:
                sigma = proc.sample_sigma(B, strategy="log-normal", p_mean=p_mean, p_std=p_std)
            else:
                sigma = proc.sample_sigma(B, strategy="log-uniform")
            sigma = sigma.to(device).to(torch.float32)  # [B]

            xt = x0_clean + sigma.view(-1, 1) * torch.randn_like(x0_clean)

            # Self-conditioning: replicate the training forward so the only
            # difference from the reported train/val metric is the masking.
            x0_hat = None
            if sc_enabled and p_sc > 0.0:
                sc_mask = torch.rand(B, device=device) < p_sc
                if sc_mask.any():
                    x0_hat = torch.zeros_like(xt)
                    with autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                        logits_sc = _model_logits_continuous(model, cfg, xt, sigma, None)
                    if logits_sc.dim() == 3 and logits_sc.size(-1) == 1:
                        logits_sc = logits_sc.squeeze(-1)
                    sc_pred = torch.sigmoid(logits_sc.float()).detach().to(xt.dtype)
                    if sc_mask.all():
                        x0_hat.copy_(sc_pred)
                    else:
                        x0_hat[sc_mask] = sc_pred[sc_mask]

            with autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                logits = _model_logits_continuous(model, cfg, xt, sigma, x0_hat)
                for g in group_names:
                    mask_b = None
                    if g != "all":
                        mask_b = masks[g].view(1, S).expand(B, S)
                    _, metric = binary_score_interpolation_loss(
                        logits,
                        x0_clean,
                        sigma,
                        cfg,
                        return_entropy_metric=True,
                        mask=mask_b,
                    )  # metric: [B]
                    buffers[g]["sigma"].append(sigma.detach().float().cpu())
                    buffers[g]["metric"].append(metric.detach().float().cpu())

    out = SigmaLossPairs(num_examples=int(n_examples), sigma_sampling=sigma_sampling, mc_samples=mc)
    for g in group_names:
        if len(buffers[g]["sigma"]) == 0:
            continue
        out.groups[g] = {
            "sigma": torch.cat(buffers[g]["sigma"]).contiguous(),
            "metric": torch.cat(buffers[g]["metric"]).contiguous(),
        }
    return out


# -----------------------------------------------------------------------------
# Entropy profile (PDF / CDF) construction — same recipe as the trainer
# -----------------------------------------------------------------------------
@dataclass
class EntropyProfile:
    sigmas_mid: torch.Tensor   # [K]
    edges: torch.Tensor        # [K+1]
    pdf: torch.Tensor          # [K]   q(bin), the entropy-weighted sampling density
    cdf: torch.Tensor          # [K]
    mean_rate: torch.Tensor    # [K]   mean(metric/σ²) per bin (loss "rate" curve)
    mean_metric: torch.Tensor  # [K]   mean(metric) per bin (raw loss curve)
    count: torch.Tensor        # [K]   number of samples per bin
    num_bins: int
    c: float
    n: float
    power: float
    ln_mu: Optional[float] = None
    ln_std: Optional[float] = None


def build_entropy_profile(
    sigma: torch.Tensor,
    metric: torch.Tensor,
    *,
    sigma_min: float,
    sigma_max: float,
    num_bins: int = 256,
    c: float = 0.1,
    n: float = 3.0,
    power: float = 1.0,
) -> EntropyProfile:
    """Replicate ``EntropyScheduleController.recompute_entropy_from_buffer``.

    All tensors are computed on CPU in float32. Returns the per-bin mean rate and
    mean metric (the loss-vs-σ curves) alongside the entropy PDF/CDF.
    """
    sig = sigma.detach().to("cpu", dtype=torch.float32).flatten()
    metr = metric.detach().to("cpu", dtype=torch.float32).flatten()

    finite = torch.isfinite(sig) & torch.isfinite(metr)
    sig = sig[finite]
    metr = metr[finite]

    num_bins = max(2, int(num_bins))
    sigma_min = max(float(sigma_min), 1e-12)
    sigma_max = max(float(sigma_max), sigma_min * 1.0000001)
    sig = sig.clamp(min=sigma_min, max=sigma_max)

    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    edges_log = torch.linspace(log_min, log_max, num_bins + 1, dtype=torch.float32)

    bin_idx = torch.bucketize(sig.log(), edges_log, right=False) - 1
    bin_idx.clamp_(0, num_bins - 1)

    rate_samples = torch.clamp(metr / (sig * sig + 1e-12), min=0.0)

    sum_rate = torch.zeros((num_bins,), dtype=torch.float32)
    sum_metric = torch.zeros((num_bins,), dtype=torch.float32)
    count = torch.zeros((num_bins,), dtype=torch.float32)
    sum_rate.scatter_add_(0, bin_idx, rate_samples)
    sum_metric.scatter_add_(0, bin_idx, metr)
    count.scatter_add_(0, bin_idx, torch.ones_like(rate_samples))

    mean_rate = torch.clamp(sum_rate / (count + 1e-8), min=0.0)
    mean_metric = sum_metric / (count + 1e-8)

    mid_log = 0.5 * (edges_log[:-1] + edges_log[1:])
    mids = mid_log.exp().clamp(min=1e-12)
    edges_sigma = edges_log.exp().clamp(min=1e-12)

    reg = _generalized_regularizer(mids, c=c, n=n).to(torch.float32)
    rate_term = mean_rate.pow(power) if power != 0.0 else torch.ones_like(mean_rate)
    unnormalized = reg * rate_term

    good = bool(torch.isfinite(unnormalized).all()) and float(unnormalized.sum()) > 0.0
    if not good:
        pdf = torch.full((num_bins,), 1.0 / float(num_bins), dtype=torch.float32)
    else:
        pdf = unnormalized / unnormalized.sum()

    cdf = torch.cumsum(pdf, dim=0)
    cdf = torch.maximum(cdf, torch.cummax(cdf, dim=0)[0])
    cdf[-1] = 1.0

    # Diagnostic log-normal fit over log σ (same as schedule_controller).
    ln_mu = ln_std = None
    tot = float(pdf.sum())
    if math.isfinite(tot) and tot > 0.0:
        p = pdf / tot
        log_m = mids.clamp_min(1e-12).log()
        mu = (p * log_m).sum()
        var = (p * (log_m - mu).pow(2)).sum()
        ln_mu = float(mu.item())
        ln_std = float(torch.sqrt(var.clamp_min(1e-12)).item())

    return EntropyProfile(
        sigmas_mid=mids,
        edges=edges_sigma,
        pdf=pdf,
        cdf=cdf,
        mean_rate=mean_rate,
        mean_metric=mean_metric,
        count=count,
        num_bins=num_bins,
        c=float(c),
        n=float(n),
        power=float(power),
        ln_mu=ln_mu,
        ln_std=ln_std,
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_sigma_loss_profiles(
    profiles: Dict[str, EntropyProfile],
    *,
    out_path,
    title: str = "",
):
    """Plot, per bit group, the entropy CDF and the loss-vs-σ curves.

    Panels (x-axis = log10 σ for all):
      1. entropy PDF q(σ)              (the σ-sampling density)
      2. entropy CDF                   (the requested "CDF of entropy")
      3. mean rate = mean(metric/σ²)   (entropy-rate curve)
      4. mean metric = mean(loss)      (raw denoising loss curve)

    With cluster guidance the "info" and "cluster" groups are overlaid so their
    shapes can be compared directly against the combined "all" group.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(8, 14), sharex=True)
    ax_pdf, ax_cdf, ax_rate, ax_metric = axes

    color_for = {"all": "k", "info": "C0", "cluster": "C3"}

    for g, prof in profiles.items():
        x = prof.sigmas_mid.clamp(min=1e-12).log10().numpy()
        color = color_for.get(g, None)
        ax_pdf.plot(x, prof.pdf.numpy(), marker=".", lw=1, label=g, color=color)
        ax_cdf.plot(x, prof.cdf.numpy(), marker=".", lw=1, label=g, color=color)
        ax_rate.plot(x, prof.mean_rate.numpy(), marker=".", lw=1, label=g, color=color)
        ax_metric.plot(x, prof.mean_metric.numpy(), marker=".", lw=1, label=g, color=color)

    ax_pdf.set_ylabel("entropy PDF  q(bin)")
    ax_pdf.set_title(title or "σ-loss profile (test split)")
    ax_pdf.legend(loc="best")

    ax_cdf.set_ylabel("entropy CDF")
    ax_cdf.set_ylim(0.0, 1.0)
    ax_cdf.legend(loc="best")

    ax_rate.set_ylabel("mean rate  E[loss/σ²]")
    ax_rate.set_yscale("log")
    ax_rate.grid(True, which="both", ls="-", alpha=0.2)
    ax_rate.legend(loc="best")

    ax_metric.set_ylabel("mean loss  E[loss]")
    ax_metric.set_yscale("log")
    ax_metric.set_xlabel("log10 σ")
    ax_metric.grid(True, which="both", ls="-", alpha=0.2)
    ax_metric.legend(loc="best")

    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
