"""DDIM probability-flow sampler with EDM churn and self-conditioning (CoBit), with a
per-modality variant for H-CoBit.

Two loops share one denoiser call:

* ``DDIMSampler.sample``: a single information-uniform noise grid for every bit (CoBit).
* ``DDIMSampler._sample_dual``: token bits and cluster bits each follow their own
  information-uniform grid, coupled through one step index (Section 4.3 / Appendix C.2).
  Churn can be set per modality: a cluster churn rate gamma_2 confined to a window of
  the cluster schedule (the per-modality sampler of Table 1).

Churn (Karras et al., 2022): at step i the state is pushed back to
sigma_hat = (1 + gamma_i) sigma_i before the ODE step, with
gamma_i = min(S_churn / N, sqrt(2) - 1) inside a sigma window read off the entropy CDF.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from tqdm import tqdm

from diffusion.continuous.logit_postprocess import _model_logits_continuous
from utils.bit_groups import bit_group_masks


def _infer_model_device(model, cfg_device: str) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(cfg_device)


def logits_to_x0_hat(logits: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Bitwise clean-data estimate D(x, sigma) = sigmoid(logits), logits [B, S]."""
    if logits.dim() != 2:
        raise ValueError(f"expected bit logits [B,S], got {tuple(logits.shape)}")
    return torch.sigmoid(logits.float()).to(dtype=dtype)


# -----------------------------------------------------------------------------
# EDM churn
# -----------------------------------------------------------------------------

@dataclass
class _StochasticSamplerCfg:
    enabled: bool
    s_churn: float = 0.0
    s_noise: float = 1.0
    s_tmin: float = 0.0
    s_tmax: float = float("inf")
    window_mode: str = "deterministic"
    # Per-modality churn (per-modality schedule only). None => the cluster bits share
    # s_churn and the token-bit window.
    s_churn_cluster: Optional[float] = None
    s_tmin_cluster: Optional[float] = None
    s_tmax_cluster: Optional[float] = None


def _clamp_prob01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _resolve_sigma_bounds(cfg, *, sigma_min_override, sigma_max_override) -> tuple[float, float]:
    sigma_min = float(sigma_min_override) if sigma_min_override is not None else float(cfg.diffusion.continuous.sigma_min)
    sigma_max = float(sigma_max_override) if sigma_max_override is not None else float(cfg.diffusion.continuous.sigma_max)
    if sigma_max < sigma_min:
        sigma_max, sigma_min = sigma_min, sigma_max
    return sigma_min, sigma_max


def _compute_edm_gamma(sigma_cur, *, num_intervals: int, s_churn: float, s_tmin: float, s_tmax: float) -> float:
    """gamma_i = min(S_churn / N, sqrt(2) - 1) if sigma_i in [S_tmin, S_tmax], else 0."""
    s = float(sigma_cur.item()) if isinstance(sigma_cur, torch.Tensor) else float(sigma_cur)
    if s < float(s_tmin) or s > float(s_tmax) or num_intervals <= 0:
        return 0.0
    return max(0.0, min(float(s_churn) / float(num_intervals), math.sqrt(2.0) - 1.0))


def _apply_edm_churn(x, sigma_cur, *, gamma, s_noise: float):
    """sigma_hat = (1 + gamma) sigma; x_hat = x + sqrt(sigma_hat^2 - sigma^2) * S_noise * eps.

    ``gamma`` is a float, or a per-position tensor broadcastable to ``sigma_cur`` ([1, S])
    for per-modality churn. One ``randn_like(x)`` draw either way; none when gamma == 0.
    """
    if isinstance(gamma, torch.Tensor):
        if bool((gamma <= 0.0).all()):
            return x, sigma_cur
        gamma = gamma.to(device=sigma_cur.device, dtype=sigma_cur.dtype)
    elif gamma <= 0.0:
        return x, sigma_cur
    sigma_hat = sigma_cur * (1.0 + gamma)
    sigma_delta = (sigma_hat.square() - sigma_cur.square()).clamp_min(0.0).sqrt()
    eps = torch.randn_like(x) * float(s_noise)
    return x + sigma_delta.to(device=x.device, dtype=x.dtype) * eps, sigma_hat


# -----------------------------------------------------------------------------
# Prompt conditioning (GSM8K): clean prompt bits are re-imposed after every step
# -----------------------------------------------------------------------------

def _clamp_mask_(x: torch.Tensor, full: torch.Tensor, mask: Optional[torch.Tensor]) -> None:
    if mask is None or not bool(mask.any().item()):
        return
    x[mask] = full[mask]


def _zero_mask_(d: torch.Tensor, mask: Optional[torch.Tensor]) -> None:
    if mask is None or not bool(mask.any().item()):
        return
    d[mask] = 0.0


def _score_from_probs(probs: torch.Tensor, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Score (D(x, sigma) - x) / sigma^2; sigma is a scalar or per-position [1, S] / [B, S]."""
    if sigma.device != x_t.device:
        sigma = sigma.to(device=x_t.device)
    if sigma.dim() >= 2:
        sigma2 = (sigma ** 2).to(torch.float32)
    else:
        if sigma.dim() == 0:
            sigma = sigma.expand(x_t.size(0))
        sigma2 = (sigma ** 2).view(-1, 1).to(torch.float32)
    return (probs.to(torch.float32) - x_t.to(torch.float32)) / sigma2


def _build_prompt_conditioning(B, S, device, prefix_full, prefix_mask):
    """-> (enabled, prefix_full [B,S], prefix_mask [B,S] bool)."""
    if prefix_full is None and prefix_mask is None:
        return False, None, None
    if prefix_full is None or prefix_mask is None:
        raise ValueError("pass both conditioning_prefix_full and cond_prefix_mask, or neither")
    pf = prefix_full.to(device=device, dtype=torch.float32)
    pm = prefix_mask.to(device=device, dtype=torch.bool)
    if pf.dim() == 1:
        pf = pf.view(1, S).expand(B, S).contiguous()
    if pm.dim() == 1:
        pm = pm.view(1, S).expand(B, S).contiguous()
    if tuple(pf.shape) != (B, S) or tuple(pm.shape) != (B, S):
        raise ValueError(f"prompt must be [B,S]=({B},{S}); got {tuple(pf.shape)} / {tuple(pm.shape)}")
    if not bool(pm.any().item()):
        return False, None, None
    return True, pf, pm


# -----------------------------------------------------------------------------
# Information-uniform noise grids
# -----------------------------------------------------------------------------

class SigmaSchedule:
    """Noise grids placed uniformly in the CDF of an entropy-rate density (Appendix B.5)."""

    def __init__(self, process, cfg, device: torch.device):
        self.process = process
        self.cfg = cfg
        self.device = device

    def _load_entropy_tables(self, *, entropy_run_dir):
        """(pdf, cdf, sigmas) from entropy_{pdf,cdf,sigmas}.pt, or Nones if absent."""
        d = Path(entropy_run_dir)
        paths = [d / f"entropy_{t}.pt" for t in ("pdf", "cdf", "sigmas")]
        if not all(p.exists() for p in paths):
            return None, None, None
        return tuple(torch.load(p, map_location=self.device, weights_only=True).to(self.device) for p in paths)

    def entropy_quantile(self, q: float, *, entropy_run_dir=None, cdf=None, sigmas=None) -> Optional[float]:
        """sigma at quantile q of an entropy CDF (sigma table ascending)."""
        if cdf is None or sigmas is None:
            _, cdf, sigmas = self._load_entropy_tables(entropy_run_dir=entropy_run_dir)
        if cdf is None or sigmas is None or cdf.numel() == 0:
            return None
        cdf = cdf.to(self.device).float().clone()
        sig = sigmas.to(self.device).float()
        cdf[-1] = 1.0
        q_t = torch.tensor(_clamp_prob01(q), device=self.device, dtype=torch.float32)
        i = int(torch.searchsorted(cdf, q_t, right=False).clamp(0, cdf.numel() - 1).item())
        if i == 0:
            return float(sig[0].item())
        w = (q_t - cdf[i - 1]) / (cdf[i] - cdf[i - 1]).clamp_min(1e-20)
        return float((sig[i - 1] + w * (sig[i] - sig[i - 1])).item())

    def resolve_stochastic_cfg(
        self,
        *,
        entropy_run_dir=None,
        sigma_min_override=None,
        sigma_max_override=None,
        window_cdf=None,
        window_sigmas=None,
        window_cdf_cluster=None,
        window_sigmas_cluster=None,
    ) -> _StochasticSamplerCfg:
        """Churn settings from cfg.evaluation.stochastic.

        The churn window [S_tmin, S_tmax] is the sigma range between two quantiles of the
        entropy CDF (``window_cdf``/``window_sigmas`` when given, else the tables in
        ``entropy_run_dir``). With the per-modality schedule, cluster-specific quantiles
        (``entropy_quantile_{lo,hi}_cluster``) are read off the CLUSTER CDF.
        """
        st = getattr(getattr(self.cfg, "evaluation", None), "stochastic", None)
        if st is None or not bool(getattr(st, "enabled", False)):
            return _StochasticSamplerCfg(enabled=False)
        window_mode = str(getattr(st, "window_mode", "entropy_cdf")).lower().strip()
        if window_mode in {"deterministic", "off", "none"}:
            return _StochasticSamplerCfg(enabled=False)
        if window_mode != "entropy_cdf":
            raise ValueError(f"unsupported evaluation.stochastic.window_mode={window_mode!r}")

        sigma_min, sigma_max = _resolve_sigma_bounds(
            self.cfg, sigma_min_override=sigma_min_override, sigma_max_override=sigma_max_override
        )
        s_churn = max(0.0, float(getattr(st, "s_churn", 0.0)))
        s_noise = max(0.0, float(getattr(st, "s_noise", 1.0)))
        s_churn_cluster = getattr(st, "s_churn_cluster", None)
        if s_churn_cluster is not None:
            s_churn_cluster = max(0.0, float(s_churn_cluster))
        eff_churn = max(s_churn, s_churn_cluster or 0.0)

        def window(q_lo, q_hi, cdf, sigmas):
            q_lo, q_hi = _clamp_prob01(q_lo), _clamp_prob01(q_hi)
            if q_hi < q_lo:
                q_lo, q_hi = q_hi, q_lo
            lo = self.entropy_quantile(q_lo, entropy_run_dir=entropy_run_dir, cdf=cdf, sigmas=sigmas)
            hi = self.entropy_quantile(q_hi, entropy_run_dir=entropy_run_dir, cdf=cdf, sigmas=sigmas)
            if lo is None or hi is None:
                return None
            lo = max(sigma_min, min(float(lo), sigma_max))
            hi = max(sigma_min, min(float(hi), sigma_max))
            return (hi, lo) if hi < lo else (lo, hi)

        w = window(getattr(st, "entropy_quantile_lo", 0.10), getattr(st, "entropy_quantile_hi", 0.90),
                   window_cdf, window_sigmas)
        if w is None:
            if str(getattr(st, "entropy_fallback", "deterministic")).lower() != "deterministic":
                raise ValueError("only entropy_fallback='deterministic' is supported")
            return _StochasticSamplerCfg(enabled=False)

        wc = None
        q_lo_c = getattr(st, "entropy_quantile_lo_cluster", None)
        q_hi_c = getattr(st, "entropy_quantile_hi_cluster", None)
        if None not in (q_lo_c, q_hi_c, window_cdf_cluster, window_sigmas_cluster):
            wc = window(q_lo_c, q_hi_c, window_cdf_cluster, window_sigmas_cluster)
        return _StochasticSamplerCfg(
            enabled=(eff_churn > 0.0 and w[1] >= w[0]),
            s_churn=s_churn,
            s_noise=s_noise,
            s_tmin=w[0],
            s_tmax=w[1],
            window_mode=window_mode,
            s_churn_cluster=s_churn_cluster,
            s_tmin_cluster=None if wc is None else wc[0],
            s_tmax_cluster=None if wc is None else wc[1],
        )

    @staticmethod
    def _interp1d_monotone(x, y, xq):
        xq = xq.clamp(min=x[0], max=x[-1])
        idx = torch.searchsorted(x, xq, right=False).clamp(min=1, max=x.numel() - 1)
        w = (xq - x[idx - 1]) / (x[idx] - x[idx - 1]).clamp_min(1e-20)
        return y[idx - 1] + w * (y[idx] - y[idx - 1])

    def _inverse_cdf_sample_truncated(self, sigmas_base, cdf, *, N: int, sigma_min: float, sigma_max: float):
        """N sigmas, descending from sigma_max to sigma_min, uniform in the CDF."""
        sig = sigmas_base.to(self.device).float()
        F = cdf.to(self.device).float()
        if sig.numel() < 2:
            out = torch.full((N,), float(sigma_max), device=self.device, dtype=torch.float32)
            out[-1] = float(sigma_min)
            return out

        table_min, table_max = float(sig[0].item()), float(sig[-1].item())
        sigma_min_eff = float(max(sigma_min, table_min))
        sigma_max_eff = float(min(sigma_max, table_max))
        if sigma_min_eff > sigma_max_eff:
            val = float(max(min(sigma_max, table_max), table_min))
            return torch.full((N,), val, device=self.device, dtype=torch.float32)

        F_min = self._interp1d_monotone(sig, F, torch.tensor(sigma_min_eff, device=self.device))
        F_max = self._interp1d_monotone(sig, F, torch.tensor(sigma_max_eff, device=self.device))
        if float((F_max - F_min).abs().item()) < 1e-12:
            lo = torch.log(torch.tensor(sigma_min_eff, device=self.device))
            hi = torch.log(torch.tensor(sigma_max_eff, device=self.device))
            s = torch.flip(torch.exp(torch.linspace(lo, hi, N, device=self.device, dtype=torch.float32)), dims=[0])
            s[0], s[-1] = sigma_max_eff, sigma_min_eff
            return s

        u = torch.linspace(0.0, 1.0, N, device=self.device, dtype=torch.float32)
        u = (F_min + u * (F_max - F_min)).clamp(min=F[0].item(), max=1.0 - 1e-7)
        idx = torch.searchsorted(F, u, right=False).clamp(min=1, max=F.numel() - 1)
        w = (u - F[idx - 1]) / (F[idx] - F[idx - 1]).clamp_min(1e-20)
        sig_forward = sig[idx - 1] + w * (sig[idx] - sig[idx - 1])
        sig_forward[0], sig_forward[-1] = sigma_min_eff, sigma_max_eff
        sigmas = torch.flip(sig_forward, dims=[0])
        sigmas[0], sigmas[-1] = sigma_max_eff, sigma_min_eff
        return sigmas

    def prepare(self, *, num_steps, entropy_run_dir, sigma_min_override=None, sigma_max_override=None, **_):
        """Single information-uniform grid from entropy_{cdf,sigmas}.pt in entropy_run_dir."""
        sigma_min, sigma_max = _resolve_sigma_bounds(
            self.cfg, sigma_min_override=sigma_min_override, sigma_max_override=sigma_max_override
        )
        _, cdf, sigmas_base = self._load_entropy_tables(entropy_run_dir=entropy_run_dir)
        if cdf is None:
            raise FileNotFoundError(f"entropy_{{pdf,cdf,sigmas}}.pt not found in {entropy_run_dir}")
        cdf = cdf.to(self.device).clone().float()
        cdf[-1] = 1.0
        return self._inverse_cdf_sample_truncated(
            sigmas_base.to(self.device).float(), cdf, N=int(num_steps), sigma_min=sigma_min, sigma_max=sigma_max
        )

    def _load_profile_cdf(self, path: Path):
        """(cdf, sigmas) from a sigma_loss_profile group file (entropy_profile_<group>.pt)."""
        d = torch.load(Path(path), map_location=self.device, weights_only=False)
        cdf = d["cdf"].to(self.device).float().clone()
        cdf[-1] = 1.0
        return cdf, d["sigmas"].to(self.device).float()

    def _load_group_cdf(self, *, profile_path, entropy_run_dir, group: str):
        """(cdf, sigmas) of one modality: an explicit sigma_loss_profile file, else the
        per-modality tables the trainer persists (entropy_{cdf,sigmas}_<group>.pt)."""
        if profile_path:
            return self._load_profile_cdf(Path(profile_path))
        run_dir = Path(entropy_run_dir)
        cdf_p, sig_p = run_dir / f"entropy_cdf_{group}.pt", run_dir / f"entropy_sigmas_{group}.pt"
        if not (cdf_p.exists() and sig_p.exists()):
            raise FileNotFoundError(f"no {group} entropy profile: set a profile path or provide {cdf_p}")
        cdf = torch.load(cdf_p, map_location=self.device, weights_only=True).to(self.device).float().clone()
        cdf[-1] = 1.0
        sig = torch.load(sig_p, map_location=self.device, weights_only=True).to(self.device).float()
        return cdf, sig

    def prepare_dual(self, *, num_steps, info_profile=None, cluster_profile=None, entropy_run_dir=None,
                     sigma_min_override=None, sigma_max_override=None):
        """Token and cluster grids from their own CDFs on one shared base time
        u = linspace(0, 1, N), so step k is a single clock for both modalities (eq. 11).

        Returns (sigmas_info[N], sigmas_cluster[N], cdf_info, sig_info, cdf_cluster, sig_cluster).
        """
        sigma_min, sigma_max = _resolve_sigma_bounds(
            self.cfg, sigma_min_override=sigma_min_override, sigma_max_override=sigma_max_override
        )
        cdf_i, sig_i = self._load_group_cdf(profile_path=info_profile, entropy_run_dir=entropy_run_dir, group="info")
        cdf_c, sig_c = self._load_group_cdf(profile_path=cluster_profile, entropy_run_dir=entropy_run_dir, group="cluster")
        N = int(num_steps)
        grid_i = self._inverse_cdf_sample_truncated(sig_i, cdf_i, N=N, sigma_min=sigma_min, sigma_max=sigma_max)
        grid_c = self._inverse_cdf_sample_truncated(sig_c, cdf_c, N=N, sigma_min=sigma_min, sigma_max=sigma_max)
        return grid_i, grid_c, cdf_i, sig_i, cdf_c, sig_c


# -----------------------------------------------------------------------------
# Sampler
# -----------------------------------------------------------------------------

class DDIMSampler:
    """First-order probability-flow ODE sampler with EDM churn and self-conditioning."""

    def __init__(self, model, forward_process, cfg):
        self.model = model
        self.process = forward_process
        self.cfg = cfg
        self.device = _infer_model_device(model, cfg.device)
        self.sigmas = SigmaSchedule(self.process, self.cfg, self.device)
        self.sc_enabled = bool(getattr(self.cfg.model, "self_condition", False))
        self.data_center = float(getattr(self.cfg.diffusion.continuous, "data_center", 0.5))

    def _denoise(self, x, sigma_B, x0_prev, **sigma_pos):
        cond_in = x0_prev if self.sc_enabled else torch.zeros_like(x)
        logits = _model_logits_continuous(self.model, self.cfg, x, sigma_B, cond_in, **sigma_pos)
        return logits_to_x0_hat(logits, dtype=x.dtype)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        seq_len: int,
        *,
        schedule: str = "entropic",
        num_steps: int,
        entropy_run_dir=None,
        sigma_min_override: Optional[float] = None,
        sigma_max_override: Optional[float] = None,
        conditioning_prefix_full: Optional[torch.Tensor] = None,
        cond_prefix_mask: Optional[torch.Tensor] = None,
        return_probs: bool = False,
        progress: bool = True,
    ):
        """Generate [B, S] bits. Returns x, or (x, final denoised probabilities)."""
        B, S = int(num_samples), int(seq_len)
        cond_enabled, prefix_full, prefix_mask = _build_prompt_conditioning(
            B, S, self.device, conditioning_prefix_full, cond_prefix_mask
        )
        dual_cfg = getattr(getattr(self.cfg, "evaluation", object()), "dual_schedule", None)
        if bool(getattr(dual_cfg, "enabled", False)) or str(schedule).lower() == "entropic_dual":
            return self._sample_dual(
                B, S, num_steps=num_steps, entropy_run_dir=entropy_run_dir,
                sigma_min_override=sigma_min_override, sigma_max_override=sigma_max_override,
                return_probs=return_probs, progress=progress, dual_cfg=dual_cfg,
                cond_enabled=cond_enabled, prefix_full=prefix_full, prefix_mask=prefix_mask,
            )

        sigmas = self.sigmas.prepare(
            num_steps=num_steps, entropy_run_dir=entropy_run_dir,
            sigma_min_override=sigma_min_override, sigma_max_override=sigma_max_override,
        )
        stoch_cfg = self.sigmas.resolve_stochastic_cfg(
            entropy_run_dir=entropy_run_dir,
            sigma_min_override=sigma_min_override, sigma_max_override=sigma_max_override,
        )

        # With a prompt (GSM8K), the clean prompt bits are re-imposed after every step.
        x = torch.randn(B, S, device=self.device, dtype=torch.float32) * sigmas[0] + self.data_center
        x0_hat = torch.zeros_like(x) if self.sc_enabled else None
        if cond_enabled:
            _clamp_mask_(x, prefix_full, prefix_mask)
            if x0_hat is not None:
                _clamp_mask_(x0_hat, prefix_full, prefix_mask)
        indices = range(len(sigmas) - 1)
        if progress:
            indices = tqdm(indices, desc="DDIM Sampler", leave=False)
        for i in indices:
            sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
            if cond_enabled:
                _clamp_mask_(x, prefix_full, prefix_mask)
            gamma_i = _compute_edm_gamma(
                sigma_cur, num_intervals=max(1, len(sigmas) - 1), s_churn=stoch_cfg.s_churn,
                s_tmin=stoch_cfg.s_tmin, s_tmax=stoch_cfg.s_tmax,
            ) if stoch_cfg.enabled else 0.0
            x_state, sigma_state = _apply_edm_churn(x, sigma_cur, gamma=gamma_i, s_noise=stoch_cfg.s_noise)
            if cond_enabled:
                _clamp_mask_(x_state, prefix_full, prefix_mask)
                if x0_hat is not None:
                    _clamp_mask_(x0_hat, prefix_full, prefix_mask)

            probs = self._denoise(x_state, sigma_state.expand(B), x0_hat)
            if cond_enabled:
                _clamp_mask_(probs, prefix_full, prefix_mask)
            d_cur = -sigma_state * _score_from_probs(probs, x_state, sigma_state)
            if cond_enabled:
                _zero_mask_(d_cur, prefix_mask)
            x = x_state + (sigma_next - sigma_state) * d_cur
            if cond_enabled:
                _clamp_mask_(x, prefix_full, prefix_mask)
            x0_hat = probs  # self-conditioning carries the step's own estimate

        if return_probs:
            probs = self._denoise(x, sigmas[-1].expand(B), x0_hat)
            if cond_enabled:
                _clamp_mask_(probs, prefix_full, prefix_mask)
            return x, probs
        return x

    @torch.no_grad()
    def _sample_dual(
        self,
        num_samples: int,
        seq_len: int,
        *,
        num_steps: int,
        entropy_run_dir=None,
        sigma_min_override: Optional[float] = None,
        sigma_max_override: Optional[float] = None,
        return_probs: bool = False,
        progress: bool = True,
        dual_cfg=None,
        cond_enabled: bool = False,
        prefix_full: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ):
        """DDIM loop with a per-modality noise schedule.

        Token-bit and cluster-bit positions follow their own sigma grid. The model's time
        embedding receives the token sigma (scalar); the per-position sigma drives the churn,
        the ODE step, the score and, with ``dual_schedule.per_position_cin`` /
        ``per_position_matched_filter``, the input scaling and the matched-filter output.

        With a prompt, the clean prompt bits are re-imposed after every step, as in training.
        """
        B, S = int(num_samples), int(seq_len)
        per_pos_cin = bool(getattr(dual_cfg, "per_position_cin", False))
        per_pos_mf = bool(getattr(dual_cfg, "per_position_matched_filter", False))

        grid_i, grid_c, cdf_i, sig_i, cdf_c, sig_c = self.sigmas.prepare_dual(
            num_steps=num_steps,
            info_profile=getattr(dual_cfg, "info_profile", None),
            cluster_profile=getattr(dual_cfg, "cluster_profile", None),
            entropy_run_dir=entropy_run_dir,
            sigma_min_override=sigma_min_override,
            sigma_max_override=sigma_max_override,
        )
        L = int(grid_i.numel())
        # The token-bit churn window comes from the token CDF; a cluster window, when
        # set, from the cluster CDF.
        stoch_cfg = self.sigmas.resolve_stochastic_cfg(
            entropy_run_dir=entropy_run_dir,
            sigma_min_override=sigma_min_override, sigma_max_override=sigma_max_override,
            window_cdf=cdf_i, window_sigmas=sig_i,
            window_cdf_cluster=cdf_c, window_sigmas_cluster=sig_c,
        )

        masks = bit_group_masks(self.cfg, self.device)
        if masks is None:
            raise RuntimeError("the per-modality schedule requires cluster bits")
        m_info = masks["info"].view(1, S).to(torch.float32)
        m_cluster = masks["cluster"].view(1, S).to(torch.float32)

        def sigma_pos_at(k: int) -> torch.Tensor:
            return grid_i[k] * m_info + grid_c[k] * m_cluster  # [1, S]

        def sigma_kwargs(sig_pos):
            return dict(
                sigma_pos=sig_pos.expand(B, S) if per_pos_cin else None,
                sigma_pos_mf=sig_pos.expand(B, S) if per_pos_mf else None,
            )

        num_intervals = max(1, L - 1)
        x = torch.randn(B, S, device=self.device, dtype=torch.float32) * sigma_pos_at(0) + self.data_center
        if cond_enabled:
            _clamp_mask_(x, prefix_full, prefix_mask)
        x0_hat = torch.zeros_like(x) if self.sc_enabled else None

        indices = range(L - 1)
        if progress:
            indices = tqdm(indices, desc="DDIM Dual", leave=False)
        for i in indices:
            sig_cur_pos, sig_next_pos = sigma_pos_at(i), sigma_pos_at(i + 1)
            sig_info_cur = grid_i[i]

            gamma_info_i = _compute_edm_gamma(
                sig_info_cur, num_intervals=num_intervals, s_churn=stoch_cfg.s_churn,
                s_tmin=stoch_cfg.s_tmin, s_tmax=stoch_cfg.s_tmax,
            ) if stoch_cfg.enabled else 0.0

            has_cluster_window = stoch_cfg.s_tmin_cluster is not None and stoch_cfg.s_tmax_cluster is not None
            if stoch_cfg.enabled and (stoch_cfg.s_churn_cluster is not None or has_cluster_window):
                # Per-modality churn: the cluster bits get their own rate and/or window,
                # gated on the cluster schedule -> a [1, S] gamma.
                gamma_cluster_i = _compute_edm_gamma(
                    grid_c[i],
                    num_intervals=num_intervals,
                    s_churn=stoch_cfg.s_churn_cluster if stoch_cfg.s_churn_cluster is not None else stoch_cfg.s_churn,
                    s_tmin=stoch_cfg.s_tmin_cluster if has_cluster_window else stoch_cfg.s_tmin,
                    s_tmax=stoch_cfg.s_tmax_cluster if has_cluster_window else stoch_cfg.s_tmax,
                )
                gamma = gamma_info_i * m_info + gamma_cluster_i * m_cluster
            else:
                gamma = gamma_info_i

            x_state, sig_state_pos = _apply_edm_churn(x, sig_cur_pos, gamma=gamma, s_noise=stoch_cfg.s_noise)
            sig_info_state = sig_info_cur * (1.0 + gamma_info_i)  # token sigma after churn: time embedding
            if cond_enabled:
                _clamp_mask_(x_state, prefix_full, prefix_mask)
                if self.sc_enabled:
                    _clamp_mask_(x0_hat, prefix_full, prefix_mask)

            probs = self._denoise(x_state, sig_info_state.reshape(1).expand(B), x0_hat, **sigma_kwargs(sig_state_pos))
            if cond_enabled:
                _clamp_mask_(probs, prefix_full, prefix_mask)

            d_cur = -sig_state_pos * _score_from_probs(probs, x_state, sig_state_pos)
            if cond_enabled:
                _zero_mask_(d_cur, prefix_mask)
            x = x_state + (sig_next_pos - sig_state_pos) * d_cur
            if cond_enabled:
                _clamp_mask_(x, prefix_full, prefix_mask)
            x0_hat = probs

        if return_probs:
            if cond_enabled and self.sc_enabled:
                _clamp_mask_(x0_hat, prefix_full, prefix_mask)
            probs = self._denoise(x, grid_i[-1].reshape(1).expand(B), x0_hat, **sigma_kwargs(sigma_pos_at(L - 1)))
            if cond_enabled:
                _clamp_mask_(probs, prefix_full, prefix_mask)
            return x, probs
        return x
