"""Resolve cfg.evaluation.sampling_sweep.specs into concrete sampling specs.

Each spec fixes a sampler, a target NFE, the terminal sigma and the churn settings. Its
tag names it in results.csv and in the text cache, and seeds its generation, so the tag
format must not change: e.g.

    ddim_entropic_dual_scr-carry_tnfe-64_anfe-64_steps-63_gs0.0_term-1.10_stoch-ch2.52-clch25.2-n1.003-q0_1-qc0_0.5
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

_STOCH_SPEC_KEYS = (
    "stochastic_enabled",
    "s_churn",
    "s_churn_cluster",
    "s_noise",
    "window_mode",
    "entropy_quantile_lo",
    "entropy_quantile_hi",
    "entropy_quantile_lo_cluster",
    "entropy_quantile_hi_cluster",
    "s_tmin",
    "s_tmax",
    "entropy_fallback",
)


def steps_for_target_nfe(target_nfe: int) -> tuple[int, int]:
    """(ODE steps, actual NFE): one denoiser call per step (self-conditioning carries the
    previous estimate) plus the final denoise."""
    num_steps = max(1, int(target_nfe) - 1)
    return num_steps, num_steps + 1


def _stochastic_tag_suffix(spec: Dict[str, Any]) -> str | None:
    if not bool(spec.get("stochastic_enabled", False)):
        return None
    s_churn = float(spec.get("s_churn", 0.0))
    s_noise = float(spec.get("s_noise", 1.0))
    scc = spec.get("s_churn_cluster")
    clch = "" if scc is None else f"-clch{float(scc):g}"
    qc_lo, qc_hi = spec.get("entropy_quantile_lo_cluster"), spec.get("entropy_quantile_hi_cluster")
    clwin = "" if (qc_lo is None or qc_hi is None) else f"-qc{float(qc_lo):g}_{float(qc_hi):g}"
    window_mode = str(spec.get("window_mode", "entropy_cdf")).lower()
    if window_mode != "entropy_cdf":
        raise ValueError(f"unsupported window_mode={window_mode!r}")
    q_lo = float(spec.get("entropy_quantile_lo", 0.10))
    q_hi = float(spec.get("entropy_quantile_hi", 0.90))
    return f"stoch-ch{s_churn:g}{clch}-n{s_noise:g}-q{q_lo:g}_{q_hi:g}{clwin}"


def _tag_for_spec(spec: Dict[str, Any]) -> str:
    parts = [
        str(spec["sampler_name"]),
        f"scr-{spec['sc_refresh_mode']}",
        f"tnfe-{int(spec['target_nfe'])}",
        f"anfe-{int(spec['actual_nfe'])}",
        f"steps-{int(spec['num_steps'])}",
        f"gs{float(spec['guidance_scale']):.1f}",
    ]
    sigma = spec.get("terminal_sigma")
    if sigma is not None and float(sigma) > 0.0:
        parts.append(f"term{math.log10(float(sigma)):.2f}")
    suffix = _stochastic_tag_suffix(spec)
    if suffix is not None:
        parts.append(suffix)
    return "_".join(parts)


def build_sampling_specs(*, cfg: Any, metric_cfg: Any) -> List[Dict[str, Any]]:
    sweep = cfg.evaluation.sampling_sweep
    terminal_sigmas = list(getattr(metric_cfg, "terminal_sigmas", [0.08]))
    out: List[Dict[str, Any]] = []
    for base in sweep.specs:
        sc_modes = list(getattr(base, "sc_refresh_modes", ["carry"]))
        if sc_modes != ["carry"]:
            raise ValueError("only sc_refresh_mode='carry' is supported")
        for target_nfe in list(base.target_nfes):
            num_steps, actual_nfe = steps_for_target_nfe(target_nfe)
            for sigma in terminal_sigmas:
                spec = dict(
                    sampler_name=str(base.sampler_name),
                    sc_refresh_mode="carry",
                    ati_eta=0.0,
                    target_nfe=int(target_nfe),
                    actual_nfe=int(actual_nfe),
                    num_steps=int(num_steps),
                    terminal_sigma=float(sigma),
                    guidance_scale=0.0,
                )
                for k in _STOCH_SPEC_KEYS:
                    if hasattr(base, k):
                        spec[k] = getattr(base, k)
                if bool(spec.get("stochastic_enabled", False)):
                    if float(spec.get("s_churn", 0.0)) <= 0.0:
                        raise ValueError(f"stochastic spec with s_churn <= 0: {spec}")
                    for k, v in (("s_noise", 1.0), ("window_mode", "entropy_cdf"),
                                 ("entropy_quantile_lo", 0.10), ("entropy_quantile_hi", 0.90),
                                 ("entropy_quantile_lo_cluster", None), ("entropy_quantile_hi_cluster", None),
                                 ("s_tmin", None), ("s_tmax", None), ("entropy_fallback", "deterministic")):
                        spec.setdefault(k, v)
                spec["tag"] = _tag_for_spec(spec)
                out.append(spec)
    tags = [s["tag"] for s in out]
    if len(set(tags)) != len(tags):
        raise ValueError("duplicate sampling spec tags")
    return out
