"""
Driver: estimate the loss-vs-σ shape (entropy CDF) on the test split.

Reproduces, post-training, the σ/loss buffer + entropy-CDF mechanism that
``trainers/trainer.py`` runs online during training. Under cluster-bit guidance
it additionally estimates a separate profile for the information bits and for the
trailing semantic ("additional") cluster bits, so their shapes can be compared.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from evaluation.utils import resolve_eval_amp
from evaluation.results import _append_results_online
from evaluation.sigma_loss_profile import (
    collect_sigma_loss_pairs,
    build_entropy_profile,
    plot_sigma_loss_profiles,
)


def _profile_cfg_get(cfg, name, default):
    """Read cfg.evaluation.sigma_loss_profile.<name>, then cfg.train.<name>."""
    ev = getattr(getattr(cfg, "evaluation", object()), "sigma_loss_profile", None)
    if ev is not None and getattr(ev, name, None) is not None:
        return getattr(ev, name)
    tr = getattr(cfg, "train", None)
    if tr is not None and getattr(tr, name, None) is not None:
        return getattr(tr, name)
    return default


def evaluate_sigma_loss_profile(cfg, model, test_loader, device, rank0, run_meta, max_batches=None):
    if not rank0:
        # Single-rank computation over the full (unsharded) test loader.
        return

    print("\n── Estimating σ-loss profile (entropy CDF over test split) ─────")

    use_amp, amp_dtype = resolve_eval_amp(device, cfg)

    # cfg.evaluation.sigma_loss_profile > cfg.train > default.
    sigma_sampling = str(_profile_cfg_get(cfg, "sigma_sampling", "log-uniform"))
    mc_samples = int(_profile_cfg_get(cfg, "mc_samples", 1))
    num_bins = int(_profile_cfg_get(cfg, "entropy_num_bins", 256))

    # Entropy-profile recipe parameters — match training defaults so the post-hoc
    # profile is comparable to the online one.
    c = float(_profile_cfg_get(cfg, "entropy_regularizer_c", 0.1))
    n = float(_profile_cfg_get(cfg, "entropy_regularizer_n", 3.0))
    target = _profile_cfg_get(cfg, "entropy_target", "rate")
    if isinstance(target, str):
        t = target.lower()
        power = 1.0 if t == "rate" else (0.5 if t in {"sqrt", "sqrt_rate", "sqrt-rate"} else float(t))
    else:
        power = float(target)

    sigma_min = float(cfg.diffusion.continuous.sigma_min)
    sigma_max = float(cfg.diffusion.continuous.sigma_max)

    pairs = collect_sigma_loss_pairs(
        model=model,
        cfg=cfg,
        data_loader=test_loader,
        device=device,
        sigma_sampling=sigma_sampling,
        mc_samples=mc_samples,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        max_batches=max_batches,
        progress=True,
    )

    if not pairs.groups:
        print("⚠️  sigma_loss_profile produced no pairs.")
        return

    profiles = {}
    for g, d in pairs.groups.items():
        profiles[g] = build_entropy_profile(
            d["sigma"], d["metric"],
            sigma_min=sigma_min, sigma_max=sigma_max,
            num_bins=num_bins, c=c, n=n, power=power,
        )

    # ── Output dir ──────────────────────────────────────────────────────────
    out_dir = Path(getattr(cfg.evaluation, "out_dir", f"runs/{cfg.experiment}/evaluation")).expanduser()
    prof_dir = out_dir / "sigma_loss_profile"
    prof_dir.mkdir(parents=True, exist_ok=True)

    # Raw (σ, loss) pairs per group (for re-plotting / further analysis).
    torch.save(
        {g: {"sigma": d["sigma"], "metric": d["metric"]} for g, d in pairs.groups.items()},
        prof_dir / "sigma_loss_pairs.pt",
    )

    # Per-group profile tensors (pdf/cdf/sigmas/edges/curves).
    for g, prof in profiles.items():
        torch.save(
            dict(
                sigmas=prof.sigmas_mid, edges=prof.edges, pdf=prof.pdf, cdf=prof.cdf,
                mean_rate=prof.mean_rate, mean_metric=prof.mean_metric, count=prof.count,
            ),
            prof_dir / f"entropy_profile_{g}.pt",
        )

    cluster_on = bool(getattr(cfg.data, "cluster_bits_enabled", False))
    title = (
        f"{cfg.experiment}\nσ-loss profile (test) — "
        f"{'info vs cluster bits' if cluster_on else 'all bits'} | "
        f"dist={sigma_sampling}, bins={num_bins}, power={power}"
    )
    plot_path = prof_dir / "sigma_loss_profile.png"
    plot_sigma_loss_profiles(profiles, out_path=plot_path, title=title)

    # JSON summary (per-group diagnostics + per-bin curves).
    summary = dict(
        experiment=str(cfg.experiment),
        checkpoint=str(getattr(cfg.evaluation, "checkpoint_path", "")),
        sigma_sampling=sigma_sampling,
        mc_samples=int(mc_samples),
        num_bins=int(num_bins),
        c=c, n=n, power=power,
        sigma_min=sigma_min, sigma_max=sigma_max,
        num_examples=int(pairs.num_examples),
        cluster_bits_enabled=cluster_on,
        groups={
            g: dict(
                n_pairs=int(d["sigma"].numel()),
                metric_mean=float(d["metric"].mean().item()),
                ln_mu=profiles[g].ln_mu,
                ln_std=profiles[g].ln_std,
                sigmas=profiles[g].sigmas_mid.tolist(),
                cdf=profiles[g].cdf.tolist(),
                mean_rate=profiles[g].mean_rate.tolist(),
                mean_metric=profiles[g].mean_metric.tolist(),
            )
            for g, d in pairs.groups.items()
        },
    )
    with open(prof_dir / "sigma_loss_profile.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ σ-loss profile written to: {prof_dir}")
    print(f"  plot:    {plot_path}")
    rows_out = []
    for g, prof in profiles.items():
        d = pairs.groups[g]
        print(
            f"  [{g:7s}] n={d['sigma'].numel()}  E[loss]={d['metric'].mean().item():.4g}  "
            f"logσ μ={prof.ln_mu:.3f} σ={prof.ln_std:.3f}"
        )
        rows_out.append(
            dict(
                **run_meta, metric="sigma_loss_profile", split="test",
                value=float(d["metric"].mean().item()),
                details=f"group={g},ln_mu={prof.ln_mu:.4f},ln_std={prof.ln_std:.4f},"
                        f"dist={sigma_sampling},bins={num_bins},power={power},n_pairs={int(d['sigma'].numel())}",
            )
        )
    _append_results_online(cfg, rows_out)
