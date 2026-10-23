"""MAUVE of generated text against held-out test text (Table 11), one row group per spec.

Reads the texts the GenPPL pass generated for the same spec (same cache key), so MAUVE,
GenPPL and entropy describe one generation.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from diffusion.continuous.processes import ContinuousForwardProcess
from evaluation.distributed import barrier as ddp_barrier
from evaluation.results import (
    _append_results_online,
    _generation_meta_fields,
    _load_scored_keys,
    _shared_cache_root,
    spec_cache_fields,
)
from evaluation.sampling_specs import build_sampling_specs
from evaluation.text_generations import SharedGenerationCache


# -----------------------------------------------------------------------------
# MAUVE (Pillutla et al., 2021) with mauve-text's defaults, averaged over k-means seeds.
# GPT-2 Large features are computed once; each seed only re-runs the quantization.
# -----------------------------------------------------------------------------

@dataclass
class MauveConfig:
    featurize_model_name: str = "gpt2-large"
    max_text_length: int = 1024
    num_buckets: Union[int, str] = "auto"  # mauve-text default: n / 10 clusters
    device_id: Optional[int] = None
    batch_size: int = 64


class MauveEvaluator:
    def __init__(self, *, cfg: Optional[MauveConfig] = None):
        self.cfg = cfg or MauveConfig()

    def score_multi(self, *, p_text: List[str], q_text: List[str], seeds: Sequence[int]) -> Dict[str, Any]:
        """Mean, sample std, min and max of each MAUVE statistic over the k-means seeds."""
        import mauve
        from mauve.compute_mauve import get_features_from_input

        cfg = self.cfg
        feats = {
            name: get_features_from_input(
                None, None, list(texts), cfg.featurize_model_name, int(cfg.max_text_length),
                cfg.device_id, name=name, batch_size=int(cfg.batch_size), verbose=False,
            )
            for name, texts in (("p", p_text), ("q", q_text))
        }
        nb = cfg.num_buckets if cfg.num_buckets == "auto" else int(cfg.num_buckets)
        per_seed: Dict[str, List[float]] = {}
        res: Dict[str, Any] = {}
        for s in seeds:
            out = mauve.compute_mauve(p_features=feats["p"], q_features=feats["q"], num_buckets=nb, seed=int(s), verbose=False)
            for k in ("mauve", "mauve_star", "frontier_integral", "frontier_integral_star"):
                v = getattr(out, k, None)
                if isinstance(v, (int, float, np.number)) and not np.isnan(float(v)):
                    per_seed.setdefault(k, []).append(float(v))
                    res[f"{k}_seed{int(s)}"] = float(v)
        for k, vals in per_seed.items():
            res[k] = float(np.mean(vals))
            res[f"{k}_min"] = float(np.min(vals))
            res[f"{k}_max"] = float(np.max(vals))
            if len(vals) > 1:
                res[f"{k}_std"] = float(statistics.stdev(vals))
        res["num_buckets_used"] = float(len(p_text)) / 10.0 if nb == "auto" else float(nb)
        res["n_seeds"] = float(len(seeds))
        res["n_p"] = float(len(p_text))
        res["n_q"] = float(len(q_text))
        return res



def evaluate_mauve(cfg, model, use_ema, test_loader, device, rank0, run_meta):
    m_cfg = cfg.evaluation.mauve
    n_samples = int(getattr(m_cfg, "num_samples", 1024))
    micro_bs = int(getattr(m_cfg, "micro_batch_size", 512))
    sigma_max = getattr(m_cfg, "sigma_max", None)
    sigma_max = None if sigma_max is None else float(sigma_max)
    seed = int(getattr(m_cfg, "seed", 42))
    kmeans_seeds = [int(s) for s in getattr(m_cfg, "kmeans_seeds", [25])]
    amp_dtype = str(getattr(cfg.evaluation, "amp_dtype", "bf16"))
    ckpt_name = Path(cfg.evaluation.checkpoint_path).name
    mcfg = MauveConfig(
        featurize_model_name=str(getattr(m_cfg, "featurizer_name", "gpt2-large")),
        max_text_length=int(getattr(m_cfg, "max_tokens", 1024)),
        num_buckets=getattr(m_cfg, "num_buckets", "auto"),
        device_id=(device.index or 0) if device.type == "cuda" else None,
        batch_size=int(getattr(m_cfg, "batch_size", 64)),
    )

    shared_cache = SharedGenerationCache(cfg, _shared_cache_root(cfg, m_cfg))
    proc = ContinuousForwardProcess(cfg)
    specs = build_sampling_specs(cfg=cfg, metric_cfg=m_cfg)
    done = _load_scored_keys(cfg, "full_mauve")
    evaluator = MauveEvaluator(cfg=mcfg) if rank0 else None
    if rank0:
        print(f"\n── MAUVE ({mcfg.featurize_model_name}), N={n_samples}, k-means seeds {kmeans_seeds} ──")

    for spec in specs:
        tag = spec["tag"]
        if (ckpt_name, tag) in done:
            continue
        common = spec_cache_fields(spec, sigma_max=sigma_max, use_ema=use_ema)
        cache_key = shared_cache.make_cache_key(
            checkpoint_name=ckpt_name, split="test", num_samples=n_samples,
            num_steps=int(spec["num_steps"]), micro_batch_size=micro_bs, seed=seed, **common,
        )
        texts = shared_cache.get_or_create(
            checkpoint_name=ckpt_name, split="test", cache_key=cache_key, tag=tag,
            model=model, proc=proc, device=device, data_loader=test_loader,
            num_samples=n_samples, seed=seed, use_amp=True, amp_dtype=amp_dtype,
            micro_batch_size=micro_bs, sigma_max=sigma_max,
            meta=dict(tag=tag, num_samples=n_samples, num_steps=int(spec["num_steps"]),
                      nfe=int(spec["actual_nfe"]), target_nfe=spec["target_nfe"], **common),
            sampling_specs=[spec],
        )
        if rank0:
            with torch.amp.autocast("cuda", enabled=False):
                res = evaluator.score_multi(p_text=texts.full, q_text=texts.ref_full, seeds=kmeans_seeds)
            base_fields = dict(
                split="test", tag=tag, num_sampling_steps=int(spec["num_steps"]),
                micro_batch_size=micro_bs, num_samples=n_samples,
                featurizer_name=mcfg.featurize_model_name, max_text_length=mcfg.max_text_length,
                nfe=int(spec["actual_nfe"]), target_nfe=spec["target_nfe"],
                num_buckets=mcfg.num_buckets, kmeans_seeds="|".join(map(str, kmeans_seeds)),
                **common, **_generation_meta_fields(texts.meta),
            )
            details = f"view=full,tag={tag},N={n_samples},steps={spec['num_steps']},nfe={spec['actual_nfe']}"
            rows_out: List[Dict[str, Any]] = [
                {**run_meta, **base_fields, "metric": f"full_{k}", "value": float(v), "details": details}
                for k, v in res.items()
                if isinstance(v, (int, float)) and not np.isnan(v)
            ]
            _append_results_online(cfg, rows_out)
            print(f"   [{tag}] MAUVE={res.get('mauve', float('nan')):.4f} +/- {res.get('mauve_std', float('nan')):.4f}", flush=True)
        ddp_barrier()
