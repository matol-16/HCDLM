"""GenPPL: GPT-2 Large perplexity of generated (and real) text, one row group per spec (Table 1)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from diffusion.continuous.processes import ContinuousForwardProcess
from evaluation.distributed import barrier as ddp_barrier
from evaluation.results import (
    _append_results_online,
    _generation_meta_fields,
    _load_scored_keys,
    _shared_cache_root,
    spec_cache_fields,
)
from evaluation.external_perplexity import HFExternalPerplexityEvaluator
from evaluation.sampling_specs import build_sampling_specs
from evaluation.text_generations import SharedGenerationCache


def _append_result_rows(rows_out, run_meta, base_fields, metric_dict, metric_prefix, details):
    for k, v in metric_dict.items():
        rows_out.append({**run_meta, **base_fields, "metric": f"{metric_prefix}_{k}", "value": float(v), "details": details})


def evaluate_genppl(cfg, model, use_ema, test_loader, device, rank0, run_meta):
    ext_cfg = cfg.evaluation.external_ppl
    model_name = str(getattr(ext_cfg, "hf_model_name", "openai-community/gpt2-large"))
    n_samples = int(getattr(ext_cfg, "num_samples", 1024))
    micro_bs = int(getattr(ext_cfg, "micro_batch_size", 512))
    sigma_max = getattr(ext_cfg, "sigma_max", None)
    sigma_max = None if sigma_max is None else float(sigma_max)
    seed = int(getattr(ext_cfg, "seed", 42))
    compute_real = bool(getattr(ext_cfg, "compute_real_reference", True))
    amp_dtype = str(getattr(cfg.evaluation, "amp_dtype", "bf16"))
    ckpt_name = Path(cfg.evaluation.checkpoint_path).name

    shared_cache = SharedGenerationCache(cfg, _shared_cache_root(cfg, ext_cfg))
    proc = ContinuousForwardProcess(cfg)
    specs = build_sampling_specs(cfg=cfg, metric_cfg=ext_cfg)
    # Resume: skip specs already scored in this results.csv (same set on every rank).
    done = set() if bool(getattr(ext_cfg, "force_rescore", False)) else _load_scored_keys(cfg, "gen_")
    ppl_evaluator = None
    if rank0:
        print(f"\n── GenPPL ({model_name}), N={n_samples}, {len(specs)} spec(s), {len(done)} already scored ──")

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
            if ppl_evaluator is None:
                ppl_evaluator = HFExternalPerplexityEvaluator(
                    model_name=model_name,
                    revision=getattr(ext_cfg, "hf_revision", None),
                    device=device,
                    torch_dtype=str(getattr(ext_cfg, "hf_dtype", "bfloat16")),
                    attn_implementation=getattr(ext_cfg, "attn_implementation", "sdpa"),
                    use_amp=True,
                )
            base_fields = dict(
                split="test", tag=tag, num_sampling_steps=int(spec["num_steps"]),
                micro_batch_size=micro_bs, num_samples=n_samples, nfe=int(spec["actual_nfe"]),
                target_nfe=spec["target_nfe"], external_lm_name=model_name, **common,
                **_generation_meta_fields(texts.meta),
            )
            details = (f"view=full,tag={tag},N={n_samples},steps={spec['num_steps']},"
                       f"nfe={spec['actual_nfe']},lm={model_name},sc_refresh_mode=carry")
            rows_out: List[Dict[str, Any]] = []
            gen_scores = ppl_evaluator.score_texts(texts.full)
            _append_result_rows(rows_out, run_meta, base_fields, gen_scores, "gen_full", details)
            if compute_real:
                real_scores = ppl_evaluator.score_texts(texts.ref_full)
                _append_result_rows(rows_out, run_meta, {**base_fields, "tag": "REAL"}, real_scores, "real_full",
                                    details.replace(f"tag={tag}", "tag=REAL"))
            _append_results_online(cfg, rows_out)
            print(f"   [{tag}] GenPPL={gen_scores.get('external_ppl')}", flush=True)
        ddp_barrier()
