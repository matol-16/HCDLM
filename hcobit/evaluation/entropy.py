"""Token unigram entropy of generated and real text (the x-axis of the GenPPL frontier).

Per sample, the entropy of its token unigram distribution; averaged over samples. Read
from the text cache the GenPPL pass wrote, so it describes the same generation.
LM1B: BERT ids. OpenWebText: the GPT-2 ids the gpt2id_bpe16 codes expand to, with each
row cut at its true length (EOS padding would deflate the entropy).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch

from evaluation.results import (
    _append_results_online,
    _generation_meta_fields,
    _load_scored_keys,
    _shared_cache_root,
    spec_cache_fields,
)
from evaluation.sampling_specs import build_sampling_specs
from evaluation.text_generations import SharedGenerationCache
from evaluation.text_metrics import token_unigram_entropy_from_token_ids
from utils.text_decode import (
    bitstreams_to_token_ids_raw_binary,
    code_ids_to_gpt2_token_id_lists_for_eval,
    _owt_assets,
    decode_bitstreams_to_token_ids_for_eval,
)


def _token_rows(cfg, bits: List[List[int]], dataset_obj) -> List[List[int]]:
    bits = torch.tensor(bits, dtype=torch.long)
    if str(getattr(cfg.data, "sequence_codec", "base")).lower() == "gpt2id_bpe16":
        code_ids = bitstreams_to_token_ids_raw_binary(bits, bits_per_token=int(cfg.data.bits_per_token), cfg=cfg)
        gpt2_tok, code_tok, code_meta = _owt_assets(cfg, dataset_obj)
        return code_ids_to_gpt2_token_id_lists_for_eval(
            code_ids, gpt2_tokenizer=gpt2_tok, code_tokenizer=code_tok, code_meta=code_meta
        )
    return [row.tolist() for row in decode_bitstreams_to_token_ids_for_eval(cfg, bits, dataset_obj=dataset_obj)]


def _avg_entropy(rows: List[List[int]]) -> float:
    if not rows:
        return float("nan")
    return float(sum(token_unigram_entropy_from_token_ids(torch.tensor(r, dtype=torch.long)) for r in rows) / len(rows))


def evaluate_entropy(cfg, test_loader, run_meta, *, use_ema: bool, include_real: bool = True) -> None:
    """Append gen/real token-unigram-entropy rows for every GenPPL spec (rank 0 only)."""
    ext_cfg = cfg.evaluation.external_ppl
    n_samples = int(getattr(ext_cfg, "num_samples", 1024))
    micro_bs = int(getattr(ext_cfg, "micro_batch_size", 512))
    sigma_max = getattr(ext_cfg, "sigma_max", None)
    sigma_max = None if sigma_max is None else float(sigma_max)
    seed = int(getattr(ext_cfg, "seed", 42))
    ckpt_name = Path(cfg.evaluation.checkpoint_path).name
    dataset_obj = getattr(test_loader, "dataset", None)
    cache = SharedGenerationCache(cfg, _shared_cache_root(cfg, ext_cfg))
    done = _load_scored_keys(cfg, "gen_full_token_unigram_entropy")

    rows_out: List[Dict[str, Any]] = []
    for spec in build_sampling_specs(cfg=cfg, metric_cfg=ext_cfg):
        tag = spec["tag"]
        if (ckpt_name, tag) in done:
            continue
        common = spec_cache_fields(spec, sigma_max=sigma_max, use_ema=use_ema)
        key = cache.make_cache_key(checkpoint_name=ckpt_name, split="test", num_samples=n_samples,
                                   num_steps=int(spec["num_steps"]), micro_batch_size=micro_bs, seed=seed, **common)
        path = cache._path(checkpoint_name=ckpt_name, split="test", cache_key=key, tag=tag)
        if not path.exists():
            continue
        texts = cache.load(checkpoint_name=ckpt_name, split="test", cache_key=key, tag=tag)
        base = {
            **run_meta,
            "split": "test",
            "tag": tag,
            "nfe": int(spec["actual_nfe"]),
            "target_nfe": spec["target_nfe"],
            "num_samples": n_samples,
            "cache_file": str(path),
            **common,
            **_generation_meta_fields(texts.meta),
            "metric_source": "posthoc_cached_token_entropy",
            "details": "source=cache,entropy=token_unigram",
        }
        rows_out.append({**base, "metric": "gen_full_token_unigram_entropy",
                         "value": _avg_entropy(_token_rows(cfg, texts.gen_bits, dataset_obj))})
        if include_real:
            rows_out.append({**base, "metric": "real_full_token_unigram_entropy",
                             "value": _avg_entropy(_token_rows(cfg, texts.ref_bits, dataset_obj))})
    _append_results_online(cfg, rows_out)
    print(f"Token unigram entropy: {len(rows_out)} row(s) added.")
