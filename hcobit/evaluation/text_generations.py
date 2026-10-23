"""Generate-once text cache shared by the GenPPL, entropy and MAUVE passes.

One JSONL file per (checkpoint, split, cache key, sampling spec): a meta line, then one
row per sample with the decoded generation and reference text and their raw bit / token
payloads. The cache key fixes everything that determines the samples, so the MAUVE pass
reads exactly the texts the GenPPL pass generated.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from evaluation.generation_driver import GenerationBatch, GenerationDriver
from utils.text_decode import bitstreams_to_token_ids_raw_binary, decode_bitstreams_for_eval


def _ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank_world() -> tuple[int, int]:
    if not _ddp_is_on():
        return 0, 1
    return int(dist.get_rank()), int(dist.get_world_size())


def barrier() -> None:
    if not _ddp_is_on():
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def _sanitize_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(tag))


def _short_cache_key_dir(cache_key: str, *, prefix_chars: int = 64) -> str:
    """Filesystem-safe directory name for a long cache key (prefix + blake2b digest)."""
    key = str(cache_key)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
    prefix = re.sub(r"[^A-Za-z0-9_.=-]+", "_", key)[:prefix_chars].strip("._-") or "cache"
    return f"{prefix}__h={digest}"


def _tensor_to_nested_list(x: Optional[torch.Tensor]) -> Optional[List[List[int]]]:
    if x is None:
        return None
    x = x.detach().to("cpu")
    x = (x > 0.5).to(torch.long) if x.is_floating_point() else x.to(torch.long)
    return x.tolist()


@dataclass
class CachedTexts:
    full: List[str]
    ref_full: List[str]
    gen_bits: Optional[List[List[int]]]
    ref_bits: Optional[List[List[int]]]
    gen_tokens: Optional[List[List[int]]]  # OWT: gpt2id_bpe16 code ids, kept for the entropy pass
    ref_tokens: Optional[List[List[int]]]
    meta: Dict[str, Any]

    @staticmethod
    def empty() -> "CachedTexts":
        return CachedTexts([], [], None, None, None, None, {})


def resolve_checkpoints(spec, base_ckpt_dir: Path) -> list[Path]:
    """Existing checkpoint files named by ``spec`` (absolute, or relative to base_ckpt_dir)."""
    names = spec if isinstance(spec, (list, tuple)) else ([] if spec is None else [spec])
    out = {}
    for s in names:
        p = Path(str(s))
        p = p if p.is_absolute() else base_ckpt_dir / p
        if p.is_file():
            out[str(p.resolve())] = p
    return sorted(out.values(), key=lambda p: p.name)


class SharedGenerationCache:
    """Generates once (sharded across ranks), decodes and writes on rank 0."""

    def __init__(self, cfg: Any, cache_root: Path):
        self.cfg = cfg
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.driver = GenerationDriver(cfg)

    @staticmethod
    def make_cache_key(
        *,
        checkpoint_name: str,
        split: str,
        num_samples: int,
        num_steps: int,
        micro_batch_size: int,
        sigma_max: Optional[float],
        use_ema: bool,
        seed: int,
        sampler_name: Optional[str] = None,
        terminal_sigma: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        sc_refresh_mode: Optional[str] = None,
        ati_eta: Optional[float] = None,
        stochastic_enabled: Optional[bool] = None,
        s_churn: Optional[float] = None,
        s_noise: Optional[float] = None,
        window_mode: Optional[str] = None,
        entropy_quantile_lo: Optional[float] = None,
        entropy_quantile_hi: Optional[float] = None,
        s_tmin: Optional[float] = None,
        s_tmax: Optional[float] = None,
        entropy_fallback: Optional[str] = None,
        s_churn_cluster: Optional[float] = None,
        entropy_quantile_lo_cluster: Optional[float] = None,
        entropy_quantile_hi_cluster: Optional[float] = None,
    ) -> str:
        def f(x):
            return "none" if x is None else f"{float(x):.8g}"

        def s(x):
            return "none" if x is None else str(x)

        st = "none" if stochastic_enabled is None else str(int(bool(stochastic_enabled)))
        return (
            f"ckpt={checkpoint_name}|split={split}|N={int(num_samples)}|"
            f"steps={int(num_steps)}|mb={int(micro_batch_size)}|"
            f"sigma_max={f(sigma_max)}|ema={int(bool(use_ema))}|seed={int(seed)}|"
            f"sampler={s(sampler_name)}|term={f(terminal_sigma)}|gs={f(guidance_scale)}|"
            f"scr={s(sc_refresh_mode)}|ati={f(ati_eta)}|"
            f"stoch={st}|s_churn={f(s_churn)}|s_noise={f(s_noise)}|"
            f"window={s(window_mode)}|qlo={f(entropy_quantile_lo)}|qhi={f(entropy_quantile_hi)}|"
            f"s_tmin={f(s_tmin)}|s_tmax={f(s_tmax)}|fallback={s(entropy_fallback)}"
        ) + (
            # per-modality sampler: cluster churn rate and window
            "" if s_churn_cluster is None else
            f"|s_churn_cluster={f(s_churn_cluster)}|qlo_cluster={f(entropy_quantile_lo_cluster)}"
            f"|qhi_cluster={f(entropy_quantile_hi_cluster)}"
        )

    def _path(self, *, checkpoint_name: str, split: str, cache_key: str, tag: str) -> Path:
        p = self.cache_root / checkpoint_name / split / _short_cache_key_dir(cache_key)
        p.mkdir(parents=True, exist_ok=True)
        manifest = p / "_cache_key.json"
        if _rank_world()[0] == 0 and not manifest.exists():
            manifest.write_text(json.dumps({"cache_key": str(cache_key)}, indent=2) + "\n")
        return p / f"{_sanitize_tag(tag)}.jsonl"

    def load(self, *, checkpoint_name: str, split: str, cache_key: str, tag: str) -> CachedTexts:
        path = self._path(checkpoint_name=checkpoint_name, split=split, cache_key=cache_key, tag=tag)
        out = CachedTexts.empty()
        cols = {"gen_bits": [], "ref_bits": [], "gen_tokens": [], "ref_tokens": []}
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                if i == 0 and "_meta" in row:
                    out.meta = row["_meta"]
                    continue
                out.full.append(row["full"])
                out.ref_full.append(row["ref_full"])
                for k, v in cols.items():
                    if row.get(k) is not None:
                        v.append(row[k])
        for k, v in cols.items():
            setattr(out, k, v or None)
        return out

    def save(self, *, checkpoint_name: str, split: str, cache_key: str, tag: str, texts: CachedTexts) -> Path:
        path = self._path(checkpoint_name=checkpoint_name, split=split, cache_key=cache_key, tag=tag)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": texts.meta}, ensure_ascii=False) + "\n")
            for i in range(len(texts.full)):
                row = {"full": texts.full[i], "ref_full": texts.ref_full[i]}
                for k in ("gen_bits", "ref_bits", "gen_tokens", "ref_tokens"):
                    v = getattr(texts, k)
                    row[k] = None if v is None else v[i]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
        return path

    def _decode_batch(self, batch: GenerationBatch, dataset_obj) -> CachedTexts:
        def decode(bits):
            return decode_bitstreams_for_eval(
                self.cfg, bits, prompt_len_bits=batch.prompt_len_bits, mode="full", dataset_obj=dataset_obj
            )

        gen_tokens = ref_tokens = None
        if str(getattr(self.cfg.data, "sequence_codec", "base")).lower() == "gpt2id_bpe16":
            bpt = int(self.cfg.data.bits_per_token)
            gen_tokens = _tensor_to_nested_list(bitstreams_to_token_ids_raw_binary(batch.gen_bits, bits_per_token=bpt, cfg=self.cfg))
            ref_tokens = _tensor_to_nested_list(bitstreams_to_token_ids_raw_binary(batch.ref_bits, bits_per_token=bpt, cfg=self.cfg))
        return CachedTexts(
            full=decode(batch.gen_bits),
            ref_full=decode(batch.ref_bits),
            gen_bits=_tensor_to_nested_list(batch.gen_bits),
            ref_bits=_tensor_to_nested_list(batch.ref_bits),
            gen_tokens=gen_tokens,
            ref_tokens=ref_tokens,
            meta={},
        )

    @torch.no_grad()
    def get_or_create(
        self,
        *,
        checkpoint_name: str,
        split: str,
        cache_key: str,
        tag: str,
        model,
        proc,
        device: torch.device,
        data_loader,
        num_samples: int,
        seed: int,
        use_amp: bool,
        amp_dtype: str,
        micro_batch_size: int,
        sigma_max: Optional[float],
        meta: Dict[str, Any],
        sampling_specs: List[Dict[str, Any]],
    ) -> CachedTexts:
        """Cached texts for one spec, generating them if absent (rank 0 gets the texts)."""
        path = self._path(checkpoint_name=checkpoint_name, split=split, cache_key=cache_key, tag=tag)
        rank0 = _rank_world()[0] == 0
        exists = [bool(path.exists()) if rank0 else False]
        if _ddp_is_on():
            dist.broadcast_object_list(exists, src=0)
        if exists[0]:
            barrier()
            if not rank0:
                return CachedTexts.empty()
            return self.load(checkpoint_name=checkpoint_name, split=split, cache_key=cache_key, tag=tag)

        batches = self.driver.generate(
            model=model,
            proc=proc,
            device=device,
            loader=data_loader,
            num_samples=num_samples,
            sampling_specs=sampling_specs,
            entropy_run_dir=getattr(self.cfg.evaluation, "entropy_run_dir", None),
            seed=seed,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            micro_batch_size=micro_batch_size,
            sigma_max=sigma_max,
        )
        if not rank0:
            barrier()
            return CachedTexts.empty()

        batch = batches[tag]
        texts = self._decode_batch(batch, getattr(data_loader, "dataset", None))
        texts.meta = {**meta, **(batch.stats or {})}
        self.save(checkpoint_name=checkpoint_name, split=split, cache_key=cache_key, tag=tag, texts=texts)
        barrier()
        return texts
