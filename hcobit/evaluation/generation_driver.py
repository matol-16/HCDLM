"""Unconditional bitstream generation for evaluation, sharded across DDP ranks.

Every chunk of samples is drawn under its own seed, derived from the run seed, the
sampling spec's tag and the chunk's global index, so a generation is reproducible
independently of the number of GPUs. Each rank also takes its share of reference rows
from the test split (for the real-text GenPPL and MAUVE references).
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank_world() -> Tuple[int, int]:
    if not _ddp_is_on():
        return 0, 1
    return int(dist.get_rank()), int(dist.get_world_size())


def _split_counts(N: int, world_size: int) -> List[int]:
    base, rem = N // world_size, N % world_size
    return [base + (1 if r < rem else 0) for r in range(world_size)]


def _stable_int_hash(s: str) -> int:
    """Stable across processes and runs (unlike hash())."""
    return int(zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reduce_scalar(value: float, device: torch.device, op) -> float:
    if not _ddp_is_on():
        return float(value)
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=op)
    return float(t.item())


def _resolve_autocast_dtype(amp_dtype: str) -> torch.dtype:
    mode = str(amp_dtype).lower()
    if mode in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if mode in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unknown amp_dtype='{amp_dtype}'")


def _build_speed_stats(*, device, wall_time_local_sec, num_samples_local, seq_len_tokens, seq_len_model) -> Dict[str, Any]:
    SUM, MAX = dist.ReduceOp.SUM, dist.ReduceOp.MAX
    total = int(round(_reduce_scalar(num_samples_local, device, SUM)))
    lm_tokens = total * int(seq_len_tokens)
    positions = total * int(seq_len_model)
    wall = _reduce_scalar(wall_time_local_sec, device, MAX)
    denom = max(wall, 1e-12)
    return dict(
        generation_wall_time_sec=float(wall),
        world_size=int(_rank_world()[1]),
        num_samples=total,
        sequence_len_tokens=int(seq_len_tokens),
        sequence_len_model=int(seq_len_model),
        full_lm_tokens=lm_tokens,
        generated_lm_tokens=lm_tokens,
        full_model_positions=positions,
        generated_model_positions=positions,
        samples_per_sec=float(total / denom),
        full_lm_tokens_per_sec=float(lm_tokens / denom),
        generated_lm_tokens_per_sec=float(lm_tokens / denom),
        lm_tokens_per_sec=float(lm_tokens / denom),
        full_model_positions_per_sec=float(positions / denom),
        generated_model_positions_per_sec=float(positions / denom),
        model_positions_per_sec=float(positions / denom),
    )


@dataclass
class SamplerBundle:
    sampler: Any
    schedule: str


def create_sampler(cfg: Any, model: Any, proc: Any, sampler_name: str) -> SamplerBundle:
    """ddim_entropic: single information-uniform schedule (CoBit).
    ddim_entropic_dual: per-modality schedule (H-CoBit)."""
    from diffusion.continuous.samplers import DDIMSampler

    name = str(sampler_name).lower()
    if name in {"ddim", "ddim_entropic", "entropic"}:
        return SamplerBundle(DDIMSampler(model, proc, cfg), "entropic")
    if name in {"ddim_entropic_dual", "entropic_dual"}:
        return SamplerBundle(DDIMSampler(model, proc, cfg), "entropic_dual")
    raise ValueError(f"Unknown sampler '{sampler_name}'")


@torch.no_grad()
def gather_varlen_firstdim_to_rank0(x: torch.Tensor, dst: int = 0) -> Optional[torch.Tensor]:
    """Concatenate each rank's [n_r, ...] tensor on rank ``dst`` (None elsewhere)."""
    if not _ddp_is_on():
        return x
    rank, world = _rank_world()
    my = torch.tensor([x.size(0)], device=x.device, dtype=torch.long)
    sizes = [torch.zeros_like(my) for _ in range(world)]
    dist.all_gather(sizes, my)
    sizes_int = [int(s.item()) for s in sizes]
    mx = max(sizes_int)
    x_pad = x.detach()
    if x_pad.size(0) < mx:
        pad = torch.zeros((mx - x_pad.size(0),) + x_pad.shape[1:], device=x.device, dtype=x.dtype)
        x_pad = torch.cat([x_pad, pad], dim=0)
    gathered = [torch.empty_like(x_pad) for _ in range(world)]
    dist.all_gather(gathered, x_pad)
    if rank != dst:
        return None
    chunks = [gathered[i][: sizes_int[i]] for i in range(world) if sizes_int[i] > 0]
    return torch.cat(chunks, dim=0) if chunks else x_pad[:0]


@dataclass
class GenerationBatch:
    gen_bits: Optional[torch.Tensor] = None
    ref_bits: Optional[torch.Tensor] = None
    prompt_len_bits: Optional[torch.Tensor] = None
    stats: Optional[Dict[str, Any]] = None


#: spec key -> cfg.evaluation.stochastic key, with the value used when a spec omits it.
_STOCHASTIC_KEYS = {
    "stochastic_enabled": ("enabled", False),
    "s_churn": ("s_churn", 0.0),
    "s_churn_cluster": ("s_churn_cluster", None),
    "s_noise": ("s_noise", 1.0),
    "window_mode": ("window_mode", "entropy_cdf"),
    "entropy_quantile_lo": ("entropy_quantile_lo", 0.10),
    "entropy_quantile_hi": ("entropy_quantile_hi", 0.90),
    "entropy_quantile_lo_cluster": ("entropy_quantile_lo_cluster", None),
    "entropy_quantile_hi_cluster": ("entropy_quantile_hi_cluster", None),
    "s_tmin": ("s_tmin", None),
    "s_tmax": ("s_tmax", None),
    "entropy_fallback": ("entropy_fallback", "deterministic"),
}


class GenerationDriver:
    def __init__(self, cfg: Any):
        self.cfg = cfg

    def _iter_take_ref(self, loader: DataLoader, n_local: int, device: torch.device, *, skip: int = 0) -> torch.Tensor:
        """This rank's n_local reference rows: the test split in fixed order, skipping the
        rows of lower ranks (unless the loader already shards with a DistributedSampler)."""
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(0)
            skip = 0
        chunks, remaining, to_skip = [], n_local, int(skip)
        for batch in loader:
            if remaining <= 0:
                break
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            if batch.dim() == 3:
                batch = batch.squeeze(1)
            if to_skip > 0:
                drop = min(to_skip, batch.size(0))
                batch, to_skip = batch[drop:], to_skip - drop
                if batch.size(0) == 0:
                    continue
            take = min(remaining, batch.size(0))
            chunks.append(batch[:take].to(device))
            remaining -= take
        if n_local == 0 or not chunks:
            S = int(self.cfg.data.sequence_len)
            return torch.zeros((max(0, n_local), S), device=device, dtype=torch.float32)
        ref = torch.cat(chunks, dim=0)
        return ref if ref.dtype in {torch.float32, torch.float64} else ref.float()

    @torch.no_grad()
    def generate(
        self,
        *,
        model: Any,
        proc: Any,
        device: torch.device,
        loader: DataLoader,
        num_samples: int,
        sampling_specs: List[Dict[str, Any]],
        entropy_run_dir: Optional[str] = None,
        seed: int = 42,
        use_amp: bool = True,
        amp_dtype: str = "bf16",
        micro_batch_size: Optional[int] = None,
        sigma_max: Optional[float] = None,
    ) -> Dict[str, GenerationBatch]:
        """{spec tag: GenerationBatch} on rank 0 (empty dict on other ranks)."""
        rank, world = _rank_world()
        counts = _split_counts(int(num_samples), world)
        n_local = counts[rank]
        global_offset = int(sum(counts[:rank]))
        MOD = 2**63 - 1
        devices_for_fork = [int(device.index)] if (device.type == "cuda" and device.index is not None) else []
        local_bs = int(micro_batch_size) if micro_batch_size else 128
        local_bs = max(1, min(local_bs, n_local if n_local > 0 else 1))

        ref_local = self._iter_take_ref(loader, n_local, device=device, skip=global_offset)
        S_bits = int(ref_local.size(1)) if ref_local.numel() else int(self.cfg.data.sequence_len)
        S_tok = int(self.cfg.data.sequence_len_tokens)
        no_prompt = torch.zeros((n_local, S_bits), device=device, dtype=torch.bool)
        amp_enabled = bool(use_amp and device.type == "cuda")
        dtype = _resolve_autocast_dtype(amp_dtype)

        ev = self.cfg.evaluation
        if getattr(ev, "stochastic", None) is None:
            from ml_collections import config_dict
            ev.stochastic = config_dict.ConfigDict()
        st_block = ev.stochastic

        results: Dict[str, GenerationBatch] = {}
        for spec in sampling_specs:
            tag = str(spec["tag"])
            if tag in results:
                raise ValueError(f"duplicate sampling spec tag '{tag}'")
            # The sampler reads its churn settings from cfg.evaluation.stochastic: write
            # this spec's values there (defaults for any it omits), restore afterwards.
            backup = {k: (hasattr(st_block, k), getattr(st_block, k, None)) for k, _ in _STOCHASTIC_KEYS.values()}
            for spec_k, (cfg_k, default) in _STOCHASTIC_KEYS.items():
                setattr(st_block, cfg_k, spec.get(spec_k, default))
            if bool(spec.get("stochastic_enabled", False)) and float(st_block.s_churn) <= 0.0:
                raise RuntimeError(f"stochastic spec with s_churn <= 0: {tag}")

            tag_hash = _stable_int_hash(tag)
            try:
                bundle = create_sampler(self.cfg, model, proc, str(spec["sampler_name"]))
                _sync_device(device)
                t0 = time.perf_counter()
                gen_chunks = []
                for i in range(0, n_local, local_bs):
                    end = min(i + local_bs, n_local)
                    chunk_seed = (int(seed) * 1_000_003 + tag_hash * 10_007 + int(global_offset + i)) % MOD
                    with torch.random.fork_rng(devices=devices_for_fork, enabled=True):
                        torch.manual_seed(int(chunk_seed))
                        if device.type == "cuda":
                            torch.cuda.manual_seed_all(int(chunk_seed))
                        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=dtype):
                            _, probs = bundle.sampler.sample(
                                end - i,
                                S_bits,
                                schedule=bundle.schedule,
                                num_steps=int(spec["num_steps"]),
                                sigma_min_override=float(spec["terminal_sigma"]),
                                sigma_max_override=sigma_max,
                                entropy_run_dir=entropy_run_dir,
                                conditioning_prefix_full=ref_local[i:end],
                                cond_prefix_mask=no_prompt[i:end],
                                return_probs=True,
                                progress=(rank == 0 and i == 0),
                            )
                    gen_chunks.append((probs > 0.5).to(torch.uint8))
                    del probs
                gen_bits = torch.cat(gen_chunks, dim=0) if gen_chunks else torch.empty((0, S_bits), device=device, dtype=torch.uint8)
                ref_u8 = (ref_local > 0.5).to(torch.uint8)
                prompt_len = torch.zeros(n_local, device=device, dtype=torch.long)
                _sync_device(device)

                stats = _build_speed_stats(
                    device=device, wall_time_local_sec=time.perf_counter() - t0,
                    num_samples_local=n_local, seq_len_tokens=S_tok, seq_len_model=S_bits,
                )
                gen_g = gather_varlen_firstdim_to_rank0(gen_bits)
                ref_g = gather_varlen_firstdim_to_rank0(ref_u8)
                len_g = gather_varlen_firstdim_to_rank0(prompt_len)
                if rank == 0:
                    results[tag] = GenerationBatch(
                        gen_bits=gen_g.cpu()[:num_samples],
                        ref_bits=ref_g.cpu()[:num_samples],
                        prompt_len_bits=len_g.cpu()[:num_samples],
                        stats=stats,
                    )
            finally:
                for k, (existed, val) in backup.items():
                    if existed:
                        setattr(st_block, k, val)
                    elif k in st_block:
                        delattr(st_block, k)
        return results
