from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int


def is_torchrun_env() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and ("LOCAL_RANK" in os.environ)


def init_distributed_if_needed(backend: str = "nccl") -> DistInfo:
    """
    Initialize torch.distributed if launched under torchrun.

    Notes:
    - We use the standard init_process_group(...) path after setting CUDA device.
    - We intentionally do not pass device_id=... here. The simpler, conventional
      initialization path is less brittle for debugging first-collective issues.
    - Timeout remains 20 minutes because rank 0 may spend substantial time in
      single-rank evaluation work (for example HF external PPL scoring).
    """
    if not is_torchrun_env():
        return DistInfo(enabled=False, rank=0, local_rank=0, world_size=1)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    if dist.is_available() and not dist.is_initialized():
        timeout = datetime.timedelta(minutes=20)
        dist.init_process_group(
            backend=backend,
            timeout=timeout,
        )

    return DistInfo(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def barrier() -> None:
    """
    Plain process-group barrier.

    We intentionally avoid dist.barrier(device_ids=[...]) here while debugging
    first-collective NCCL issues. A plain barrier is the simplest and most
    portable synchronization primitive.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    dist.barrier()


def set_global_seed(seed: int, rank: int = 0) -> None:
    """
    Deterministic-ish seeding for eval generation to reduce duplicates across ranks.
    """
    s = int(seed) + 1000 * int(rank)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)