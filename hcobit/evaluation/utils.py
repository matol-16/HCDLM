from pathlib import Path
from typing import Any, Tuple

import torch

from utils.ema import EMA


def resolve_eval_amp(device: torch.device, cfg: Any) -> Tuple[bool, torch.dtype]:
    """(autocast enabled, dtype) from cfg.evaluation.use_amp / amp_dtype."""
    if device.type != "cuda" or not bool(getattr(cfg.evaluation, "use_amp", True)):
        return False, torch.float32
    dtype = str(getattr(cfg.evaluation, "amp_dtype", "bf16")).lower()
    return True, (torch.float16 if dtype in {"fp16", "float16"} else torch.bfloat16)


def unwrap_all(model: torch.nn.Module) -> torch.nn.Module:
    """Peel DDP and torch.compile wrappers."""
    m = model
    while True:
        if hasattr(m, "module"):
            m = m.module
        elif hasattr(m, "_orig_mod"):
            m = m._orig_mod
        else:
            return m


def load_checkpoint(model: torch.nn.Module, ema: EMA, ckpt_path: Path, device: torch.device, *, apply_ema: bool = True):
    """Load ckpt["model"] and, if present and requested, copy the EMA weights over it.

    Released checkpoints already hold the EMA weights (no "ema" key); full training
    checkpoints carry both.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {}
    for k, v in ckpt["model"].items():
        k = k.replace("_orig_mod.", "")
        state[k[7:] if k.startswith("module.") else k] = v
    unwrap_all(model).load_state_dict(state, strict=True)
    if apply_ema and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
        ema.to(device)
        ema.apply(model)
