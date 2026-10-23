#!/usr/bin/env python
"""Extract the input token-embedding matrix from an S-FLM (jdeschena/s-flm) checkpoint.

The S-FLM TinyGSM checkpoints are PyTorch-Lightning `.ckpt` files that pickle references to
the S-FLM training module `dataloader` (e.g. `dataloader.VocabSizeTokenizerWrapper`, stored in
`hyper_parameters`). A plain `torch.load` therefore raises `ModuleNotFoundError: dataloader`
unless the whole s-flm training stack is importable. We only want the `state_dict`'s embedding
tensor, so we fabricate any genuinely-missing module as a harmless dummy, load, pull out the
embedding, and re-save it as a CLEAN bare tensor that
scripts/cluster_maps/tinygsm.py can consume via `--checkpoint_path` with no such hazard.

Handles both key layouts observed in the TinyGSM checkpoints:
  * MDLM/Duo/AR : backbone.vocab_embed.embedding        [49154, 768]
  * SFM (sphere): backbone.sphere_embed.weight          [49153, 768]

Usage:
  python scripts/cluster_maps/extract_sflm_embeddings.py \
      --ckpt $WORK/checkpoints/s-flm/tinygsm/sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt \
      --out  $WORK/assets/sflm/sfm_tinygsm_input_embeddings.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import types

import torch


def _fabricate(name: str) -> None:
    """Register a dummy module (and its parents) so unpickling can reference it."""
    class _Dummy:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, s):
            pass

        def __call__(self, *a, **k):
            return self

    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        n = ".".join(parts[:i])
        if n not in sys.modules:
            m = types.ModuleType(n)
            m.__getattr__ = lambda _k: _Dummy  # type: ignore[attr-defined]
            sys.modules[n] = m


def load_ckpt_forgiving(path: str):
    """torch.load a Lightning ckpt, fabricating missing (s-flm) modules on demand."""
    _fabricate("dataloader")  # the one we know about; others discovered iteratively
    for _ in range(32):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as e:
            if not e.name:
                raise
            print(f"[extract] fabricating missing module: {e.name}", flush=True)
            _fabricate(e.name)
    raise RuntimeError(f"could not load {path} after fabricating modules")


def find_embedding(state_dict, *, key: str | None = None) -> tuple[str, torch.Tensor]:
    if key is not None:
        return key, state_dict[key].detach().float().cpu()
    cands = [
        (k, v) for k, v in state_dict.items()
        if hasattr(v, "ndim") and v.ndim == 2 and 40000 <= v.shape[0] <= 60000
    ]
    if not cands:
        raise SystemExit("no embedding-like 2D tensor (rows in [40k,60k]) found in the state dict")
    # Prefer an input-embedding-looking name over the tied output head / lm_head.
    pref = [(k, v) for k, v in cands if any(t in k.lower() for t in ("embed", "vocab", "sphere", "wte", "word"))]
    pref = [kv for kv in pref if "head" not in kv[0].lower() and "output" not in kv[0].lower()] or pref
    k, v = (pref or cands)[0]
    return k, v.detach().float().cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="S-FLM .ckpt file (Lightning)")
    ap.add_argument("--out", required=True, help="output bare-tensor .pt path")
    ap.add_argument("--key", default=None, help="explicit state-dict key (optional)")
    args = ap.parse_args()

    ck = load_ckpt_forgiving(args.ckpt)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    key, emb = find_embedding(sd, key=args.key)
    norms = emb.norm(dim=1)
    print(f"[extract] key={key} shape={tuple(emb.shape)} "
          f"row-norm mean={norms.mean():.3f} std={norms.std():.3f} "
          f"(builder unit-normalizes before cosine k-means)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(emb, args.out)
    print(f"[extract] saved bare embedding tensor -> {args.out}")


if __name__ == "__main__":
    main()
