"""
Robust loader for a pretrained model's input token-embedding matrix.

Custom diffusion-LM backbones (e.g. MDLM, LLaDA) do not always expose their
token embeddings through ``get_input_embeddings()`` — MDLM keeps them at
``backbone.vocab_embed.embedding`` and is loaded with ``AutoModelForMaskedLM``.
This module mirrors the extraction logic used in the mdm-prime subtokenizer
(``text/subtokenizer/evaluate_semantic_attribution.py``) so the same checkpoints
can drive the cluster-bit guidance here.

Returns a CPU float tensor of shape [vocab_size, embed_dim].
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


# Embedding tensors are looked up under these names inside a checkpoint state dict.
_PREFERRED_STATE_KEYS = [
    "backbone.vocab_embed.embedding",
    "backbone.vocab_embed.embedding.weight",
    "model.backbone.vocab_embed.embedding",
    "model.backbone.vocab_embed.embedding.weight",
    "vocab_embed.embedding",
    "vocab_embed.embedding.weight",
    "model.embed_tokens.weight",
    "transformer.wte.weight",
    "backbone.transformer.wte.weight",
]

_STATE_KEY_SUFFIXES = (
    "vocab_embed.embedding",
    "vocab_embed.embedding.weight",
    "transformer.wte.weight",
    "embed_tokens.weight",
    "word_embeddings.weight",
)

# Attribute chains tried on a loaded model object before falling back to
# get_input_embeddings().
_EXPLICIT_MODEL_PATHS = [
    ("vocab_embed", "embedding"),
    ("backbone", "vocab_embed", "embedding"),
    ("transformer", "wte", "weight"),
    ("backbone", "transformer", "wte", "weight"),
    ("model", "embed_tokens", "weight"),
    ("embeddings", "word_embeddings", "weight"),
]


def _extract_from_state_dict(
    state,
    *,
    key: Optional[str] = None,
    expected_vocab_size: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if isinstance(state, torch.Tensor) and state.ndim == 2:
        return state
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        return None

    if key is not None:
        value = state.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return value
        return None

    for k in _PREFERRED_STATE_KEYS:
        value = state.get(k)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            if expected_vocab_size is not None and int(value.size(0)) != int(expected_vocab_size):
                continue
            return value

    for k, value in state.items():
        if isinstance(value, torch.Tensor) and value.ndim == 2 and any(
            k.endswith(suffix) for suffix in _STATE_KEY_SUFFIXES
        ):
            if expected_vocab_size is not None and int(value.size(0)) != int(expected_vocab_size):
                continue
            return value

    candidates = [v for v in state.values() if isinstance(v, torch.Tensor) and v.ndim == 2]
    if not candidates:
        return None
    if expected_vocab_size is not None:
        for value in candidates:
            if int(value.size(0)) == int(expected_vocab_size):
                return value
    vocab_like = [v for v in candidates if v.size(0) >= 10000]
    pool = vocab_like if vocab_like else candidates
    pool = sorted(pool, key=lambda t: (t.size(0), t.size(1)), reverse=True)
    return pool[0]


def _extract_from_model(model) -> Optional[torch.Tensor]:
    for chain in _EXPLICIT_MODEL_PATHS:
        current = model
        ok = True
        for attr in chain:
            if hasattr(current, attr):
                current = getattr(current, attr)
            else:
                ok = False
                break
        if ok and isinstance(current, torch.Tensor) and current.ndim == 2:
            return current.detach()

    if hasattr(model, "get_input_embeddings"):
        emb_layer = model.get_input_embeddings()
        if emb_layer is not None and hasattr(emb_layer, "weight"):
            return emb_layer.weight.detach()
    return None


def _load_model(source: str, *, trust_remote_code: bool = True):
    from transformers import AutoModel, AutoModelForMaskedLM

    name = str(source).lower()
    # Masked / diffusion LMs (MDLM, LangFlow, etc.) expose embeddings via the
    # MaskedLM head and are loaded with AutoModelForMaskedLM.
    masked_first = (
        ("md" in name)
        or ("llada" in name)
        or ("mask" in name)
        or ("bert" in name)
        or ("langflow" in name)
        or ("flow" in name)
    )
    order = (AutoModelForMaskedLM, AutoModel) if masked_first else (AutoModel, AutoModelForMaskedLM)

    last_err = None
    for loader in order:
        if loader is None:
            continue
        try:
            return loader.from_pretrained(source, trust_remote_code=bool(trust_remote_code))
        except Exception as e:  # noqa: BLE001 - try the next loader
            last_err = e
    raise RuntimeError(f"Could not load model '{source}' via AutoModel/AutoModelForMaskedLM: {last_err}")


def load_pretrained_token_embeddings(
    *,
    model_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    from_huggingface: bool = False,
    hf_repo_id: Optional[str] = None,
    embeddings_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    expected_vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Load a pretrained input token-embedding matrix [vocab_size, embed_dim].

    Resolution order:
      1. If ``model_name`` is given, load the model and read its embedding tensor
         (custom backbone paths first, then ``get_input_embeddings()``).
      2. Otherwise, read a checkpoint state dict:
         - ``checkpoint_path`` as a hub filename when ``from_huggingface`` is set
           (downloaded from ``hf_repo_id``), or
         - ``checkpoint_path`` as a local ``.pt``/``.safetensors``-style file.
    """
    # ---- (1) live model ----
    source = model_name or (hf_repo_id if from_huggingface else None)
    if source is not None:
        try:
            model = _load_model(source, trust_remote_code=trust_remote_code)
            emb = _extract_from_model(model)
            if emb is not None:
                return emb.detach().to("cpu").float()
        except Exception as e:  # noqa: BLE001 - fall back to checkpoint parsing
            if not checkpoint_path:
                raise

    # ---- (2) checkpoint state dict ----
    if checkpoint_path:
        if from_huggingface:
            from huggingface_hub import hf_hub_download

            if hf_repo_id is None:
                raise ValueError("--hf_repo_id is required when --from_huggingface is set.")
            ckpt = hf_hub_download(repo_id=hf_repo_id, filename=checkpoint_path, cache_dir=cache_dir)
        else:
            ckpt = checkpoint_path
            if not Path(ckpt).exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        state = torch.load(ckpt, map_location="cpu")
        emb = _extract_from_state_dict(state, key=embeddings_key, expected_vocab_size=expected_vocab_size)
        if emb is not None:
            return emb.detach().to("cpu").float()

    raise ValueError(
        "Could not load pretrained token embeddings. Provide a valid --pretrained_model "
        "(HF id) and/or --checkpoint_path (with --from_huggingface + --hf_repo_id for hub files)."
    )
