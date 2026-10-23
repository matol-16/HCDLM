"""Bitstreams -> token ids -> text, for evaluation.

Each token is a patch of bits_per_token bits: its id (MSB first), then for H-CoBit its
cluster bits, which are dropped before decoding.

LM1B: the ids are BERT ids (wrapped modulo the vocabulary size).
OpenWebText: the ids are gpt2id_bpe16 codes. A row is read up to its first code-level
PAD / EOSEQ, its codes are expanded back to GPT-2 ids, one leading and one trailing
GPT-2 EOS are removed, and the row keeps its true length (EOS padding would otherwise
leak into the text and deflate its entropy).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def extract_dataset_attr(ds, name: str):
    if hasattr(ds, name):
        return getattr(ds, name)
    if hasattr(ds, "dataset"):
        return extract_dataset_attr(ds.dataset, name)
    return None


def _cluster_bits_from_cfg(cfg) -> int:
    d = cfg.data
    if not bool(getattr(d, "cluster_bits_enabled", False)):
        return 0
    return int(math.ceil(math.log2(max(2, int(d.cluster_bits_n_clusters)))))


def _vocab_size(tokenizer) -> int:
    v = getattr(tokenizer, "vocab_size", None)
    return int(v) if v is not None else int(tokenizer.get_vocab_size())


@torch.no_grad()
def bitstreams_to_token_ids_raw_binary(bits: torch.Tensor, *, bits_per_token: int, cfg=None) -> torch.Tensor:
    """[B, S] bits -> [B, T] token ids (cluster bits dropped)."""
    if bits.dim() == 1:
        bits = bits.unsqueeze(0)
    bits = bits.long()
    cluster_bits = _cluster_bits_from_cfg(cfg) if cfg is not None else 0
    m = int(bits_per_token) - cluster_bits
    B, S = bits.shape
    T = S // (m + cluster_bits)
    data_bits = bits[:, : T * (m + cluster_bits)].reshape(B, T, m + cluster_bits)[:, :, :m]
    powers = 2 ** torch.arange(m - 1, -1, -1, device=bits.device)
    return (data_bits * powers).sum(dim=-1).to(torch.long).cpu()


def _tokenizer(cfg, dataset_obj):
    tok = extract_dataset_attr(dataset_obj, "tokenizer") if dataset_obj is not None else None
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(cfg.data.tokenizer_name), use_fast=True)
    return tok


def _is_owt_codec(cfg) -> bool:
    return str(getattr(cfg.data, "sequence_codec", "base")).lower() == "gpt2id_bpe16"


# -----------------------------------------------------------------------------
# OpenWebText gpt2id_bpe16 codec
# -----------------------------------------------------------------------------

def load_openwebtext_gpt2id_bpe16_assets(root: Path, code_tokenizer_path: str,
                                         code_tokenizer_meta_path: Optional[str], tokenizer_name: str = "gpt2"):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tok_path = Path(code_tokenizer_path)
    tok_path = tok_path if tok_path.is_absolute() else Path(root) / tok_path
    meta_path = Path(code_tokenizer_meta_path) if code_tokenizer_meta_path else tok_path.with_suffix(".meta.json")
    meta_path = meta_path if meta_path.is_absolute() else Path(root) / meta_path
    gpt2_tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if gpt2_tok.pad_token_id is None:
        gpt2_tok.pad_token = gpt2_tok.eos_token
    return gpt2_tok, Tokenizer.from_file(str(tok_path)), json.loads(meta_path.read_text())


def _owt_assets(cfg, dataset_obj):
    names = ("tokenizer", "code_tokenizer", "code_meta")
    assets = [extract_dataset_attr(dataset_obj, n) if dataset_obj is not None else None for n in names]
    if any(a is None for a in assets):
        d = cfg.data
        assets = load_openwebtext_gpt2id_bpe16_assets(
            Path(d.root), str(d.code_tokenizer_path), getattr(d, "code_tokenizer_meta_path", None), str(d.tokenizer_name)
        )
    return assets


def _code_row_to_gpt2_ids(row: List[int], *, code_tokenizer, code_meta: Dict[str, Any], eos_id: int,
                          vocab_gpt2: int, vocab_code: int) -> List[int]:
    pad_id, eoseq_id, unk_id = (code_meta.get(k) for k in ("pad_id", "eoseq_id", "unk_id"))
    kept = []
    for tid in row:
        if (pad_id is not None and tid == int(pad_id)) or (eoseq_id is not None and tid == int(eoseq_id)):
            break  # the code level is the only place a row stops
        if 0 <= tid < vocab_code or (unk_id is not None and tid == int(unk_id)):
            kept.append(tid)
    if not kept:
        return []
    pua = code_tokenizer.decode(kept, skip_special_tokens=False) or ""
    offset = int(code_meta["base_char_offset"])
    ids = [ord(ch) - offset for ch in str(pua) if 0 <= ord(ch) - offset < vocab_gpt2]
    if ids and ids[0] == eos_id:  # synthetic leading EOS
        ids = ids[1:]
    if ids and ids[-1] == eos_id:  # synthetic trailing EOS
        ids = ids[:-1]
    return ids  # internal EOS tokens are kept


@torch.no_grad()
def code_ids_to_gpt2_token_id_lists_for_eval(code_ids: torch.Tensor, *, gpt2_tokenizer, code_tokenizer,
                                             code_meta: Dict[str, Any], return_debug_stats: bool = False):
    if code_ids.dim() == 1:
        code_ids = code_ids.unsqueeze(0)
    eos_id = int(gpt2_tokenizer.eos_token_id)
    vocab_code = int(code_meta.get("actual_code_vocab_size", _vocab_size(code_tokenizer)))
    rows = [
        _code_row_to_gpt2_ids([int(t) for t in r.tolist()], code_tokenizer=code_tokenizer, code_meta=code_meta,
                              eos_id=eos_id, vocab_gpt2=_vocab_size(gpt2_tokenizer), vocab_code=vocab_code)
        for r in code_ids.long().cpu()
    ]
    return (rows, None) if return_debug_stats else rows


# -----------------------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------------------

@torch.no_grad()
def decode_bitstreams_to_token_ids_for_eval(cfg, samples_bits: torch.Tensor, *, dataset_obj=None) -> torch.Tensor:
    """LM1B: [B, T] BERT ids of a batch of bitstreams."""
    if _is_owt_codec(cfg):
        raise ValueError("use code_ids_to_gpt2_token_id_lists_for_eval for the OpenWebText codec")
    tok = _tokenizer(cfg, dataset_obj)
    ids = bitstreams_to_token_ids_raw_binary(samples_bits, bits_per_token=int(cfg.data.bits_per_token), cfg=cfg)
    return ids % _vocab_size(tok)


@torch.no_grad()
def decode_bitstreams_for_eval(cfg, samples_bits: torch.Tensor, *, prompt_len_bits=None, mode: str = "full",
                               dataset_obj=None) -> List[str]:
    """Texts of a batch of unconditional bitstreams (mode='full')."""
    if mode != "full":
        raise ValueError("only full-text decoding is used")
    if samples_bits.dim() == 1:
        samples_bits = samples_bits.unsqueeze(0)
    bits = (samples_bits.clamp(0, 1) > 0.5).long() if samples_bits.is_floating_point() else (samples_bits != 0).long()
    if _is_owt_codec(cfg):
        gpt2_tok, code_tok, code_meta = _owt_assets(cfg, dataset_obj)
        code_ids = bitstreams_to_token_ids_raw_binary(bits, bits_per_token=int(cfg.data.bits_per_token), cfg=cfg)
        rows = code_ids_to_gpt2_token_id_lists_for_eval(code_ids, gpt2_tokenizer=gpt2_tok, code_tokenizer=code_tok,
                                                        code_meta=code_meta)
        return [str(gpt2_tok.decode(r, skip_special_tokens=False) or "") for r in rows]
    tok = _tokenizer(cfg, dataset_obj)
    ids = decode_bitstreams_to_token_ids_for_eval(cfg, bits, dataset_obj=dataset_obj)
    return tok.batch_decode(ids.tolist(), skip_special_tokens=False)
