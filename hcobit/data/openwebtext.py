"""OpenWebText as bitstreams of gpt2id_bpe16 codes (CoBit's codec).

Documents are GPT-2 tokenized (an EOS after each) into one flat stream; the stream is cut
into 1022-token windows framed as [BOS] window [EOS], and each framed window is
re-encoded by the second-stage BPE gpt2id_bpe16 into a fixed 1024-code row (EOSEQ then
PAD). Each code is written in 16 bits (MSB first), followed for H-CoBit by its cluster
id. The last 100k documents of the train split are the validation (= test) split.
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from filelock import FileLock
from ml_collections import config_dict
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from data.lm1b import _ceil_log2, token_bits_table


def _safe_name(s: str) -> str:
    return str(s).replace("/", "_").replace("-", "_").replace(":", "_")


def _ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _ddp_rank_world() -> tuple[int, int]:
    if not _ddp_is_on():
        return 0, 1
    return int(dist.get_rank()), int(dist.get_world_size())


def _is_rank0() -> bool:
    return _ddp_rank_world()[0] == 0


def _dist_barrier() -> None:
    if _ddp_is_on():
        dist.barrier()


def load_gpt2_tokenizer(tokenizer_name: str = "gpt2"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_id = int(tok.eos_token_id)
    bos_id = int(tok.bos_token_id) if tok.bos_token_id is not None else eos_id
    return tok, bos_id, eos_id


def load_cluster_map(map_dir: Path, tokenizer_name: str, code_vocab_size: int) -> Tuple[Dict[int, int], int]:
    """(code id -> cluster id, n_clusters) for the gpt2id_bpe16 code vocabulary."""
    path = Path(map_dir) / f"cluster_mapping_{_safe_name(tokenizer_name)}_gpt2id_bpe16_v{int(code_vocab_size)}.json"
    data = json.loads(path.read_text())
    return {int(k): int(v) for k, v in data["token_to_cluster"].items()}, int(data["n_clusters"])


def load_gpt2id_bpe16_tokenizer_and_meta(root: Path, tokenizer_path: str, meta_path: Optional[str] = None):
    from tokenizers import Tokenizer

    tok_path = Path(tokenizer_path)
    tok_path = tok_path if tok_path.is_absolute() else Path(root) / tok_path
    meta_file = Path(meta_path) if meta_path else tok_path.with_suffix(".meta.json")
    meta_file = meta_file if meta_file.is_absolute() else Path(root) / meta_file
    if not tok_path.exists() or not meta_file.exists():
        raise RuntimeError(f"Missing code tokenizer {tok_path} (copy assets/owt_codec/ into the data root)")
    return Tokenizer.from_file(str(tok_path)), json.loads(meta_file.read_text())


# -----------------------------------------------------------------------------
# Flat token cache from HF openwebtext
# -----------------------------------------------------------------------------

def _load_memmap(cache_path: Path, meta_path: Path) -> tuple[np.memmap, dict]:
    if not cache_path.exists() or not meta_path.exists():
        raise RuntimeError(f"Missing cache files: {cache_path} / {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n = int(meta["n_tokens"])
    mm = np.memmap(cache_path, dtype=np.uint16, mode="r", shape=(n,))
    return mm, meta


def build_or_load_openwebtext_flat_cache(
    *,
    root: Path,
    split_name: str,
    hf_split: str,
    tokenizer_name: str = "gpt2",
    insert_eos: bool = True,
    encode_batch_size: int = 1000,
    write_batch_size: int = 8192,
    num_proc: Optional[int] = None,
    overwrite: bool = False,
) -> tuple[np.memmap, dict]:
    """
    DDP-safe builder/loader for a flat uint16 OpenWebText token cache.
    """
    from datasets import load_dataset
    from tqdm.auto import tqdm

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    safe_tok = _safe_name(tokenizer_name)
    cache_path = root / f"cache_{split_name}_{safe_tok}_flat_eos{int(insert_eos)}.uint16"
    meta_path = root / f"cache_{split_name}_{safe_tok}_flat_eos{int(insert_eos)}.meta.json"
    tmp_path = root / f"cache_{split_name}_{safe_tok}_flat_eos{int(insert_eos)}.uint16.tmp"
    lock_path = root / f"cache_{split_name}_{safe_tok}_flat_eos{int(insert_eos)}.lock"

    if (not overwrite) and cache_path.exists() and meta_path.exists():
        print(f"[owt_cache] Reusing existing cache: {cache_path}")
        return _load_memmap(cache_path, meta_path)

    tok0, bos_id, eos_id = load_gpt2_tokenizer(tokenizer_name)
    vocab_size = int(tok0.vocab_size)

    if vocab_size > 65535:
        raise RuntimeError(f"Tokenizer vocab_size={vocab_size} too large for uint16 storage.")

    if num_proc is None:
        cpu_total = mp.cpu_count()
        fallback = cpu_total // 2 if cpu_total >= 8 else max(1, cpu_total - 1)
        num_proc = max(1, min(16, fallback))
    num_proc = max(1, int(num_proc))

    with FileLock(str(lock_path)):
        if (not overwrite) and cache_path.exists() and meta_path.exists():
            print(f"[owt_cache] Reusing existing cache after waiting: {cache_path}")
            return _load_memmap(cache_path, meta_path)

        if overwrite:
            if tmp_path.exists():
                print(f"[owt_cache] Removing stale temporary file: {tmp_path}")
                tmp_path.unlink(missing_ok=True)
            if cache_path.exists():
                print(f"[owt_cache] overwrite=True -> removing existing cache: {cache_path}")
                cache_path.unlink(missing_ok=True)
            if meta_path.exists():
                print(f"[owt_cache] overwrite=True -> removing existing meta: {meta_path}")
                meta_path.unlink(missing_ok=True)

        if tmp_path.exists():
            print(f"[owt_cache] Removing stale temporary file from interrupted run: {tmp_path}")
            tmp_path.unlink(missing_ok=True)

        print(
            f"[owt_cache] Loading HF dataset split={hf_split!r} into {root / 'hf_cache'} "
            f"(split_name={split_name}, tokenizer={tokenizer_name}, num_proc={num_proc})"
        )

        t0 = time.time()
        ds = load_dataset(
            "openwebtext",
            split=hf_split,
            cache_dir=str(root / "hf_cache"),
        )
        t_load = time.time() - t0

        try:
            n_examples = len(ds)
        except Exception:
            n_examples = None

        if n_examples is not None:
            print(f"[owt_cache] Loaded dataset with {n_examples:,} examples in {t_load:.1f}s")
        else:
            print(f"[owt_cache] Loaded dataset in {t_load:.1f}s")

        _TOK_CACHE = {"tok": None}

        def tokenize_batch(examples):
            tok = _TOK_CACHE["tok"]
            if tok is None:
                tok, _, _ = load_gpt2_tokenizer(tokenizer_name)
                _TOK_CACHE["tok"] = tok

            texts = [
                (t or "").replace("\r\n", "\n").replace("\r", "\n").strip()
                for t in examples["text"]
            ]

            enc = tok(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            out_ids = []
            for ids in enc["input_ids"]:
                if not ids:
                    out_ids.append([])
                else:
                    out_ids.append(ids + [eos_id] if insert_eos else ids)
            return {"input_ids": out_ids}

        print(
            f"[owt_cache] Tokenizing split={split_name} "
            f"(batch_size={encode_batch_size}, num_proc={num_proc})..."
        )
        t1 = time.time()

        tokenized_ds = ds.map(
            tokenize_batch,
            batched=True,
            batch_size=int(encode_batch_size),
            num_proc=num_proc,
            remove_columns=ds.column_names,
            desc=f"Tokenizing {split_name}",
            load_from_cache_file=True,
            keep_in_memory=False,
        )

        t_tok = time.time() - t1
        print(f"[owt_cache] Tokenization finished in {t_tok / 60.0:.2f} min")

        n_tokens = 0
        n_docs = 0

        print(
            f"[owt_cache] Flattening + writing uint16 stream to {cache_path.name} "
            f"(write_batch_size={write_batch_size})..."
        )
        t2 = time.time()

        total_rows = len(tokenized_ds) if hasattr(tokenized_ds, "__len__") else None

        try:
            with open(tmp_path, "wb") as fout:
                row_bar = tqdm(
                    total=total_rows,
                    desc=f"Writing {split_name}",
                    unit="docs",
                    dynamic_ncols=True,
                )

                for batch in tokenized_ds.iter(batch_size=int(write_batch_size)):
                    ids_batch = batch["input_ids"]

                    valid_docs = [ids for ids in ids_batch if ids]
                    if valid_docs:
                        flat_ids = list(itertools.chain.from_iterable(valid_docs))
                        arr = np.asarray(flat_ids, dtype=np.uint16)
                        fout.write(arr.tobytes())
                        n_tokens += int(arr.size)
                        n_docs += int(len(valid_docs))

                    row_bar.update(len(ids_batch))
                    row_bar.set_postfix(
                        docs=f"{n_docs:,}",
                        toks=f"{n_tokens:,}",
                    )

                row_bar.close()

            if n_tokens <= 0:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"No tokens written for split_name={split_name}, hf_split={hf_split}")

            os.replace(tmp_path, cache_path)

            t_write = time.time() - t2
            t_total = time.time() - t0

            meta = {
                "dataset_name": "openwebtext",
                "split_name": split_name,
                "hf_split": hf_split,
                "tokenizer_name": tokenizer_name,
                "vocab_size": vocab_size,
                "n_tokens": int(n_tokens),
                "n_docs": int(n_docs),
                "insert_eos": bool(insert_eos),
                "bos_token_id": int(bos_id),
                "eos_token_id": int(eos_id),
                "num_proc": int(num_proc),
                "encode_batch_size": int(encode_batch_size),
                "write_batch_size": int(write_batch_size),
                "build_seconds_total": float(t_total),
                "build_seconds_load_dataset": float(t_load),
                "build_seconds_tokenize": float(t_tok),
                "build_seconds_write": float(t_write),
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            toks_per_sec = n_tokens / max(t_total, 1e-8)
            docs_per_sec = n_docs / max(t_total, 1e-8)
            print(
                f"[owt_cache] Done. docs={n_docs:,} toks={n_tokens:,} "
                f"time={t_total / 60.0:.2f} min "
                f"({docs_per_sec:.1f} docs/s, {toks_per_sec:.1f} toks/s)"
            )

        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        return _load_memmap(cache_path, meta_path)


# -----------------------------------------------------------------------------
# Fixed-length second-stage code cache for gpt2id_bpe16
# -----------------------------------------------------------------------------

def build_or_load_openwebtext_code_cache(
    *,
    root: Path,
    split_name: str,
    base_mm: np.memmap,
    base_meta: dict,
    tokenizer_name: str,
    code_tokenizer_path: str,
    code_tokenizer_meta_path: Optional[str],
    base_sequence_len_tokens: int,
    code_sequence_len_tokens: int,
    batch_size: int = 2048,
    overwrite: bool = False,
) -> tuple[np.memmap, dict]:
    from tqdm.auto import tqdm

    root = Path(root)
    safe_tok = _safe_name(tokenizer_name)

    code_tok, code_meta = load_gpt2id_bpe16_tokenizer_and_meta(
        root=root,
        tokenizer_path=code_tokenizer_path,
        meta_path=code_tokenizer_meta_path,
    )

    code_vocab_size = int(code_meta["actual_code_vocab_size"])
    if code_vocab_size > 65536:
        raise RuntimeError(f"Code vocab too large for uint16 cache: {code_vocab_size}")

    pad_id = int(code_meta["pad_id"])
    eoseq_id = int(code_meta["eoseq_id"])
    base_char_offset = int(code_meta["base_char_offset"])

    content_len = int(base_sequence_len_tokens) - 2
    if content_len <= 0:
        raise RuntimeError("base_sequence_len_tokens must be >= 3")

    n_base_tokens = int(base_mm.shape[0])
    n_seq = n_base_tokens // content_len
    if n_seq <= 0:
        raise RuntimeError("No sequences available for code-cache construction")

    cache_path = root / (
        f"cache_{split_name}_{safe_tok}_gpt2id_bpe16_"
        f"v{code_vocab_size}_base{base_sequence_len_tokens}_len{code_sequence_len_tokens}.uint16"
    )
    meta_path = cache_path.with_suffix(".meta.json")
    lock_path = root / (cache_path.name + ".lock")
    tmp_path = root / (cache_path.name + ".tmp")

    if (not overwrite) and cache_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        mm = np.memmap(
            cache_path,
            dtype=np.uint16,
            mode="r",
            shape=(int(meta["num_sequences"]), int(meta["code_sequence_len_tokens"])),
        )
        print(f"[owt_code_cache] Reusing existing cache: {cache_path}")
        return mm, meta

    with FileLock(str(lock_path)):
        if (not overwrite) and cache_path.exists() and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            mm = np.memmap(
                cache_path,
                dtype=np.uint16,
                mode="r",
                shape=(int(meta["num_sequences"]), int(meta["code_sequence_len_tokens"])),
            )
            print(f"[owt_code_cache] Reusing existing cache after waiting: {cache_path}")
            return mm, meta

        if overwrite:
            cache_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            tmp_path.unlink(missing_ok=True)

        if tmp_path.exists():
            print(f"[owt_code_cache] Removing stale temporary file: {tmp_path}")
            tmp_path.unlink(missing_ok=True)

        bos_id = int(base_meta["bos_token_id"])
        eos_id = int(base_meta["eos_token_id"])
        bos_ch = chr(base_char_offset + bos_id)
        eos_ch = chr(base_char_offset + eos_id)

        print(
            f"[owt_code_cache] Building code cache for split={split_name} "
            f"num_sequences={n_seq:,}, batch_size={int(batch_size)} -> {cache_path.name}"
        )

        arr = np.memmap(
            tmp_path,
            dtype=np.uint16,
            mode="w+",
            shape=(n_seq, int(code_sequence_len_tokens)),
        )

        try:
            for start in tqdm(
                range(0, n_seq, int(batch_size)),
                desc=f"Encoding {split_name} code-cache",
                dynamic_ncols=True,
            ):
                end = min(start + int(batch_size), n_seq)

                batch_strings = []
                for i in range(start, end):
                    s = i * content_len
                    e = s + content_len
                    content = base_mm[s:e]
                    body = "".join(map(chr, content.astype(np.uint32, copy=False) + base_char_offset))
                    batch_strings.append(bos_ch + body + eos_ch)

                encs = code_tok.encode_batch(batch_strings)

                for j, enc in enumerate(encs):
                    ids = list(enc.ids)
                    ids.append(eoseq_id)

                    if len(ids) > int(code_sequence_len_tokens):
                        ids = ids[: int(code_sequence_len_tokens)]
                        ids[-1] = eoseq_id

                    if len(ids) < int(code_sequence_len_tokens):
                        ids.extend([pad_id] * (int(code_sequence_len_tokens) - len(ids)))

                    arr[start + j, :] = np.asarray(ids, dtype=np.uint16)

            arr.flush()
            del arr
            os.replace(tmp_path, cache_path)

        except Exception:
            try:
                del arr
            except Exception:
                pass
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        meta = {
            "format": "openwebtext_gpt2id_bpe16_cache",
            "split_name": split_name,
            "num_sequences": int(n_seq),
            "base_sequence_len_tokens": int(base_sequence_len_tokens),
            "code_sequence_len_tokens": int(code_sequence_len_tokens),
            "code_vocab_size": int(code_vocab_size),
            "pad_id": int(pad_id),
            "eoseq_id": int(eoseq_id),
            "source_flat_hf_split": base_meta.get("hf_split"),
            "source_flat_cache_tokens": int(base_mm.shape[0]),
            "code_tokenizer_path": str(code_tokenizer_path),
            "code_tokenizer_meta_path": str(code_tokenizer_meta_path) if code_tokenizer_meta_path is not None else None,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        mm = np.memmap(
            cache_path,
            dtype=np.uint16,
            mode="r",
            shape=(int(meta["num_sequences"]), int(meta["code_sequence_len_tokens"])),
        )
        print(f"[owt_code_cache] Finished building: {cache_path}")
        return mm, meta


# -----------------------------------------------------------------------------
# Cache prebuild coordinator
# -----------------------------------------------------------------------------

def ensure_openwebtext_caches_ready(
    config: config_dict.ConfigDict,
    *,
    split: str,
) -> None:
    """
    Ensure all caches required for this split exist.

    In DDP:
      - rank 0 builds if needed
      - all ranks barrier
      - nonzero ranks then proceed to load only

    Outside DDP:
      - build/load directly
    """
    assert split in {"train", "val", "test"}

    root = Path(getattr(config.data, "root", "datasets/openwebtext"))
    tokenizer_name = str(getattr(config.data, "tokenizer_name", "gpt2"))

    sequence_codec = str(getattr(config.data, "sequence_codec", "base")).lower().strip()

    train_split = str(getattr(config.data, "train_split", "train[:-100000]"))
    valid_split = str(getattr(config.data, "valid_split", "train[-100000:]"))
    insert_train_eos = bool(getattr(config.data, "insert_train_eos", True))
    insert_valid_eos = bool(getattr(config.data, "insert_valid_eos", True))
    encode_batch_size = int(getattr(config.data, "cache_encode_batch_size", 1000))
    write_batch_size = int(getattr(config.data, "cache_write_batch_size", 8192))
    cache_num_proc = getattr(config.data, "cache_num_proc", None)
    cache_overwrite = bool(getattr(config.data, "cache_overwrite", False))

    base_sequence_len_tokens = int(
        getattr(config.data, "base_sequence_len_tokens", getattr(config.data, "sequence_len_tokens", 1024))
    )
    code_sequence_len_tokens = int(getattr(config.data, "sequence_len_tokens", 1024))
    code_cache_batch_size = int(getattr(config.data, "code_cache_batch_size", 2048))
    code_cache_overwrite = bool(getattr(config.data, "code_cache_overwrite", False))

    if split == "train":
        split_name = "train"
        hf_split = train_split
        insert_eos = insert_train_eos
    else:
        split_name = "val"
        hf_split = valid_split
        insert_eos = insert_valid_eos

    def _build_all():
        base_mm, base_meta = build_or_load_openwebtext_flat_cache(
            root=root,
            split_name=split_name,
            hf_split=hf_split,
            tokenizer_name=tokenizer_name,
            insert_eos=insert_eos,
            encode_batch_size=encode_batch_size,
            write_batch_size=write_batch_size,
            num_proc=cache_num_proc,
            overwrite=cache_overwrite,
        )

        if sequence_codec == "gpt2id_bpe16":
            code_tokenizer_path = str(getattr(config.data, "code_tokenizer_path"))
            code_tokenizer_meta_path = getattr(config.data, "code_tokenizer_meta_path", None)

            build_or_load_openwebtext_code_cache(
                root=root,
                split_name=split_name,
                base_mm=base_mm,
                base_meta=base_meta,
                tokenizer_name=tokenizer_name,
                code_tokenizer_path=code_tokenizer_path,
                code_tokenizer_meta_path=code_tokenizer_meta_path,
                base_sequence_len_tokens=base_sequence_len_tokens,
                code_sequence_len_tokens=code_sequence_len_tokens,
                batch_size=code_cache_batch_size,
                overwrite=code_cache_overwrite,
            )

    if not _ddp_is_on():
        _build_all()
        return

    rank, world = _ddp_rank_world()
    if _is_rank0():
        print(f"[owt_cache][rank0/{world}] Ensuring caches for split={split} are ready...")
        _build_all()

    _dist_barrier()


class OpenWebTextDataset(Dataset):
    is_text_dataset = True

    def __init__(self, config: config_dict.ConfigDict, *, split: str):
        super().__init__()
        assert split in {"train", "val", "test"}
        d = config.data
        if str(getattr(d, "sequence_codec", "")) != "gpt2id_bpe16":
            raise ValueError("OpenWebText is used with sequence_codec='gpt2id_bpe16'")
        self.split = split
        self.root = Path(d.root)
        self.tokenizer_name = str(d.tokenizer_name)
        self.tokenizer, self.bos_token_id, self.eos_token_id = load_gpt2_tokenizer(self.tokenizer_name)
        self.vocab_size_base = int(self.tokenizer.vocab_size)
        self.base_seq_len_tokens = int(d.base_sequence_len_tokens)
        self.seq_len_tokens = int(d.sequence_len_tokens)
        self.bits_per_token = int(d.bits_per_token)
        self.cluster_bits_enabled = bool(getattr(d, "cluster_bits_enabled", False))
        self.cluster_bits = _ceil_log2(int(d.cluster_bits_n_clusters)) if self.cluster_bits_enabled else 0
        self.data_bits_per_token = self.bits_per_token - self.cluster_bits
        self.seq_len_bits = self.seq_len_tokens * self.bits_per_token

        cache_split = "train" if split == "train" else "val"
        self.mm, self.meta = build_or_load_openwebtext_flat_cache(
            root=self.root,
            split_name=cache_split,
            hf_split=str(d.train_split if split == "train" else d.valid_split),
            tokenizer_name=self.tokenizer_name,
            insert_eos=bool(d.insert_train_eos if split == "train" else d.insert_valid_eos),
            encode_batch_size=int(getattr(d, "cache_encode_batch_size", 1000)),
            write_batch_size=int(getattr(d, "cache_write_batch_size", 8192)),
            num_proc=getattr(d, "cache_num_proc", None),
            overwrite=bool(getattr(d, "cache_overwrite", False)),
        )
        self.code_tokenizer, self.code_meta = load_gpt2id_bpe16_tokenizer_and_meta(
            self.root, str(d.code_tokenizer_path), getattr(d, "code_tokenizer_meta_path", None)
        )
        self.code_vocab_size = int(self.code_meta["actual_code_vocab_size"])
        self.code_mm, self.code_cache_meta = build_or_load_openwebtext_code_cache(
            root=self.root,
            split_name=cache_split,
            base_mm=self.mm,
            base_meta=self.meta,
            tokenizer_name=self.tokenizer_name,
            code_tokenizer_path=str(d.code_tokenizer_path),
            code_tokenizer_meta_path=getattr(d, "code_tokenizer_meta_path", None),
            base_sequence_len_tokens=self.base_seq_len_tokens,
            code_sequence_len_tokens=self.seq_len_tokens,
            batch_size=int(getattr(d, "code_cache_batch_size", 2048)),
            overwrite=bool(getattr(d, "code_cache_overwrite", False)),
        )
        self.num_sequences = int(self.code_mm.shape[0])

        token_to_cluster = None
        if self.cluster_bits_enabled:
            token_to_cluster, n = load_cluster_map(Path(d.cluster_map_dir), self.tokenizer_name, self.code_vocab_size)
            if n != int(d.cluster_bits_n_clusters):
                raise ValueError(f"cluster map has n_clusters={n}, config says {d.cluster_bits_n_clusters}")
        self.token_to_bits_table = token_bits_table(
            self.code_vocab_size, self.data_bits_per_token, self.cluster_bits, token_to_cluster
        )
        print(f"[owt] split={split} rows={self.num_sequences} bits/token={self.bits_per_token} "
              f"(code={self.data_bits_per_token}+cluster={self.cluster_bits})")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        toks = torch.from_numpy(np.array(self.code_mm[idx], dtype=np.uint16, copy=True)).long()
        return self.token_to_bits_table[toks].view(-1)


def get_dataloaders(config: config_dict.ConfigDict, *, batch_size: Optional[int] = None, seed: int = 42):
    # Build the caches once (rank 0) before every rank opens them.
    ensure_openwebtext_caches_ready(config, split="train")
    ensure_openwebtext_caches_ready(config, split="val")
    batch = int(batch_size or config.train.batch_size)
    num_workers = int(getattr(config.data, "num_workers", 8))
    g = torch.Generator()
    g.manual_seed(int(seed))
    rank, world_size = _ddp_rank_world()

    def _worker_init_fn(worker_id: int) -> None:
        base = int(seed) + int(worker_id)
        np.random.seed(base % (2**32 - 1))
        torch.manual_seed(base)

    def make_loader(ds, *, shuffle, drop_last):
        sampler = None
        if _ddp_is_on():
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle,
                                         drop_last=drop_last, seed=int(seed))
        return DataLoader(
            ds, batch_size=batch, shuffle=shuffle if sampler is None else False, sampler=sampler,
            drop_last=drop_last, num_workers=num_workers, pin_memory=bool(getattr(config.data, "pin_memory", True)),
            persistent_workers=num_workers > 0,
            prefetch_factor=int(getattr(config.data, "prefetch_factor", 4)) if num_workers > 0 else None,
            generator=g if (shuffle and sampler is None) else None,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )

    return (
        make_loader(OpenWebTextDataset(config, split="train"), shuffle=True, drop_last=True),
        make_loader(OpenWebTextDataset(config, split="val"), shuffle=False, drop_last=False),
        make_loader(OpenWebTextDataset(config, split="test"), shuffle=False, drop_last=False),
    )
