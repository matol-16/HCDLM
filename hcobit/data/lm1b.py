"""LM1B as bitstreams: 128 BERT-base-uncased tokens per row, each token written as its id
in 15 bits (MSB first), followed for H-CoBit by its cluster id in ceil(log2 n) bits.

Rows come from the packed token caches written by scripts/prepare_data.py lm1b. The last
``val_fraction`` of the train rows are held out for validation; the test split is LM1B's.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from ml_collections import config_dict
from torch.utils.data import DataLoader, Dataset


def _ceil_log2(x: int) -> int:
    return int(math.ceil(math.log2(max(2, int(x)))))


def cluster_map_filename(tokenizer_name: str) -> str:
    safe = str(tokenizer_name).replace("/", "_").replace("-", "_").replace(":", "_")
    return f"cluster_mapping_{safe}.json"


def load_cluster_map(map_dir: Path, tokenizer_name: str) -> Tuple[Dict[int, int], int]:
    """(token id -> cluster id, n_clusters) from cluster_maps/<dataset>/k_<n>/."""
    path = Path(map_dir) / cluster_map_filename(tokenizer_name)
    data = json.loads(path.read_text())
    return {int(k): int(v) for k, v in data["token_to_cluster"].items()}, int(data["n_clusters"])


def token_bits_table(vocab_size: int, info_bits: int, cluster_bits: int = 0,
                     token_to_cluster: Optional[Dict[int, int]] = None) -> torch.Tensor:
    """[vocab_size, info_bits + cluster_bits] 0/1 table: token id, then cluster id, MSB first."""
    ids = torch.arange(vocab_size, dtype=torch.long)
    shifts = torch.arange(info_bits - 1, -1, -1, dtype=torch.long)
    table = (ids.unsqueeze(1) >> shifts) & 1
    if cluster_bits > 0:
        cid = torch.tensor([int(token_to_cluster.get(t, 0)) for t in range(vocab_size)], dtype=torch.long)
        cshifts = torch.arange(cluster_bits - 1, -1, -1, dtype=torch.long)
        table = torch.cat([table, (cid.unsqueeze(1) >> cshifts) & 1], dim=1)
    return table


def _load_memmap(cache_path: Path, meta_path: Path) -> tuple[np.memmap, dict]:
    if not cache_path.exists() or not meta_path.exists():
        raise RuntimeError(f"Missing LM1B cache {cache_path}: run `python scripts/prepare_data.py lm1b`.")
    meta = json.loads(meta_path.read_text())
    if meta.get("cache_format") != "packed_token_blocks":
        raise RuntimeError(f"{cache_path} is not a packed-block cache; rebuild it with scripts/prepare_data.py")
    mm = np.memmap(cache_path, dtype=np.uint16, mode="r", shape=(int(meta["n_sequences"]), int(meta["seq_len_tokens"])))
    return mm, meta


class LM1BDataset(Dataset):
    is_text_dataset = True

    def __init__(self, config: config_dict.ConfigDict, *, split: str):
        super().__init__()
        assert split in {"train", "val", "test"}
        from transformers import AutoTokenizer

        d = config.data
        self.split = split
        self.root = Path(d.root)
        self.tokenizer_name = str(d.tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        self.vocab_size_base = int(self.tokenizer.vocab_size)
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.unk_token_id = None if self.tokenizer.unk_token_id is None else int(self.tokenizer.unk_token_id)

        self.cluster_bits_enabled = bool(getattr(d, "cluster_bits_enabled", False))
        self.cluster_bits = _ceil_log2(int(d.cluster_bits_n_clusters)) if self.cluster_bits_enabled else 0
        self.token_to_cluster: Dict[int, int] = {}
        if self.cluster_bits_enabled:
            self.token_to_cluster, n = load_cluster_map(Path(d.cluster_map_dir), self.tokenizer_name)
            if n != int(d.cluster_bits_n_clusters):
                raise ValueError(f"cluster map has n_clusters={n}, config says {d.cluster_bits_n_clusters}")

        self.bits_per_token = int(d.bits_per_token)
        self.data_bits_per_token = self.bits_per_token - self.cluster_bits
        if self.data_bits_per_token < _ceil_log2(self.vocab_size_base):
            raise ValueError(f"{self.data_bits_per_token} token bits cannot encode vocab_size={self.vocab_size_base}")
        self.seq_len_tokens = int(d.sequence_len_tokens)
        self.seq_len_bits = self.seq_len_tokens * self.bits_per_token

        train_mm, _ = _load_memmap(self.root / "cache_train_tokens.uint16", self.root / "cache_train_tokens.meta.json")
        test_mm, _ = _load_memmap(self.root / "cache_test_tokens.uint16", self.root / "cache_test_tokens.meta.json")
        n_total = int(train_mm.shape[0])
        n_val = max(1, int(round(n_total * float(getattr(d, "val_fraction", 0.005)))))
        if split == "train":
            self.mm, self.start, self.end = train_mm, 0, n_total - n_val
        elif split == "val":
            self.mm, self.start, self.end = train_mm, n_total - n_val, n_total
        else:
            self.mm, self.start, self.end = test_mm, 0, int(test_mm.shape[0])
        self.num_sequences = self.end - self.start

        self.token_to_bits_table = token_bits_table(
            self.vocab_size_base, self.data_bits_per_token, self.cluster_bits, self.token_to_cluster
        )
        print(f"[lm1b] split={split} rows={self.num_sequences} bits/token={self.bits_per_token} "
              f"(token={self.data_bits_per_token}+cluster={self.cluster_bits})")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        toks = torch.from_numpy(np.array(self.mm[self.start + idx], dtype=np.int64, copy=True))
        return self.token_to_bits_table[toks].view(-1)


def get_dataloaders(config: config_dict.ConfigDict, *, batch_size: Optional[int] = None, seed: int = 42):
    batch = int(batch_size or config.train.batch_size)
    num_workers = int(getattr(config.data, "num_workers", 8))
    g = torch.Generator()
    g.manual_seed(int(seed))

    def _worker_init_fn(worker_id: int) -> None:
        base = int(seed) + int(worker_id)
        np.random.seed(base % (2**32 - 1))
        torch.manual_seed(base)

    def make_loader(ds, *, shuffle, drop_last):
        return DataLoader(
            ds, batch_size=batch, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,
            pin_memory=bool(getattr(config.data, "pin_memory", True)), persistent_workers=num_workers > 0,
            prefetch_factor=int(getattr(config.data, "prefetch_factor", 4)) if num_workers > 0 else None,
            generator=g if shuffle else None, worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )

    return (
        make_loader(LM1BDataset(config, split="train"), shuffle=True, drop_last=True),
        make_loader(LM1BDataset(config, split="val"), shuffle=False, drop_last=False),
        make_loader(LM1BDataset(config, split="test"), shuffle=False, drop_last=False),
    )
