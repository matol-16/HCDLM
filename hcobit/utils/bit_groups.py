"""Token / cluster blocks of an H-CoBit bit patch.

Each token is one patch of ``bits_per_token`` bits: the token's own bits, then
ceil(log2 n) cluster bits (Section 4.2 / Appendix D):

    [  token bits  |  cluster bits  ]
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch


def cluster_info_bits(cfg) -> Optional[Tuple[int, int]]:
    """(token bits, cluster bits) per token, or None without the cluster modality."""
    data = cfg.data
    if not bool(getattr(data, "cluster_bits_enabled", False)):
        return None
    cluster_bits = int(math.ceil(math.log2(max(2, int(data.cluster_bits_n_clusters)))))
    info_bits = int(data.bits_per_token) - cluster_bits
    if info_bits <= 0:
        raise ValueError(f"bits_per_token={data.bits_per_token} leaves no token bits")
    return info_bits, cluster_bits


def bit_group_masks(cfg, device) -> Optional[Dict[str, torch.Tensor]]:
    """{"info": [S], "cluster": [S]} 0/1 float masks over the flattened bitstream."""
    ib = cluster_info_bits(cfg)
    if ib is None:
        return None
    info_bits, cluster_bits = ib
    info_tok = torch.zeros(info_bits + cluster_bits, device=device)
    info_tok[:info_bits] = 1.0
    seq_tokens = int(cfg.data.sequence_len_tokens)
    return {"info": info_tok.repeat(seq_tokens), "cluster": (1.0 - info_tok).repeat(seq_tokens)}
