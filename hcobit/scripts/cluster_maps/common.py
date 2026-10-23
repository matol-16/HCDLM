"""Clustering helpers shared by the three cluster-map builders (lm1b.py, owt.py, tinygsm.py):
span-averaged code-token vectors (OpenWebText), the clustering-quality report, the
code-assignment rebalancing and the token-frequency report.

Copied verbatim from the code the paper's maps were built with.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def _safe_name(s: str) -> str:
    return str(s).replace("/", "_").replace("-", "_").replace(":", "_")


def _ceil_log2(x: int) -> int:
    return int(math.ceil(math.log2(max(2, int(x)))))


def code_vectors_from_base(
    base_vecs: np.ndarray,
    base_found: np.ndarray,
    *,
    code_tokenizer,
    code_meta: dict,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive an embedding for every code token by averaging the base-token
    embeddings of the GPT-2 ids it expands to.

    Each code token's surface form is a string of Private-Use-Area characters,
    one per GPT-2 id (chr(base_char_offset + gpt2_id)); inverting that gives the
    subword span. Special tokens ([PAD]/[UNK]/[EOSEQ]) have no span and keep a
    random vector (excluded from the quality metrics).
    """
    offset = int(code_meta["base_char_offset"])
    vocab_size_gpt2 = int(code_meta["vocab_size_gpt2"])
    code_vocab_size = int(code_meta["actual_code_vocab_size"])
    dim = int(base_vecs.shape[1])

    rng = np.random.default_rng(int(seed))
    code_vecs = rng.standard_normal((code_vocab_size, dim)).astype(np.float32)
    code_found = np.zeros(code_vocab_size, dtype=bool)

    spans_len = []
    for cid in range(code_vocab_size):
        surface = code_tokenizer.id_to_token(cid)
        if surface is None:
            continue
        base_ids = [ord(ch) - offset for ch in surface]
        base_ids = [b for b in base_ids if 0 <= b < vocab_size_gpt2]
        if not base_ids:
            continue
        code_vecs[cid] = base_vecs[base_ids].mean(axis=0)
        code_found[cid] = bool(base_found[base_ids].any())
        spans_len.append(len(base_ids))

    if spans_len:
        sl = np.asarray(spans_len)
        print(
            f"[cluster_owt] Code-token spans: {len(spans_len)} mapped tokens, "
            f"span len min/mean/max = {sl.min()}/{sl.mean():.2f}/{sl.max()}"
        )
    print(f"[cluster_owt] Code vectors ready. Found {int(code_found.sum())}/{code_vocab_size} via base spans.")
    return code_vecs, code_found


# -----------------------------------------------------------------------------
# Clustering quality report
# -----------------------------------------------------------------------------

def evaluate_clustering(
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    n_clusters: int,
    inertia: float,
    found_mask: np.ndarray,
    id_to_text: Callable[[int], str],
    seed: int,
    silhouette_sample: int = 10_000,
) -> Dict[str, Any]:
    """
    Print a compact, human-readable assessment of the clustering quality and
    return the same data as a dict (written to the eval JSON by callers).

    Metrics are restricted to tokens that actually received an informative
    vector (``found_mask``); randomly-initialised vectors for unseen tokens would
    otherwise inflate the scores with meaningless points.
    """
    print("\n" + "=" * 70)
    print("[cluster_owt] Clustering quality report")
    print("=" * 70)

    V = int(vectors.shape[0])
    n_found = int(found_mask.sum())
    print(f"  tokens (vocab)         : {V}")
    print(f"  tokens with a vector   : {n_found} ({100.0 * n_found / max(V, 1):.1f}%)")
    print(f"  clusters requested     : {n_clusters}")
    print(f"  k-means inertia (SSE)  : {inertia:.2f}")

    # ---- cluster occupancy ----
    counts = np.bincount(labels, minlength=n_clusters)
    nonempty = int((counts > 0).sum())
    print(f"  non-empty clusters     : {nonempty}/{n_clusters}")
    print(
        f"  cluster size  min/mean/max : "
        f"{int(counts.min())} / {counts.mean():.1f} / {int(counts.max())}"
    )
    ideal = V / float(n_clusters)
    balance = 1.0 - float(np.abs(counts - ideal).sum()) / (2.0 * V)
    print(f"  occupancy balance      : {balance:.3f}  (1.0 = perfectly uniform)")

    result: Dict[str, Any] = {
        "vocab_size": V,
        "tokens_with_vector": n_found,
        "n_clusters": n_clusters,
        "inertia": round(float(inertia), 4),
        "non_empty_clusters": nonempty,
        "cluster_size_min": int(counts.min()),
        "cluster_size_mean": round(float(counts.mean()), 2),
        "cluster_size_max": int(counts.max()),
        "occupancy_balance": round(float(balance), 4),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "example_clusters": [],
    }

    # ---- internal validity indices (on the embedded tokens only) ----
    fv = vectors[found_mask]
    fl = labels[found_mask]
    try:
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )

        rng = np.random.default_rng(int(seed))
        if fv.shape[0] > int(silhouette_sample):
            idx = rng.choice(fv.shape[0], size=int(silhouette_sample), replace=False)
            sv, sl = fv[idx], fl[idx]
        else:
            sv, sl = fv, fl

        if len(np.unique(sl)) >= 2:
            sil = silhouette_score(sv, sl, metric="euclidean")
            dbi = davies_bouldin_score(fv, fl)
            chi = calinski_harabasz_score(fv, fl)
            print(f"  silhouette  (higher=better, [-1,1]) : {sil:.4f}  "
                  f"(on {sv.shape[0]} sampled tokens)")
            print(f"  davies-bouldin (lower=better, >=0)  : {dbi:.4f}")
            print(f"  calinski-harabasz (higher=better)   : {chi:.1f}")
            result["silhouette"] = round(float(sil), 4)
            result["davies_bouldin"] = round(float(dbi), 4)
            result["calinski_harabasz"] = round(float(chi), 2)
        else:
            print("  (silhouette/DBI skipped: fewer than 2 populated clusters)")
    except ImportError:
        print("  (install scikit-learn for silhouette / davies-bouldin indices)")

    # ---- type-occupancy entropy ----
    occ_p = counts[counts > 0] / float(counts.sum())
    H_type = float(-(occ_p * np.log2(occ_p)).sum())
    H_type_max = math.log2(max(2, nonempty))
    print(f"  type-occupancy entropy : {H_type:.3f} bits  "
          f"(max {H_type_max:.3f}, normalized {H_type / H_type_max:.3f})")

    # ---- example clusters (all non-empty, sorted by size desc) ----
    print("\n  Example clusters (all non-empty, most populated first, up to 20 tokens each):")
    order = np.argsort(-counts)
    for cid in order:
        if counts[cid] == 0:
            continue
        members = np.where(labels == cid)[0]
        members = members[found_mask[members]]  # prefer tokens with a real vector
        if members.size == 0:
            continue
        members = members[:20]
        toks: List[str] = []
        for t in members:
            try:
                txt = id_to_text(int(t)).strip()
            except Exception:
                txt = str(int(t))
            if txt != "":
                toks.append(txt)
        print(f"    cluster {int(cid):3d} (size {int(counts[cid])}): "
              + ", ".join(repr(tok) for tok in toks))
        result["example_clusters"].append({
            "cluster_id": int(cid),
            "size": int(counts[cid]),
            "example_tokens": toks,
        })
    result["type_occupancy_entropy"] = round(float(H_type), 4)
    result["type_occupancy_entropy_max"] = round(float(H_type_max), 4)
    result["type_occupancy_entropy_normalized"] = round(float(H_type / H_type_max), 4)
    print("=" * 70 + "\n")
    return result


# -----------------------------------------------------------------------------
# Code-assignment rebalancing
#
# k-means assigns cluster ids 0..K-1 arbitrarily. Those ids are encoded in
# plain binary as the cluster bits. By permuting which binary code each cluster
# receives we can make every bit position closer to 50/50, maximising the total
# per-bit entropy  sum_b H(p_b)  where p_b is the probability that bit b is 1
# (frequency-weighted over the real dataset).
#
# We use 2-opt local search: iterate over all pairs (i, j), swap their codes if
# the total entropy increases, repeat until no improving swap exists.
# -----------------------------------------------------------------------------


def rebalance_cluster_codes(
    cluster_freq: np.ndarray,
    cluster_bits: int,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Find a permutation of cluster codes that maximises total per-bit entropy.

    Args:
        cluster_freq: shape (n_clusters,) — frequency / count of each cluster
                      (need not be normalised; zero-count clusters are allowed).
        cluster_bits: number of bits used to encode cluster ids.

    Returns:
        code     : int64 array (n_clusters,) — code[k] is the binary code
                   assigned to cluster k (replaces the default code k).
        n_swaps  : total number of improving swaps performed.
        p_final  : float64 array (cluster_bits,) — final per-bit-1 probabilities.
    """
    n = len(cluster_freq)
    freq = cluster_freq.astype(np.float64)
    total = freq.sum()
    if total > 0:
        freq = freq / total

    # start with the identity permutation (cluster k -> code k)
    code = np.arange(n, dtype=np.int64)

    # p[b] = P(bit b == 1) under current code assignment
    p = np.zeros(cluster_bits, dtype=np.float64)
    for b in range(cluster_bits):
        for k in range(n):
            if (int(code[k]) >> b) & 1:
                p[b] += freq[k]

    def _h(pb: float) -> float:
        if pb <= 0.0 or pb >= 1.0:
            return 0.0
        return -pb * math.log2(pb) - (1.0 - pb) * math.log2(1.0 - pb)

    improved = True
    n_swaps = 0
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = int(code[i]), int(code[j])
                diff = ci ^ cj
                if diff == 0:
                    continue
                # compute change in total bit entropy if codes of i and j are swapped
                delta = 0.0
                for b in range(cluster_bits):
                    if not (diff >> b) & 1:
                        continue
                    pb = p[b]
                    if (ci >> b) & 1:
                        # ci has 1, cj has 0 → after swap ci gets 0, cj gets 1
                        new_pb = pb - freq[i] + freq[j]
                    else:
                        # ci has 0, cj has 1 → after swap ci gets 1, cj gets 0
                        new_pb = pb + freq[i] - freq[j]
                    delta += _h(new_pb) - _h(pb)
                if delta > 1e-12:
                    code[i], code[j] = code[j], code[i]
                    for b in range(cluster_bits):
                        if not (diff >> b) & 1:
                            continue
                        if (ci >> b) & 1:
                            p[b] += freq[j] - freq[i]
                        else:
                            p[b] += freq[i] - freq[j]
                    improved = True
                    n_swaps += 1

    return code, n_swaps, p


# -----------------------------------------------------------------------------
# Token-frequency-weighted cluster usage
#
# evaluate_clustering() above reports *type* occupancy (how many distinct vocab
# tokens fall in each cluster). This complementary report weights clusters by how
# often their tokens actually occur in a few real dataset examples, i.e. the
# cluster-usage *frequency* the diffusion model will see during training.
# -----------------------------------------------------------------------------

def _read_cache_token_ids(path: Path, *, layout: str, max_tokens: int) -> Tuple[np.ndarray, int]:
    """
    Memmap a uint16 token cache and return up to ``max_tokens`` token ids (flat),
    plus the number of example sequences scanned (0 for a flat cache).

    layout='flat'   -> 1-D cache of contiguous ids (meta['n_tokens']).
    layout='packed' -> 2-D cache of fixed-width rows (meta row/len keys vary).
    """
    meta_path = path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    if layout == "flat":
        n = int(meta.get("n_tokens", 0)) or int(Path(path).stat().st_size // 2)
        mm = np.memmap(path, dtype=np.uint16, mode="r", shape=(n,))
        k = min(n, int(max_tokens))
        return np.asarray(mm[:k], dtype=np.int64), 0

    # packed: discover (rows, row_len) from whichever meta keys are present
    if "num_sequences" in meta:
        rows, row_len = int(meta["num_sequences"]), int(meta["code_sequence_len_tokens"])
    elif "n_sequences" in meta:
        rows, row_len = int(meta["n_sequences"]), int(meta["seq_len_tokens"])
    else:
        raise SystemExit(f"Cannot determine row layout for cache: {path}")
    mm = np.memmap(path, dtype=np.uint16, mode="r", shape=(rows, row_len))
    n_rows = min(rows, max(1, int(max_tokens) // max(1, row_len)))
    return np.asarray(mm[:n_rows], dtype=np.int64).reshape(-1), int(n_rows)


def _find_cache(root: Path, patterns) -> Optional[Path]:
    for pat in patterns:
        matches = sorted(Path(root).glob(pat))
        if matches:
            return matches[0]
    return None


def cluster_frequency_report(
    token_ids: np.ndarray,
    labels: np.ndarray,
    *,
    n_clusters: int,
    exclude_ids=(),
    id_to_text: Optional[Callable[[int], str]] = None,
    examples_scanned: int = 0,
    top_k: int = 20,
    bottom_k: int = 10,
    examples_per_cluster: int = 10,
    tag: str = "cluster",
) -> Optional[Dict[str, Any]]:
    """
    Print the token-frequency-weighted cluster usage report and return the same
    data as a dict, or None if no tokens were scanned.

    Shows all clusters sorted by frequency, with up to ``examples_per_cluster``
    representative tokens per entry.  ``top_k`` and ``bottom_k`` control how many
    clusters are printed in the sorted lists (0 = all).
    """
    exclude = set(int(x) for x in exclude_ids)
    V = int(labels.shape[0])

    ids = np.asarray(token_ids).reshape(-1)
    ids = ids[(ids >= 0) & (ids < V)]
    if exclude and ids.size:
        ids = ids[~np.isin(ids, np.asarray(sorted(exclude), dtype=ids.dtype))]

    print("\n" + "=" * 70)
    print(f"[{tag}] Token-frequency-weighted cluster usage (real dataset examples)")
    print("=" * 70)

    if ids.size == 0:
        print("  (no token cache found / no tokens scanned; frequency report skipped)")
        print("=" * 70 + "\n")
        return None

    freq_counts = np.bincount(labels[ids], minlength=int(n_clusters)).astype(np.int64)
    total = int(freq_counts.sum())

    scanned = f"  (~{examples_scanned} example sequences)" if examples_scanned else ""
    print(f"  tokens scanned         : {total}{scanned}")
    nonempty = int((freq_counts > 0).sum())
    print(f"  clusters used (>0)     : {nonempty}/{n_clusters}")

    # ---- frequency-weighted entropy ----
    p = freq_counts / float(total)
    nz = p[p > 0]
    H = float(-(nz * np.log2(nz)).sum())
    Hmax = math.log2(max(2, int(n_clusters)))
    print(f"  frequency entropy      : {H:.3f} bits  (max {Hmax:.3f}, normalized {H / Hmax:.3f})")

    # ---- type-occupancy entropy (unweighted by token frequency) ----
    type_counts = np.bincount(labels, minlength=int(n_clusters)).astype(np.int64)
    occ_p = type_counts[type_counts > 0] / float(type_counts.sum())
    H_type = float(-(occ_p * np.log2(occ_p)).sum())
    H_type_max = math.log2(max(2, int((type_counts > 0).sum())))
    print(f"  type-occupancy entropy : {H_type:.3f} bits  "
          f"(max {H_type_max:.3f}, normalized {H_type / H_type_max:.3f})")

    # precompute per-type frequency once for O(V) lookups in _examples
    type_freq_in_corpus = np.bincount(ids, minlength=V).astype(np.int64) if ids.size else np.zeros(V, dtype=np.int64)

    def _examples(cid: int, n: int) -> List[str]:
        if id_to_text is None:
            return []
        members = np.where(labels == cid)[0]
        members = sorted(members.tolist(), key=lambda t: -int(type_freq_in_corpus[t]))[:n]
        toks: List[str] = []
        for t in members:
            try:
                txt = id_to_text(int(t)).strip()
            except Exception:
                txt = str(int(t))
            if txt:
                toks.append(txt)
        return toks

    order = np.argsort(-freq_counts)
    all_entries: List[Dict[str, Any]] = []
    for cid in order:
        pct = 100.0 * freq_counts[cid] / total
        ex_toks = _examples(int(cid), examples_per_cluster)
        all_entries.append({
            "cluster_id": int(cid),
            "count": int(freq_counts[cid]),
            "pct": round(float(pct), 4),
            "example_tokens": ex_toks,
        })

    def _print_entry(e: Dict[str, Any]) -> None:
        ex = ("  e.g. " + ", ".join(repr(t) for t in e["example_tokens"])) if e["example_tokens"] else ""
        print(f"    cluster {e['cluster_id']:3d}: {e['count']:8d}  ({e['pct']:5.2f}%)" + ex)

    n_top = len(all_entries) if top_k == 0 else min(top_k, len(all_entries))
    n_bot = 0 if bottom_k == 0 else min(bottom_k, max(0, len(all_entries) - n_top))

    print(f"\n  all {len(all_entries)} clusters by token frequency (most → least used):")
    for e in all_entries[:n_top]:
        _print_entry(e)
    if n_bot > 0:
        if n_top < len(all_entries) - n_bot:
            print(f"    ... ({len(all_entries) - n_top - n_bot} clusters omitted) ...")
        print(f"  bottom {n_bot} clusters (least used):")
        for e in all_entries[len(all_entries) - n_bot:]:
            _print_entry(e)
    print("=" * 70 + "\n")

    return {
        "tokens_scanned": total,
        "examples_scanned": examples_scanned,
        "n_clusters": n_clusters,
        "clusters_used": nonempty,
        "entropy_bits": round(H, 4),
        "entropy_max_bits": round(Hmax, 4),
        "entropy_normalized": round(H / Hmax, 4),
        "type_occupancy_entropy": round(H_type, 4),
        "type_occupancy_entropy_max": round(H_type_max, 4),
        "type_occupancy_entropy_normalized": round(H_type / H_type_max, 4),
        "all_clusters": all_entries,
    }

