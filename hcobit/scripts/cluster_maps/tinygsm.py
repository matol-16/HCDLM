"""Build the TinyGSM (SmolLM-135M) token -> cluster map.

TinyGSM tokens are raw SmolLM-135M ids, so the input embeddings of the S-FLM TinyGSM model
(sphere architecture) are clustered directly: unit-normalized, k-means++. The added [PAD]
token (id 49152, no native embedding, most of every packed answer) is pinned to a reserved
cluster, and the cluster codes are permuted to maximise the per-bit entropy of the cluster
bits over the training cache. Writes cluster_mapping_HuggingFaceTB_SmolLM_135M.json
(+ _eval.json).

    python scripts/cluster_maps/extract_sflm_embeddings.py --ckpt <sfm.ckpt> --out sfm_embeddings.pt
    python scripts/cluster_maps/tinygsm.py --map_dir cluster_maps/tinygsm/k_16 --n_clusters 16 \\
        --checkpoint_path sfm_embeddings.pt \\
        --reserve_special_cluster --freq_rebalance --freq_rebalance_include_specials \\
        --freq_max_tokens 8000000
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.base import DATA_ROOT  # noqa: E402
from scripts.cluster_maps.common import (  # noqa: E402
    _ceil_log2,
    _safe_name,
    cluster_frequency_report,
    evaluate_clustering,
    rebalance_cluster_codes,
)
from scripts.cluster_maps.pretrained_embeddings import load_pretrained_token_embeddings  # noqa: E402


def _get_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        # Mirror data/tinygsm.get_tokenizer: a dedicated PAD token is appended.
        tok.add_special_tokens({"pad_token": "[PAD]"})
    return tok


def _read_tinygsm_cache_ids(root: Path, max_tokens: int) -> Tuple[np.ndarray, int]:
    """Read up to ``max_tokens`` token ids (flat) from the TinyGSM uint16 cache.

    The cache is a 2-D [N, block] uint16 file ``<tag>_<split>_ids.uint16`` with an
    ``<tag>.meta.json`` giving ``meta[split]['n']``; block size is in the tag (``_bs512_``).
    Returns (ids_flat, n_rows_scanned). Empty array if no cache exists.
    """
    root = Path(root)
    for split in ("train", "validation"):
        cands = sorted(root.glob(f"*_{split}_ids.uint16"))
        if not cands:
            continue
        ids_path = cands[0]
        tag = ids_path.name[: -len(f"_{split}_ids.uint16")]
        meta_path = root / f"{tag}.meta.json"
        block = 512
        m = re.search(r"_bs(\d+)_", tag)
        if m:
            block = int(m.group(1))
        n = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                n = int(meta.get(split, {}).get("n", 0)) or None
            except Exception:
                n = None
        if n is None:
            n = int(ids_path.stat().st_size // (2 * block))
        if n <= 0:
            continue
        mm = np.memmap(ids_path, dtype=np.uint16, mode="r", shape=(n, block))
        n_rows = min(n, max(1, int(max_tokens) // block))
        return np.asarray(mm[:n_rows], dtype=np.int64).reshape(-1), int(n_rows)
    return np.array([], dtype=np.int64), 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=f"{DATA_ROOT}/tinygsm")
    ap.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    ap.add_argument("--token_vocab_size", type=int, default=49153,
                    help="Total clustered vocabulary incl. the added [PAD] "
                         "(SmolLM-135M |V|=49152 + PAD = 49153).")
    ap.add_argument("--n_clusters", type=int, default=16)
    ap.add_argument("--map_dir", type=str, default=None,
                    help="Directory to write the cluster map (default: --root).")
    ap.add_argument("--out", type=str, default=None)

    # Embedding source (pretrained input embeddings, via scripts/pretrained_embeddings.py).
    ap.add_argument("--embedding_source", type=str, default="pretrained", choices=["pretrained"])
    ap.add_argument("--pretrained_model", type=str, default=None,
                    help="HF id / local path whose input embeddings are used.")
    ap.add_argument("--checkpoint_path", type=str, default=None,
                    help="Bare embedding tensor from extract_sflm_embeddings.py, or a hub "
                         "filename with --from_huggingface + --hf_repo_id.")
    ap.add_argument("--embeddings_key", type=str, default=None,
                    help="Explicit state-dict key for the embedding tensor (optional).")
    ap.add_argument("--from_huggingface", action="store_true")
    ap.add_argument("--hf_repo_id", type=str, default=None)
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--reserve_special_cluster", action="store_true",
                    help="Hold every token WITHOUT a native embedding (the added [PAD]) out of "
                         "the k-means fit and pin them to a dedicated cluster id (n_clusters-1). "
                         "Keeps cluster_bits = ceil(log2 n_clusters). Strongly recommended.")
    ap.add_argument("--freq_rebalance", action="store_true",
                    help="Permute cluster codes to maximise per-bit entropy over real TinyGSM "
                         "token usage (2-opt). Falls back to type-occupancy if no cache exists.")
    ap.add_argument("--freq_rebalance_include_specials", action="store_true",
                    help="Count PAD when rebalancing (the model has no padding loss mask, so it "
                         "trains on / generates those slots). Recommended for TinyGSM.")
    ap.add_argument("--freq_max_tokens", type=int, default=20_000_000,
                    help="Scan up to this many tokens from the TinyGSM cache for the frequency "
                         "report / rebalance (0 disables).")
    ap.add_argument("--freq_top_k", type=int, default=20)
    ap.add_argument("--freq_bottom_k", type=int, default=10)
    ap.add_argument("--freq_examples_per_cluster", type=int, default=10)

    ap.add_argument("--kmeans_n_init", type=int, default=10)
    ap.add_argument("--kmeans_max_iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise SystemExit("Missing scikit-learn. Install: pip install scikit-learn") from e

    if not (args.pretrained_model or args.checkpoint_path):
        raise SystemExit(
            "Provide an embedding source: --checkpoint_path (the S-FLM embedding tensor from "
            "extract_sflm_embeddings.py) and/or --pretrained_model <hf_id>."
        )

    n_clusters = int(args.n_clusters)
    cluster_bits = _ceil_log2(n_clusters)
    vocab = int(args.token_vocab_size)

    root = Path(args.root)
    map_dir = Path(args.map_dir) if args.map_dir is not None else root
    out_name = f"cluster_mapping_{_safe_name(args.tokenizer_name)}.json"
    out_path = Path(args.out) if args.out is not None else (map_dir / out_name)
    if out_path.exists() and not args.force:
        print(f"[cluster_tinygsm] ✅ Cluster map found at {out_path}; skipping (use --force).")
        return

    tokenizer = _get_tokenizer(args.tokenizer_name)
    pad_id = int(tokenizer.pad_token_id)
    print(f"[cluster_tinygsm] tokenizer={args.tokenizer_name} |V|(+PAD)={len(tokenizer)} "
          f"pad_id={pad_id} clustered_vocab={vocab}")
    print(f"[cluster_tinygsm] n_clusters={n_clusters} -> {cluster_bits} cluster bits/token")

    def id_to_text(i: int) -> str:
        try:
            return tokenizer.decode([int(i)])
        except Exception:
            return str(int(i))

    # ---- 1. native token embeddings -----------------------------------------
    emb_t = load_pretrained_token_embeddings(
        model_name=args.pretrained_model,
        checkpoint_path=args.checkpoint_path,
        from_huggingface=bool(args.from_huggingface),
        hf_repo_id=args.hf_repo_id,
        embeddings_key=args.embeddings_key,
        cache_dir=args.cache_dir,
        trust_remote_code=bool(args.trust_remote_code),
        expected_vocab_size=None,
    )
    emb = emb_t.numpy().astype(np.float32)
    rows, dim = emb.shape
    print(f"[cluster_tinygsm] embedding matrix: shape=({rows}, {dim}) "
          f"source={args.pretrained_model or args.checkpoint_path}")

    rng = np.random.default_rng(int(args.seed))
    vectors = rng.standard_normal((vocab, dim)).astype(np.float32)
    found_mask = np.zeros(vocab, dtype=bool)
    n_copy = min(rows, vocab)
    vectors[:n_copy] = emb[:n_copy]
    found_mask[:n_copy] = True
    # [PAD] is an appended token: force it OUT of the semantic fit regardless of whether the
    # source model happens to have a row there. Any id beyond the embedding rows is already
    # not-found (keeps its random vector).
    if 0 <= pad_id < vocab:
        found_mask[pad_id] = False
    print(f"[cluster_tinygsm] tokens with a native vector: {int(found_mask.sum())}/{vocab} "
          f"(reserved specials: {[int(s) for s in np.where(~found_mask)[0]]})")

    # Unit-normalize so Euclidean k-means approximates cosine clustering (LM1B/OWT recipe).
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    # ---- 2. k-means++ (optionally reserving a special cluster) ---------------
    reserved_id: Optional[int] = None
    if args.reserve_special_cluster:
        n_semantic = n_clusters - 1
        if n_semantic < 2:
            raise SystemExit(f"--reserve_special_cluster needs n_clusters >= 3 (got {n_clusters})")
        reserved_id = n_semantic
        specials = np.where(~found_mask)[0]
        print(f"[cluster_tinygsm] Reserving cluster id {reserved_id} for {len(specials)} special "
              f"token(s); fitting k-means++ on {int(found_mask.sum())} tokens with k={n_semantic}...")
        km = KMeans(n_clusters=n_semantic, init="k-means++", n_init=int(args.kmeans_n_init),
                    max_iter=int(args.kmeans_max_iter), random_state=int(args.seed))
        sub_labels = km.fit_predict(vectors[found_mask]).astype(np.int64)
        labels = np.full(vocab, reserved_id, dtype=np.int64)
        labels[found_mask] = sub_labels
        quality_vectors, quality_labels = vectors[found_mask], sub_labels
        quality_k, quality_found = n_semantic, np.ones(int(found_mask.sum()), dtype=bool)
        quality_ids = np.where(found_mask)[0]

        def quality_id_to_text(i: int, _ids=quality_ids) -> str:
            return id_to_text(int(_ids[int(i)]))
    else:
        print(f"[cluster_tinygsm] Running k-means++ (k={n_clusters}) on {vocab} tokens...")
        km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=int(args.kmeans_n_init),
                    max_iter=int(args.kmeans_max_iter), random_state=int(args.seed))
        labels = km.fit_predict(vectors).astype(np.int64)
        quality_vectors, quality_labels = vectors, labels
        quality_k, quality_found = n_clusters, found_mask
        quality_id_to_text = id_to_text

    eval_data = {}
    eval_data["clustering_quality"] = evaluate_clustering(
        quality_vectors, quality_labels, n_clusters=quality_k, inertia=float(km.inertia_),
        found_mask=quality_found, id_to_text=quality_id_to_text, seed=int(args.seed),
    )
    if reserved_id is not None:
        eval_data["special_cluster"] = {
            "reserved_id": int(reserved_id),
            "n_semantic_clusters": int(n_clusters - 1),
            "special_token_ids": [int(s) for s in np.where(~found_mask)[0]],
            "note": "specials (no native embedding, incl. [PAD]) excluded from the k-means fit; "
                    "clustering_quality covers the semantic clusters only",
        }

    # ---- 3. token-frequency report over the TinyGSM cache --------------------
    scanned_ids, n_ex = (np.array([], dtype=np.int64), 0)
    if int(args.freq_max_tokens) > 0:
        scanned_ids, n_ex = _read_tinygsm_cache_ids(root, int(args.freq_max_tokens))
        if scanned_ids.size == 0:
            print(f"[cluster_tinygsm] (no TinyGSM token cache under {root}; frequency report/"
                  f"rebalance will fall back to type-occupancy)")
        else:
            print(f"[cluster_tinygsm] frequency source: TinyGSM cache ({n_ex} sequences, "
                  f"{scanned_ids.size} tokens)")
            freq = cluster_frequency_report(
                scanned_ids, labels, n_clusters=n_clusters, exclude_ids=[pad_id],
                id_to_text=id_to_text, examples_scanned=n_ex, tag="cluster_tinygsm",
                top_k=int(args.freq_top_k), bottom_k=int(args.freq_bottom_k),
                examples_per_cluster=int(args.freq_examples_per_cluster),
            )
            if freq is not None:
                eval_data["frequency_report"] = freq

    # ---- 3b. optional code-assignment rebalancing ---------------------------
    if args.freq_rebalance:
        def _bit_entropy(pb: float) -> float:
            return (-pb * math.log2(pb) - (1 - pb) * math.log2(1 - pb)) if 0.0 < pb < 1.0 else 0.0

        if scanned_ids.size > 0:
            if args.freq_rebalance_include_specials:
                valid_ids = scanned_ids
                freq_source = "token-frequency (incl. PAD; realized bitstream)"
            else:
                valid_ids = scanned_ids[scanned_ids != pad_id]
                freq_source = "token-frequency (excl. PAD)"
            cluster_freq_arr = np.bincount(labels[valid_ids], minlength=n_clusters).astype(np.float64)
        else:
            cluster_freq_arr = np.bincount(labels, minlength=n_clusters).astype(np.float64)
            freq_source = "type-occupancy (no token cache)"

        freq_norm = cluster_freq_arr / cluster_freq_arr.sum() if cluster_freq_arr.sum() > 0 else cluster_freq_arr
        p_before = np.array([
            sum(freq_norm[k] for k in range(n_clusters) if (k >> b) & 1)
            for b in range(cluster_bits)
        ])
        H_before = sum(_bit_entropy(float(p)) for p in p_before)

        code, n_swaps, p_after = rebalance_cluster_codes(cluster_freq_arr, cluster_bits)
        H_after = sum(_bit_entropy(float(p)) for p in p_after)

        print("\n" + "=" * 70)
        print(f"[cluster_tinygsm] Code-assignment rebalancing ({freq_source})")
        print(f"  swaps performed        : {n_swaps}")
        print(f"  total bit entropy      : {H_before:.4f} -> {H_after:.4f} bits "
              f"(max {float(cluster_bits):.1f})")
        for b in range(cluster_bits):
            print(f"    bit {b}: {p_before[b]:.3f} -> {p_after[b]:.3f}")
        print("=" * 70 + "\n")

        labels = code[labels]
        if reserved_id is not None:
            eval_data["special_cluster"]["reserved_id_before_rebalance"] = int(reserved_id)
            eval_data["special_cluster"]["reserved_id"] = int(code[reserved_id])
        eval_data["rebalancing"] = {
            "freq_source": freq_source,
            "n_swaps": n_swaps,
            "entropy_before": round(float(H_before), 4),
            "entropy_after": round(float(H_after), 4),
            "entropy_max": float(cluster_bits),
            "p_per_bit_before": [round(float(p), 4) for p in p_before],
            "p_per_bit_after": [round(float(p), 4) for p in p_after],
        }

    # ---- 4. save ------------------------------------------------------------
    token_to_cluster = {str(tid): int(labels[tid]) for tid in range(vocab)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenizer_name": str(args.tokenizer_name),
                "codec": "base",
                "dataset": "tinygsm",
                "embedding_source": str(args.embedding_source),
                "pretrained_model": args.pretrained_model,
                "checkpoint_path": args.checkpoint_path,
                "vocab_size": int(vocab),
                "n_clusters": int(n_clusters),
                "cluster_bits": int(cluster_bits),
                "token_to_cluster": token_to_cluster,
            },
            f,
        )
    eval_path = out_path.with_name(out_path.stem + "_eval.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)

    print(f"[cluster_tinygsm] ✅ Saved cluster map (k={n_clusters}) to: {out_path}")
    print(f"[cluster_tinygsm] ✅ Saved eval report to: {eval_path}")
    print("[cluster_tinygsm] Cluster bits are appended after the token info bits in the dataset.")


if __name__ == "__main__":
    main()
