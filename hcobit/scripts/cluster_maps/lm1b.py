"""Build the LM1B (bert-base-uncased) token -> cluster map.

The LangFlow-LM1B input embeddings are unit-normalized and clustered with k-means++.
Writes cluster_mapping_bert_base_uncased.json (+ _eval.json, the clustering-quality report).

    python scripts/cluster_maps/lm1b.py --map_dir cluster_maps/lm1b/k_16 --n_clusters 16 \\
        --pretrained_model Continuous-Rivals-Discrete/langflow-lm1b --trust_remote_code \\
        --freq_max_tokens 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.base import DATA_ROOT  # noqa: E402
from data.lm1b import cluster_map_filename  # noqa: E402
from scripts.cluster_maps.common import (  # noqa: E402
    _ceil_log2,
    _find_cache,
    _read_cache_token_ids,
    cluster_frequency_report,
    evaluate_clustering,
)


def _reserved_token_ids(tok, vocab_size: int) -> List[int]:
    reserved: List[int] = []
    for tid in [tok.pad_token_id, tok.unk_token_id, tok.cls_token_id, tok.sep_token_id, tok.mask_token_id]:
        if tid is None:
            continue
        tid = int(tid)
        if 0 <= tid < vocab_size and tid not in reserved:
            reserved.append(tid)
    return reserved


def build_embeddings_pretrained(args, vocab_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use the input token-embedding matrix of a pretrained model (e.g. LangFlow) as
    the token embeddings. The model must share the BERT token id space; extra rows
    are ignored by slicing the first ``vocab_size`` rows.

    Loading is delegated to scripts/pretrained_embeddings.py, which handles custom
    masked / diffusion-LM backbones (LangFlow, MDLM) whose embeddings are not always
    exposed via get_input_embeddings(), as well as raw / HF Hub checkpoint files.
    """
    from scripts.cluster_maps.pretrained_embeddings import load_pretrained_token_embeddings

    print(
        f"[cluster_lm1b] Loading pretrained embeddings "
        f"(model={args.pretrained_model}, checkpoint={args.checkpoint_path}, "
        f"from_huggingface={args.from_huggingface}) ..."
    )
    emb_t = load_pretrained_token_embeddings(
        model_name=args.pretrained_model,
        checkpoint_path=args.checkpoint_path,
        from_huggingface=bool(args.from_huggingface),
        hf_repo_id=args.hf_repo_id,
        embeddings_key=args.embeddings_key,
        cache_dir=args.cache_dir,
        trust_remote_code=bool(args.trust_remote_code),
        expected_vocab_size=int(vocab_size),
    )
    emb = emb_t.numpy()
    print(f"[cluster_lm1b] Pretrained embedding matrix: shape={emb.shape}")
    if emb.shape[0] < vocab_size:
        raise SystemExit(
            f"Pretrained embedding rows ({emb.shape[0]}) < tokenizer vocab_size ({vocab_size}). "
            f"The pretrained model must use the same token id space as --tokenizer_name."
        )

    vectors = np.ascontiguousarray(emb[:vocab_size]).astype(np.float32)
    found_mask = np.ones(vocab_size, dtype=bool)
    return vectors, found_mask


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default=f"{DATA_ROOT}/lm1b")
    ap.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--map_dir", type=str, default=None,
                    help="Directory to write the cluster map (default: --root). "
                         "Overridden by --out if both are given.")

    ap.add_argument("--n_clusters", type=int, default=16)

    # Where the token embeddings come from.
    ap.add_argument("--embedding_source", type=str, default="pretrained", choices=["pretrained"])
    ap.add_argument("--pretrained_model", type=str, default=None,
                    help="HF model id / local path whose input embeddings are used "
                         "(Continuous-Rivals-Discrete/langflow-lm1b).")
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="Pass trust_remote_code=True when loading custom backbones (LangFlow).")
    ap.add_argument("--from_huggingface", action="store_true",
                    help="Resolve --checkpoint_path as a file inside --hf_repo_id via the HF Hub.")
    ap.add_argument("--hf_repo_id", type=str, default=None,
                    help="HF Hub repo id holding the checkpoint file (with --from_huggingface).")
    ap.add_argument("--checkpoint_path", type=str, default=None,
                    help="Local checkpoint file, or hub filename when --from_huggingface is set.")
    ap.add_argument("--embeddings_key", type=str, default=None,
                    help="Explicit state-dict key for the embedding tensor (optional).")
    ap.add_argument("--cache_dir", type=str, default=None,
                    help="HF cache dir for downloaded checkpoints (optional).")

    ap.add_argument("--kmeans_n_init", type=int, default=10)
    ap.add_argument("--kmeans_max_iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)

    # Token-frequency-weighted cluster-usage report (scans a few dataset examples).
    ap.add_argument("--freq_max_tokens", type=int, default=1_000_000,
                    help="Scan up to this many tokens from the LM1B token cache for "
                         "the cluster-frequency report (0 disables it).")
    ap.add_argument("--freq_top_k", type=int, default=20,
                    help="Number of most-frequent clusters to print (0 = all).")
    ap.add_argument("--freq_bottom_k", type=int, default=10,
                    help="Number of least-frequent clusters to print.")
    ap.add_argument("--freq_examples_per_cluster", type=int, default=10,
                    help="Example tokens shown per cluster in the frequency report.")

    ap.add_argument("--force", action="store_true")

    args = ap.parse_args()

    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise SystemExit("Missing `scikit-learn`. Install: pip install scikit-learn") from e

    if args.embedding_source == "pretrained" and not (args.pretrained_model or args.checkpoint_path):
        raise SystemExit(
            "--embedding_source pretrained requires --pretrained_model <hf_id> "
            "and/or --checkpoint_path (with --from_huggingface + --hf_repo_id for hub files)."
        )

    n_clusters = int(args.n_clusters)
    cluster_bits = _ceil_log2(n_clusters)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    vocab_size = int(tokenizer.vocab_size)
    print(f"[cluster_lm1b] tokenizer={args.tokenizer_name} vocab_size={vocab_size}")
    print(f"[cluster_lm1b] embedding_source={args.embedding_source} "
          f"n_clusters={n_clusters} -> {cluster_bits} cluster bits/token")

    out_name = cluster_map_filename(args.tokenizer_name)
    map_dir = Path(args.map_dir) if args.map_dir is not None else root
    out_path = Path(args.out) if args.out is not None else (map_dir / out_name)
    if out_path.exists() and not args.force:
        print(f"[cluster_lm1b] ✅ Cluster map found at {out_path}")
        print("[cluster_lm1b] Skipping generation. Use --force to overwrite.")
        return

    def id_to_text(i: int) -> str:
        tok = tokenizer.convert_ids_to_tokens(int(i))
        return tok if tok is not None else str(int(i))

    eval_data: Dict[str, Any] = {}

    # ---- 1. token embeddings ----------------------------------------------
    vectors, found_mask = build_embeddings_pretrained(args, vocab_size)

    # Normalize to the unit sphere so Euclidean k-means approximates cosine
    # clustering (matches the cosine geometry used by the semantic ordering).
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    # ---- 2. k-means++ -------------------------------------------------------
    print(f"[cluster_lm1b] Running k-means++ (k={n_clusters}, n_init={args.kmeans_n_init}) "
          f"on {vocab_size} tokens...")
    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=int(args.kmeans_n_init),
        max_iter=int(args.kmeans_max_iter),
        random_state=int(args.seed),
    )
    labels = km.fit_predict(vectors).astype(np.int64)

    eval_data["clustering_quality"] = evaluate_clustering(
        vectors,
        labels,
        n_clusters=n_clusters,
        inertia=float(km.inertia_),
        found_mask=found_mask,
        id_to_text=id_to_text,
        seed=int(args.seed),
    )

    # ---- 2b. token-frequency-weighted cluster usage -------------------------
    reserved_ids = _reserved_token_ids(tokenizer, vocab_size)
    scanned_ids: np.ndarray = np.array([], dtype=np.int64)
    if int(args.freq_max_tokens) > 0:
        cache_path = _find_cache(root, ["cache_train_tokens.uint16", "cache_test_tokens.uint16"])
        if cache_path is None:
            print(f"[cluster_lm1b] (no LM1B token cache under {root}; "
                  f"skipping cluster-frequency report)")
        else:
            scanned_ids, n_ex = _read_cache_token_ids(
                cache_path, layout="packed", max_tokens=int(args.freq_max_tokens)
            )
            print(f"[cluster_lm1b] cluster-frequency source: {cache_path.name}")
            freq = cluster_frequency_report(
                scanned_ids, labels, n_clusters=n_clusters,
                exclude_ids=reserved_ids,
                id_to_text=id_to_text, examples_scanned=n_ex, tag="cluster_lm1b",
                top_k=int(args.freq_top_k),
                bottom_k=int(args.freq_bottom_k),
                examples_per_cluster=int(args.freq_examples_per_cluster),
            )
            if freq is not None:
                eval_data["frequency_report"] = freq

    # ---- 3. save ------------------------------------------------------------
    token_to_cluster: Dict[str, int] = {str(tid): int(labels[tid]) for tid in range(vocab_size)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenizer_name": str(args.tokenizer_name),
                "codec": "base",
                "embedding_source": str(args.embedding_source),
                "pretrained_model": args.pretrained_model,
                "vocab_size": int(vocab_size),
                "n_clusters": int(n_clusters),
                "cluster_bits": int(cluster_bits),
                "token_to_cluster": token_to_cluster,
            },
            f,
        )

    eval_path = out_path.with_name(out_path.stem + "_eval.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)

    print(f"[cluster_lm1b] ✅ Saved cluster map (k={n_clusters}) to: {out_path}")
    print(f"[cluster_lm1b] ✅ Saved eval report to: {eval_path}")
    print("[cluster_lm1b] Cluster bits are appended after the token info bits in the dataset.")


if __name__ == "__main__":
    main()
