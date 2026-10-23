"""Build the OpenWebText (gpt2id_bpe16) code-token -> cluster map.

Every code token gets the mean of the LangFlow-OWT input embeddings of the GPT-2 ids it
expands to; the vectors are unit-normalized and clustered with k-means++. The special code
tokens ([PAD]/[UNK]/[EOSEQ]/'\\n', no GPT-2 span) are pinned to a reserved cluster, and the
cluster codes are permuted to maximise the per-bit entropy of the cluster bits over the
training cache. Writes cluster_mapping_gpt2_gpt2id_bpe16_v65536.json (+ _eval.json).

    python scripts/cluster_maps/owt.py --map_dir cluster_maps/owt/k_16 --n_clusters 16 \\
        --pretrained_model Continuous-Rivals-Discrete/langflow-owt --trust_remote_code \\
        --reserve_special_cluster --freq_rebalance --freq_rebalance_include_specials \\
        --freq_max_tokens 50000000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs.base import DATA_ROOT  # noqa: E402
from data.openwebtext import load_gpt2id_bpe16_tokenizer_and_meta  # noqa: E402
from scripts.cluster_maps.common import (  # noqa: E402
    _ceil_log2,
    _find_cache,
    _read_cache_token_ids,
    _safe_name,
    cluster_frequency_report,
    code_vectors_from_base,
    evaluate_clustering,
    rebalance_cluster_codes,
)


def cluster_map_filename(tokenizer_name: str, *, code_vocab_size: int) -> str:
    return f"cluster_mapping_{_safe_name(tokenizer_name)}_gpt2id_bpe16_v{int(code_vocab_size)}.json"


def build_base_embeddings_pretrained(args, vocab_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use the input token-embedding matrix of a pretrained model (e.g. MDLM) as the
    base-token embeddings. The pretrained model must share the GPT-2 token id
    space (it usually does, since MDLM uses the GPT-2 tokenizer on OpenWebText);
    extra rows such as a trailing mask token are ignored by slicing the first
    ``vocab_size`` rows.

    Loading is delegated to scripts/pretrained_embeddings.py, which handles custom
    diffusion-LM backbones (MDLM/LLaDA) whose embeddings are not exposed via
    get_input_embeddings(), as well as raw / Hugging Face Hub checkpoint files.
    """
    from scripts.cluster_maps.pretrained_embeddings import load_pretrained_token_embeddings

    print(
        f"[cluster_owt] Loading pretrained embeddings "
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
    print(f"[cluster_owt] Pretrained embedding matrix: shape={emb.shape}")
    if emb.shape[0] < vocab_size:
        raise SystemExit(
            f"Pretrained embedding rows ({emb.shape[0]}) < tokenizer vocab_size ({vocab_size}). "
            f"The pretrained model must use the same token id space as --tokenizer_name."
        )

    vectors = np.ascontiguousarray(emb[:vocab_size]).astype(np.float32)
    found_mask = np.ones(vocab_size, dtype=bool)
    return vectors, found_mask


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default=f"{DATA_ROOT}/openwebtext")
    ap.add_argument("--tokenizer_name", type=str, default="gpt2")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--map_dir", type=str, default=None,
                    help="Directory to write the cluster map (default: --root). "
                         "Overridden by --out if both are given.")

    ap.add_argument("--n_clusters", type=int, default=16)

    # Which vocabulary to cluster.
    ap.add_argument("--codec", type=str, default="gpt2id_bpe16", choices=["gpt2id_bpe16"])

    # Where the base-token embeddings come from.
    ap.add_argument("--embedding_source", type=str, default="pretrained", choices=["pretrained"])
    ap.add_argument("--pretrained_model", type=str, default=None,
                    help="HF model id / local path whose input embeddings are used "
                         "(Continuous-Rivals-Discrete/langflow-owt).")
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

    # Second-stage code tokenizer.
    ap.add_argument("--code_tokenizer_path", type=str, default="tokenizer_gpt2id_bpe16_65536_base1024.json")
    ap.add_argument("--code_tokenizer_meta_path", type=str, default=None)

    ap.add_argument("--kmeans_n_init", type=int, default=10)
    ap.add_argument("--kmeans_max_iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)

    # Token-frequency-weighted cluster-usage report (scans a few dataset examples).
    ap.add_argument("--freq_max_tokens", type=int, default=1_000_000,
                    help="Scan up to this many tokens from the on-disk dataset cache "
                         "for the cluster-frequency report (0 disables it).")
    ap.add_argument("--freq_top_k", type=int, default=20,
                    help="Number of most-frequent clusters to print (0 = all).")
    ap.add_argument("--freq_bottom_k", type=int, default=10,
                    help="Number of least-frequent clusters to print.")
    ap.add_argument("--freq_examples_per_cluster", type=int, default=10,
                    help="Example tokens shown per cluster in the frequency report.")
    ap.add_argument("--freq_rebalance", action="store_true",
                    help="After clustering, permute cluster codes to maximise per-bit entropy "
                         "(2-opt local search over code assignments). Requires a token cache "
                         "(--freq_max_tokens > 0); falls back to type-occupancy counts otherwise.")
    ap.add_argument("--reserve_special_cluster", action="store_true",
                    help="Hold the special code tokens ([PAD]/[UNK]/[EOSEQ]/'\\n' — those with no "
                         "GPT-2 span, which otherwise get a RANDOM vector) out of the k-means fit "
                         "and pin them to a dedicated cluster id (n_clusters-1). Keeps "
                         "cluster_bits = ceil(log2(n_clusters)) and all bit-width invariants; "
                         "costs one semantic cluster. Strongly recommended: "
                         "PAD is ~29%% of packed slots and otherwise contaminates a real cluster.")
    ap.add_argument("--freq_rebalance_include_specials", action="store_true",
                    help="Count PAD/EOSEQ when rebalancing, i.e. optimise the per-bit entropy of "
                         "the bitstream the model ACTUALLY sees. Recommended for the packed "
                         "gpt2id_bpe16 codec, where PAD is ~29%% of slots and there is no padding "
                         "loss mask.")

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
    vocab_size_base = int(tokenizer.vocab_size)
    print(f"[cluster_owt] tokenizer={args.tokenizer_name} vocab_size={vocab_size_base}")
    print(f"[cluster_owt] codec={args.codec} embedding_source={args.embedding_source} "
          f"n_clusters={n_clusters} -> {cluster_bits} cluster bits/token")

    # ---- 1. base-token embeddings -------------------------------------------
    base_vecs, base_found = build_base_embeddings_pretrained(args, vocab_size_base)

    # ---- 2. vectors for the vocabulary we actually cluster ------------------
    code_tok, code_meta = load_gpt2id_bpe16_tokenizer_and_meta(
        root=root,
        tokenizer_path=args.code_tokenizer_path,
        meta_path=args.code_tokenizer_meta_path,
    )
    code_vocab_size = int(code_meta["actual_code_vocab_size"])
    offset = int(code_meta["base_char_offset"])
    vocab_size_gpt2 = int(code_meta["vocab_size_gpt2"])

    vectors, found_mask = code_vectors_from_base(
        base_vecs, base_found, code_tokenizer=code_tok, code_meta=code_meta, seed=args.seed
    )
    target_vocab = code_vocab_size
    out_name = cluster_map_filename(args.tokenizer_name, code_vocab_size=code_vocab_size)

    def id_to_text(i: int) -> str:
        surface = code_tok.id_to_token(int(i))
        if surface is None:
            return ""
        base_ids = [ord(ch) - offset for ch in surface]
        base_ids = [b for b in base_ids if 0 <= b < vocab_size_gpt2]
        return tokenizer.decode(base_ids) if base_ids else ""

    map_dir = Path(args.map_dir) if args.map_dir is not None else root
    out_path = Path(args.out) if args.out is not None else (map_dir / out_name)
    if out_path.exists() and not args.force:
        print(f"[cluster_owt] ✅ Cluster map found at {out_path}")
        print("[cluster_owt] Skipping generation. Use --force to overwrite.")
        return

    # Normalize to the unit sphere so Euclidean k-means approximates cosine
    # clustering (matches the cosine geometry used by the semantic ordering).
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    # ---- 3. k-means++ -------------------------------------------------------
    # Tokens with no embedding ([PAD]/[UNK]/[EOSEQ]/'\n', which have no GPT-2
    # span) otherwise get a RANDOM Gaussian vector from code_vectors_from_base and are clustered
    # on it. That is bad twice over: the assignment is arbitrary, and PAD — ~29% of every packed
    # gpt2id_bpe16 sequence — then drags whatever real semantic cluster it lands in to ~41% of
    # the realized token stream, capping per-bit entropy and conflating "is padding" with
    # whatever that cluster meant.
    #
    # --reserve_special_cluster instead holds those tokens OUT of the fit and pins them to a
    # dedicated id (the last one, n_clusters-1). The vocabulary is still partitioned into
    # exactly n_clusters ids, so cluster_bits = ceil(log2(n_clusters)) and every downstream
    # bit-width invariant is unchanged; the cost is one fewer *semantic* cluster.
    reserved_id: Optional[int] = None
    if args.reserve_special_cluster:
        n_semantic = n_clusters - 1
        if n_semantic < 2:
            raise SystemExit(f"--reserve_special_cluster needs n_clusters >= 3 (got {n_clusters})")
        reserved_id = n_semantic
        specials = np.where(~found_mask)[0]
        print(f"[cluster_owt] Reserving cluster id {reserved_id} for {len(specials)} special "
              f"token(s) {[int(s) for s in specials]}; fitting k-means++ on the remaining "
              f"{int(found_mask.sum())} tokens with k={n_semantic}, n_init={args.kmeans_n_init}...")
        km = KMeans(
            n_clusters=n_semantic,
            init="k-means++",
            n_init=int(args.kmeans_n_init),
            max_iter=int(args.kmeans_max_iter),
            random_state=int(args.seed),
        )
        sub_labels = km.fit_predict(vectors[found_mask]).astype(np.int64)
        labels = np.full(target_vocab, reserved_id, dtype=np.int64)
        labels[found_mask] = sub_labels
        # Quality is only meaningful over the tokens that were actually fitted.
        quality_vectors, quality_labels = vectors[found_mask], sub_labels
        quality_k, quality_found = n_semantic, np.ones(int(found_mask.sum()), dtype=bool)
        quality_ids = np.where(found_mask)[0]
        _id_to_text = id_to_text
        def quality_id_to_text(i: int, _ids=quality_ids, _f=_id_to_text) -> str:
            return _f(int(_ids[int(i)]))
    else:
        print(f"[cluster_owt] Running k-means++ (k={n_clusters}, n_init={args.kmeans_n_init}) "
              f"on {target_vocab} tokens...")
        km = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=int(args.kmeans_n_init),
            max_iter=int(args.kmeans_max_iter),
            random_state=int(args.seed),
        )
        labels = km.fit_predict(vectors).astype(np.int64)
        quality_vectors, quality_labels = vectors, labels
        quality_k, quality_found = n_clusters, found_mask
        quality_id_to_text = id_to_text

    eval_data: Dict[str, Any] = {}
    eval_data["clustering_quality"] = evaluate_clustering(
        quality_vectors,
        quality_labels,
        n_clusters=quality_k,
        inertia=float(km.inertia_),
        found_mask=quality_found,
        id_to_text=quality_id_to_text,
        seed=int(args.seed),
    )
    if reserved_id is not None:
        eval_data["special_cluster"] = {
            "reserved_id": int(reserved_id),
            "n_semantic_clusters": int(n_clusters - 1),
            "special_token_ids": [int(s) for s in np.where(~found_mask)[0]],
            "note": "specials excluded from the k-means fit; clustering_quality covers the "
                    "semantic clusters only",
        }

    # ---- 3b. token-frequency-weighted cluster usage -------------------------
    if int(args.freq_max_tokens) > 0:
        safe = _safe_name(args.tokenizer_name)
        cache_path = _find_cache(root, [
            f"cache_train_{safe}_gpt2id_bpe16_v{target_vocab}_*.uint16",
            f"cache_val_{safe}_gpt2id_bpe16_v{target_vocab}_*.uint16",
        ])
        exclude_ids = [int(code_meta["pad_id"]), int(code_meta["eoseq_id"])]
        layout = "packed"

        if cache_path is None:
            print(f"[cluster_owt] (no on-disk {args.codec} token cache under {root}; "
                  f"skipping cluster-frequency report)")
            scanned_ids: np.ndarray = np.array([], dtype=np.int64)
            scanned_exclude: List[int] = []
        else:
            scanned_ids, n_ex = _read_cache_token_ids(
                cache_path, layout=layout, max_tokens=int(args.freq_max_tokens)
            )
            scanned_exclude = exclude_ids
            print(f"[cluster_owt] cluster-frequency source: {cache_path.name}")
            freq = cluster_frequency_report(
                scanned_ids, labels, n_clusters=n_clusters, exclude_ids=scanned_exclude,
                id_to_text=id_to_text, examples_scanned=n_ex, tag="cluster_owt",
                top_k=int(args.freq_top_k),
                bottom_k=int(args.freq_bottom_k),
                examples_per_cluster=int(args.freq_examples_per_cluster),
            )
            if freq is not None:
                eval_data["frequency_report"] = freq
    else:
        scanned_ids = np.array([], dtype=np.int64)
        scanned_exclude = []

    # ---- 3c. optional code-assignment rebalancing ---------------------------
    if args.freq_rebalance:
        def _bit_entropy(pb: float) -> float:
            return (-pb * math.log2(pb) - (1 - pb) * math.log2(1 - pb)) if 0.0 < pb < 1.0 else 0.0

        # prefer real token-frequency counts; fall back to type-occupancy
        if scanned_ids.size > 0:
            # By default the frequency *report* excludes PAD/EOSEQ so its cluster listing is
            # readable. For the rebalance that exclusion is usually wrong: the diffusion model
            # has no padding loss mask, so it trains on and generates every slot, PAD included.
            # For gpt2id_bpe16 that is ~29% of all slots (1024 base tokens merge to ~728 code
            # tokens, then pad back to 1024), and PAD's cluster then dominates the realized
            # bitstream — so optimizing the pad-free distribution leaves the real per-bit
            # marginals badly skewed. --freq_rebalance_include_specials optimizes the realized
            # distribution instead.
            if args.freq_rebalance_include_specials:
                valid_ids = scanned_ids
                freq_source = "token-frequency (incl. PAD/EOSEQ; realized bitstream)"
            else:
                excl = set(int(x) for x in scanned_exclude)
                valid_ids = scanned_ids[~np.isin(scanned_ids, np.asarray(sorted(excl), dtype=scanned_ids.dtype))] if excl else scanned_ids
                freq_source = "token-frequency"
            cluster_freq_arr = np.bincount(labels[valid_ids], minlength=n_clusters).astype(np.float64)
        else:
            cluster_freq_arr = np.bincount(labels, minlength=n_clusters).astype(np.float64)
            freq_source = "type-occupancy (no token cache)"

        # per-bit P(1) before rebalancing (identity code: cluster k -> code k)
        freq_norm = cluster_freq_arr / cluster_freq_arr.sum() if cluster_freq_arr.sum() > 0 else cluster_freq_arr
        p_before = np.array([
            sum(freq_norm[k] for k in range(n_clusters) if (k >> b) & 1)
            for b in range(cluster_bits)
        ])
        H_before = sum(_bit_entropy(float(p)) for p in p_before)

        code, n_swaps, p_after = rebalance_cluster_codes(cluster_freq_arr, cluster_bits)
        H_after = sum(_bit_entropy(float(p)) for p in p_after)

        print("\n" + "=" * 70)
        print(f"[cluster_owt] Code-assignment rebalancing ({freq_source})")
        print(f"  swaps performed        : {n_swaps}")
        print(f"  total bit entropy      : {H_before:.4f} -> {H_after:.4f} bits "
              f"(max {float(cluster_bits):.1f})")
        print("  per-bit P(1)  before → after:")
        for b in range(cluster_bits):
            bar_before = int(round(p_before[b] * 20))
            bar_after  = int(round(p_after[b]  * 20))
            print(f"    bit {b}: {p_before[b]:.3f} [{('#' * bar_before).ljust(20)}]  "
                  f"-> {p_after[b]:.3f} [{('#' * bar_after).ljust(20)}]")
        print("=" * 70 + "\n")

        # apply permutation: token's cluster label k -> code[k]
        labels = code[labels]
        if reserved_id is not None:
            # The rebalance permutes codes, so the specials' final code is code[reserved_id],
            # not reserved_id. Record both, or the JSON lies about where PAD lives.
            eval_data["special_cluster"]["reserved_id_before_rebalance"] = int(reserved_id)
            eval_data["special_cluster"]["reserved_id"] = int(code[reserved_id])
        eval_data["rebalancing"] = {
            "freq_source": freq_source,
            "n_swaps": n_swaps,
            "entropy_before": round(float(H_before), 4),
            "entropy_after": round(float(H_after), 4),
            "entropy_max": float(cluster_bits),
            "p_per_bit_before": [round(float(p), 4) for p in p_before],
            "p_per_bit_after":  [round(float(p), 4) for p in p_after],
        }

    # ---- 4. save ------------------------------------------------------------
    token_to_cluster: Dict[str, int] = {str(tid): int(labels[tid]) for tid in range(target_vocab)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenizer_name": str(args.tokenizer_name),
                "codec": str(args.codec),
                "embedding_source": str(args.embedding_source),
                "pretrained_model": args.pretrained_model,
                "vocab_size": int(target_vocab),
                "n_clusters": int(n_clusters),
                "cluster_bits": int(cluster_bits),
                "token_to_cluster": token_to_cluster,
            },
            f,
        )

    eval_path = out_path.with_name(out_path.stem + "_eval.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)

    print(f"[cluster_owt] ✅ Saved cluster map (codec={args.codec}, k={n_clusters}) to: {out_path}")
    print(f"[cluster_owt] ✅ Saved eval report to: {eval_path}")
    print("[cluster_owt] Cluster bits are appended after the token info bits in the dataset.")


if __name__ == "__main__":
    main()
