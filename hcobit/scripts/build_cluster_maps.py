"""Rebuild the paper's cluster maps and compare them with the shipped ones.

    python scripts/build_cluster_maps.py               # all three datasets
    python scripts/build_cluster_maps.py owt tinygsm   # some of them

Downloads the embedding models, runs the builders in scripts/cluster_maps/ with the
settings of the paper, writes each map to runs/cluster_maps/<dataset>/k_16/ (never over
the shipped maps) and reports whether it assigns every token to the same cluster as
cluster_maps/<dataset>/k_16/. OpenWebText and TinyGSM need their data caches first
(scripts/prepare_data.py owt / tinygsm), which the bit-balancing step reads.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.base import DATA_ROOT  # noqa: E402

N_CLUSTERS = 16
SFLM_REPO = "jdeschena/s-flm"
SFLM_CKPT = "tinygsm/sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt"
# Rebalance the cluster codes over the training cache, PAD included, with PAD & co. in a cluster of their own.
_SPECIALS = ["--reserve_special_cluster", "--freq_rebalance", "--freq_rebalance_include_specials"]

DATASETS = {
    "lm1b": dict(
        map_name="cluster_mapping_bert_base_uncased.json",
        args=["--pretrained_model", "Continuous-Rivals-Discrete/langflow-lm1b", "--trust_remote_code",
              "--freq_max_tokens", "0"],
    ),
    "owt": dict(
        map_name="cluster_mapping_gpt2_gpt2id_bpe16_v65536.json",
        cache=("openwebtext", "cache_train_gpt2_gpt2id_bpe16_v65536_*.uint16", "owt"),
        args=["--pretrained_model", "Continuous-Rivals-Discrete/langflow-owt", "--trust_remote_code",
              *_SPECIALS, "--freq_max_tokens", "50000000"],
    ),
    "tinygsm": dict(
        map_name="cluster_mapping_HuggingFaceTB_SmolLM_135M.json",
        cache=("tinygsm", "*_train_ids.uint16", "tinygsm"),
        args=[*_SPECIALS, "--freq_max_tokens", "10000000"],  # + --checkpoint_path, see sflm_embeddings()
    ),
}


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts/cluster_maps" / script), *args]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def sflm_embeddings(work: Path) -> Path:
    """The S-FLM TinyGSM input embeddings, extracted once from its checkpoint."""
    out = work / "sfm_tinygsm_input_embeddings.pt"
    if not out.exists():
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(SFLM_REPO, SFLM_CKPT)
        run("extract_sflm_embeddings.py", "--ckpt", ckpt, "--out", str(out))
    return out


def same_assignment(a: Path, b: Path) -> bool:
    return json.loads(a.read_text())["token_to_cluster"] == json.loads(b.read_text())["token_to_cluster"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help=f"any of {', '.join(DATASETS)} (default: all)")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/cluster_maps")
    args = ap.parse_args()
    unknown = set(args.datasets) - set(DATASETS)
    if unknown:
        ap.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    names = args.datasets or list(DATASETS)

    for name in names:  # fail before any long step, not halfway through
        if "cache" in DATASETS[name]:
            sub, pattern, prep = DATASETS[name]["cache"]
            if not any((Path(DATA_ROOT) / sub).glob(pattern)):
                sys.exit(f"No {name} training cache under {Path(DATA_ROOT) / sub}: "
                         f"run `python scripts/prepare_data.py {prep}` first.")

    results = {}
    for name in names:
        spec = DATASETS[name]
        map_dir = args.out / name / f"k_{N_CLUSTERS}"
        extra = ["--checkpoint_path", str(sflm_embeddings(args.out))] if name == "tinygsm" else []
        run(f"{name}.py", "--map_dir", str(map_dir), "--n_clusters", str(N_CLUSTERS),
            *spec["args"], *extra, "--force")
        results[name] = same_assignment(map_dir / spec["map_name"],
                                        ROOT / "cluster_maps" / name / f"k_{N_CLUSTERS}" / spec["map_name"])

    print("\nSame token -> cluster assignment as the shipped map:")
    for name, same in results.items():
        note = "" if same or name != "lm1b" else "  (expected: see 'Reproducibility' in README.md)"
        print(f"  {name:8s} {'yes' if same else 'no'}{note}  -> {args.out / name / f'k_{N_CLUSTERS}'}")


if __name__ == "__main__":
    main()
