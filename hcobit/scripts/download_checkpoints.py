"""Download the checkpoints evaluated in the paper into checkpoints/<name>/.

    python scripts/download_checkpoints.py              # all of them
    python scripts/download_checkpoints.py hcobit_lm1b  # one

CoBit-S baselines are CoBit's own release (huggingface.co/gbatzolis/CoBit), H-CoBit
ours (huggingface.co/matol-16/hcdlm). Each checkpoint comes with the noise schedule it
is sampled with. All checkpoints are EMA weights.
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
HCDLM_REPO = "matol-16/hcdlm"
COBIT_REPO = "gbatzolis/CoBit"

_PROFILES = ("entropy_profile_info.pt", "entropy_profile_cluster.pt")
_COBIT_TABLES = tuple(f"entropy_{t}.pt" for t in ("pdf", "cdf", "sigmas", "edges"))
_TRAINED = tuple(f"entropy_{t}_{g}.pt" for t in ("pdf", "cdf", "sigmas", "edges") for g in ("info", "cluster")) + ("sigma_data.json",)
CHECKPOINTS = {
    "cobit_lm1b": (COBIT_REPO, {"model.pt": "checkpoints/cobit_s_lm1b_1M_ema.pt",
                                **{f: f"entropy_tables/lm1b/{f}" for f in _COBIT_TABLES}}),
    "cobit_owt": (COBIT_REPO, {"model.pt": "checkpoints/cobit_s_owt_750k_ema.pt",
                               **{f: f"entropy_tables/owt/{f}" for f in _COBIT_TABLES}}),
    "hcobit_lm1b": (HCDLM_REPO, {f: f"hcobit_lm1b/{f}" for f in ("model.pt",) + _PROFILES}),
    "hcobit_owt": (HCDLM_REPO, {f: f"hcobit_owt/{f}" for f in ("model.pt",) + _PROFILES}),
    "hcobit_tinygsm": (HCDLM_REPO, {f: f"hcobit_tinygsm/{f}" for f in ("model.pt",) + _TRAINED}),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"any of {', '.join(CHECKPOINTS)} (default: all)")
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoints")
    args = ap.parse_args()
    unknown = set(args.names) - set(CHECKPOINTS)
    if unknown:
        ap.error(f"unknown checkpoint(s): {', '.join(sorted(unknown))}")
    for name in args.names or CHECKPOINTS:
        repo, files = CHECKPOINTS[name]
        for local, remote in files.items():
            dst = args.out / name / local
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(hf_hub_download(repo, remote), dst)
            print(f"{dst}")


if __name__ == "__main__":
    main()
