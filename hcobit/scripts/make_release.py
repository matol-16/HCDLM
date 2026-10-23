"""Package a trained run as a release folder (what download_checkpoints.py fetches).

    python scripts/make_release.py --run runs/lm1b_hcobit --step 600000 --out checkpoints/hcobit_lm1b
    python scripts/make_release.py --run runs/lm1b_cobit --out checkpoints/cobit_lm1b   # final state (last.pt)
    python scripts/make_release.py --run runs/tinygsm_hcobit --step 250000 --out checkpoints/hcobit_tinygsm --task

--step picks the archival checkpoint step=<step>.pt; without it, the run's final state
(last.pt) is packaged. CoBit-S on LM1B usually ends at its 74-epoch cap before the 1M-step
LR horizon (each resume skips the rest of an epoch; CoBit's released checkpoint is at step
992,719), so there may be no step=1000000.pt.

model.pt holds the EMA weights only (evaluation applies EMA by default, so this is
numerically identical to evaluating the full training checkpoint). Next to it go the noise
schedules the evaluation reads:
  CoBit              : entropy_{pdf,cdf,sigmas,edges}.pt, as trained
  H-CoBit, LM1B / OWT: entropy_profile_{info,cluster}.pt  (python evaluate.py profile ...)
  H-CoBit, TinyGSM   : entropy_{pdf,cdf,sigmas,edges}_{info,cluster}.pt, as trained
and, for TinyGSM (--task), sigma_data.json.
"""
import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path

import torch

TABLES = ("pdf", "cdf", "sigmas", "edges")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="runs/<experiment>")
    ap.add_argument("--step", type=int, default=None, help="step=<step>.pt (default: last.pt, the final state)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile_dir", type=Path, default=None,
                    help="H-CoBit: copy entropy_profile_{info,cluster}.pt from here (else run `evaluate.py profile`)")
    ap.add_argument("--task", action="store_true", help="TinyGSM: ship the trained schedule tables + sigma_data.json")
    args = ap.parse_args()

    name = "last.pt" if args.step is None else f"step={args.step:09d}.pt"
    ckpt = torch.load(args.run / "checkpoints" / name, map_location="cpu", weights_only=False)
    model, shadow = ckpt["model"], ckpt["ema"]["shadow"]
    baked = OrderedDict(model)
    for k, v in shadow.items():
        if k not in baked:
            raise KeyError(f"EMA key {k} absent from the model state dict")
        baked[k] = v.clone()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": baked, "global_step": ckpt.get("global_step"), "epoch": ckpt.get("epoch"),
                "ema_decay": ckpt["ema"].get("decay")},
               args.out / "model.pt")

    hcobit = bool(json.loads((args.run / "config.json").read_text())["data"].get("cluster_bits_enabled", False))
    files = []
    if not hcobit:
        files = [f"entropy_{t}.pt" for t in TABLES]
    elif args.task:
        files = [f"entropy_{t}_{g}.pt" for t in TABLES for g in ("info", "cluster")]
    if args.task:
        files.append("sigma_data.json")
    for f in files:
        shutil.copy(args.run / f, args.out / f)
    if hcobit and not args.task:
        if args.profile_dir is not None:
            for g in ("info", "cluster"):
                shutil.copy(args.profile_dir / f"entropy_profile_{g}.pt", args.out / f"entropy_profile_{g}.pt")
        elif not (args.out / "entropy_profile_info.pt").exists():
            print("next: python evaluate.py profile <dataset> hcobit --checkpoint", args.out / "model.pt")
    print(f"release ({name}, step {ckpt.get('global_step')}) -> {args.out}: {sorted(x.name for x in args.out.iterdir())}")


if __name__ == "__main__":
    main()
