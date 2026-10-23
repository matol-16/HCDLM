"""Evaluate CoBit / H-CoBit checkpoints as in the paper.

    python evaluate.py profile lm1b hcobit                     # per-modality noise profiles
    python evaluate.py genppl  lm1b hcobit_pms                 # Table 1 / Figure 4 frontier
    python evaluate.py genppl  lm1b hcobit_pms --nfe 512 --gamma 0.3   # one operating point
    python evaluate.py mauve   owt cobit                       # Table 11
    python evaluate.py gsm8k   tinygsm hcobit --nfe 512 --gamma 0.38   # Table 2
    python evaluate.py genppl  lm1b cobit --checkpoint checkpoints/my_cobit/model.pt

Arms (configs/sampling.py): cobit, cobit_snoise1.05, hcobit, hcobit_pms; GSM8K takes cobit
or hcobit. The checkpoint defaults to checkpoints/<model>_<dataset>/model.pt
(scripts/download_checkpoints.py, or scripts/make_release.py for your own runs); its
noise schedule is read from the same directory.
Raw outputs go to runs/eval/<dataset>_<arm>/; one row per operating point is merged
into --results (default runs/eval/results/), which scripts/report.py reads.

GenPPL and MAUVE runs may use several GPUs (torchrun --nproc_per_node=N evaluate.py ...);
the generated samples are the same on 1 or 2 GPUs (see MICRO_BATCH in configs/sampling.py).
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", "1")

import torch  # noqa: E402

from configs import load_config  # noqa: E402
from configs import sampling  # noqa: E402

ROOT = Path(__file__).resolve().parent


def _model_cfg(dataset, arm):
    model = "hcobit" if arm.startswith("hcobit") else "cobit"
    return model, load_config(str(ROOT / f"configs/{dataset}.py") + f":{model}")


def _load_model(cfg, device, compile_model: bool):
    from evaluation.utils import load_checkpoint
    from models import create_model
    from utils.ema import EMA

    model = create_model(cfg).to(device)
    load_checkpoint(model, EMA(model, decay=0.0), Path(cfg.evaluation.checkpoint_path), device)
    model.eval()
    if compile_model:
        model = torch.compile(model, mode=str(cfg.evaluation.compile_mode), fullgraph=False)
    return model


def run_text_eval(args, metric: str):
    from data import get_loader
    from evaluation import entropy, tidy
    from evaluation.distributed import barrier, init_distributed_if_needed, set_global_seed
    from evaluation.genppl import evaluate_genppl
    from evaluation.mauve import evaluate_mauve
    from evaluation.noise_profile import evaluate_sigma_loss_profile

    model_name, cfg = _model_cfg(args.dataset, args.arm)
    ckpt = Path(args.checkpoint or ROOT / f"checkpoints/{model_name}_{args.dataset}/model.pt")
    if metric == "profile" and model_name != "hcobit":
        raise SystemExit("profile estimates H-CoBit's per-modality profiles; CoBit samples with its trained schedule")
    if metric == "profile":
        points, out = [], Path(args.out or ckpt.parent)
    else:
        kind = "mauve" if metric == "mauve" else "frontier"
        points = [(args.nfe, args.gamma)] if args.nfe else sampling.grid(args.dataset, args.arm, kind)
        out = Path(args.out or ROOT / f"runs/eval/{args.dataset}_{args.arm}/{kind}")
    sampling.eval_config(cfg, dataset=args.dataset, arm=args.arm, checkpoint=ckpt, out_dir=out, points=points)
    if args.num_samples:
        cfg.evaluation.external_ppl.num_samples = cfg.evaluation.mauve.num_samples = int(args.num_samples)

    dist_info = init_distributed_if_needed()
    rank0 = dist_info.rank == 0
    device = torch.device(f"cuda:{dist_info.local_rank}" if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    set_global_seed(42, rank=dist_info.rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out.mkdir(parents=True, exist_ok=True)

    run_meta = dict(
        timestamp_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        experiment=f"{args.dataset}_{args.arm}",
        framework=cfg.framework,
        dataset=cfg.data.dataset,
        checkpoint=ckpt.name,
        world_size=dist_info.world_size,
        compile_enabled=not args.no_compile,
        compile_mode=cfg.evaluation.compile_mode,
    )
    test_loader = get_loader(cfg, split="test")
    model = _load_model(cfg, device, compile_model=(metric != "profile" and not args.no_compile))

    if metric == "profile":
        evaluate_sigma_loss_profile(cfg, model, test_loader, device, rank0, run_meta)
        if rank0:
            for g in ("info", "cluster"):
                src = out / "sigma_loss_profile" / f"entropy_profile_{g}.pt"
                if src.exists():
                    (out / src.name).write_bytes(src.read_bytes())
        return

    evaluate_genppl(cfg, model, True, test_loader, device, rank0, run_meta)
    if metric == "mauve":
        evaluate_mauve(cfg, model, True, test_loader, device, rank0, run_meta)
    barrier()
    if rank0:
        entropy.evaluate_entropy(cfg, test_loader, run_meta, use_ema=True)
        tidy.export(Path(cfg.evaluation.results_csv), dataset=args.dataset, arm=args.arm,
                    results_dir=Path(args.results), kind="mauve" if metric == "mauve" else "frontier")


def run_gsm8k(args):
    from evaluation.gsm8k import evaluate_gsm8k

    cfg = load_config(str(ROOT / "configs/tinygsm.py") + f":{args.arm}")
    ckpt = Path(args.checkpoint or ROOT / f"checkpoints/{args.arm}_tinygsm/model.pt")
    out = Path(args.out or ROOT / f"runs/eval/tinygsm_{args.arm}/gsm8k")
    res = evaluate_gsm8k(cfg, ckpt, nfe=args.nfe or 512, gamma=args.gamma if args.gamma is not None else 0.38,
                         out_dir=out, limit=args.limit, batch_size=args.batch_size)
    path = Path(args.results) / "gsm8k.csv"
    rows = []
    if path.exists():
        with open(path) as f:
            rows = [r for r in csv.DictReader(f)
                    if (r["arm"], int(r["nfe"]), float(r["gamma"])) != (args.arm, res["nfe"], res["gamma"])]
    rows.append(dict(arm=args.arm, checkpoint=ckpt.name, nfe=res["nfe"], gamma=res["gamma"],
                     s_noise=res["s_noise"] or "", num_correct=res["num_correct"],
                     num_examples=res["num_examples"], accuracy=round(res["accuracy"], 5)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[-1]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metric", choices=["profile", "genppl", "mauve", "gsm8k"])
    ap.add_argument("dataset", choices=["lm1b", "owt", "tinygsm"])
    ap.add_argument("arm", choices=["cobit", "cobit_snoise1.05", "hcobit", "hcobit_pms"])
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--nfe", type=int, default=None, help="evaluate one operating point instead of the sweep")
    ap.add_argument("--gamma", type=float, default=None, help="churn per step, with --nfe")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--results", type=Path, default=ROOT / "runs/eval/results")
    ap.add_argument("--limit", type=int, default=None, help="gsm8k: first N problems only")
    ap.add_argument("--num_samples", type=int, default=None, help="genppl/mauve: fewer samples (smoke tests)")
    ap.add_argument("--batch_size", type=int, default=64, help="gsm8k: problems per sampling batch")
    ap.add_argument("--no_compile", action="store_true")
    args = ap.parse_args()
    if (args.nfe is None) != (args.gamma is None) and args.metric != "gsm8k":
        ap.error("--nfe and --gamma go together")
    if args.metric == "gsm8k":
        if args.dataset != "tinygsm" or args.arm not in ("cobit", "hcobit"):
            ap.error("gsm8k evaluates tinygsm cobit or tinygsm hcobit")
        run_gsm8k(args)
    else:
        if args.dataset == "tinygsm":
            ap.error("tinygsm is evaluated with the gsm8k metric")
        run_text_eval(args, args.metric)


if __name__ == "__main__":
    main()
