"""Train CoBit / H-CoBit.

    torchrun --standalone --nproc_per_node=4 train.py --config configs/lm1b.py:hcobit --stop_at_step 600000

The run lives in runs/<experiment>/. Relaunching the same command resumes from
runs/<experiment>/checkpoints/last.pt; the config saved at the first launch
(runs/<experiment>/config.json) then takes precedence over the Python config.
"""
import argparse
import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from ml_collections import config_dict

from configs import load_config
from trainers import Trainer


def _setup_ddp():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 0, 1
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=20),
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    return rank, local_rank, world_size


def _update_cfg_from_dict(cfg, d, skip_sections=("logging", "system", "evaluation")):
    for k, v in d.items():
        if k in skip_sections:
            continue
        if isinstance(v, dict):
            if not hasattr(cfg, k) or getattr(cfg, k) is None:
                setattr(cfg, k, config_dict.ConfigDict())
            _update_cfg_from_dict(getattr(cfg, k), v, skip_sections=skip_sections)
        else:
            setattr(cfg, k, v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/<dataset>.py[:<model>]")
    ap.add_argument("--stop_at_step", type=int, default=0,
                    help="Stop at this step without changing the LR horizon (optim.total_steps).")
    args = ap.parse_args()

    rank, local_rank, world_size = _setup_ddp()
    cfg = load_config(args.config)
    cfg.system = config_dict.ConfigDict()
    cfg.system.distributed = world_size > 1
    cfg.system.global_rank = rank
    cfg.system.local_rank = local_rank
    cfg.system.world_size = world_size
    if args.stop_at_step:
        os.environ["STOP_AT_STEP"] = str(args.stop_at_step)

    run_dir = Path("runs") / cfg.experiment
    saved_cfg = run_dir / "config.json"
    if (run_dir / "checkpoints" / "last.pt").exists() and saved_cfg.exists():
        _update_cfg_from_dict(cfg, json.loads(saved_cfg.read_text()))
        if rank == 0:
            print(f"Resuming '{cfg.experiment}' with its saved config.")
    cfg._config_path = args.config

    try:
        Trainer(cfg).train()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
