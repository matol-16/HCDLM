"""CoBit / H-CoBit training loop.

One step (``_step``):
  1. draw the noise level from the online information-uniform schedule
     (utils/schedule_controller.py). H-CoBit draws one per modality (sigma_token,
     sigma_cluster) and corrupts each bit block at its own level (equation 4). With a
     prompt (GSM8K), the prompt bits stay clean;
  2. self-conditioning: with probability 1/2 a first no-grad pass provides x0_hat;
  3. EDM-weighted squared error of sigmoid(logits) (equation 5), lambda_2 on the
     cluster bits;
  4. AdamW step, LR schedule, EMA; the per-example errors feed the noise schedule.
"""
from __future__ import annotations

import os

os.environ.setdefault("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", "1")

import json
import math
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch._dynamo
import torch.distributed as dist
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import get_dataloaders
from diffusion.continuous.logit_postprocess import _model_logits_continuous
from diffusion.continuous.losses import binary_score_interpolation_loss
from diffusion.continuous.processes import ContinuousForwardProcess
from models import create_model
from utils.bit_groups import bit_group_masks
from utils.ema import EMA
from utils.optim import get_optimizer_and_scheduler
from utils.schedule_controller import EntropyScheduleController

try:
    import wandb
except ImportError:  # optional
    wandb = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 64


def _ddp_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _unwrap_all(model: torch.nn.Module) -> torch.nn.Module:
    m = model
    while True:
        if hasattr(m, "module"):
            m = m.module
        elif hasattr(m, "_orig_mod"):
            m = m._orig_mod
        else:
            return m


def _set_seed(cfg):
    seed = int(cfg.train.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    deterministic = bool(getattr(cfg.train, "deterministic", True))
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def _save_config(cfg, run_dir: Path):
    """runs/<experiment>/config.json: replayed over the Python config on resume (train.py)."""
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, sort_keys=True, default=str)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        # ── distributed / device ─────────────────────────────────────────────
        self.ddp_active = bool(getattr(cfg.system, "distributed", False)) and _ddp_is_on()
        self.rank = int(cfg.system.global_rank) if self.ddp_active else 0
        self.local_rank = int(cfg.system.local_rank) if self.ddp_active else 0
        self.world_size = int(cfg.system.world_size) if self.ddp_active else 1
        self.is_master = self.rank == 0
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device.index)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        else:
            self.device = torch.device("cpu")
        self.cfg.device = str(self.device)
        _set_seed(cfg)

        # ── run directory ───────────────────────────────────────────────────
        self.run_dir = Path("runs") / cfg.experiment
        self.ckpt_dir = self.run_dir / "checkpoints"
        if self.is_master:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            _save_config(cfg, self.run_dir)
        if self.ddp_active:
            dist.barrier()
        self.use_wandb = bool(self.is_master and getattr(cfg.logging, "use_wandb", False) and wandb is not None)
        if self.use_wandb:
            wandb.init(project=cfg.logging.project, name=cfg.experiment, config=cfg.to_dict(),
                       dir=str(self.run_dir), mode=getattr(cfg.logging, "mode", "online"))

        ck = cfg.train.checkpointing
        self.save_top_k = int(ck.save_top_k)
        self.best_metric = math.inf
        self.best_ckpts: List[dict] = []
        self.interval_every = int(ck.interval.every_steps) if bool(ck.interval.enabled) else 0
        self.resume_every = int(ck.resume_interval.every_steps) if bool(ck.resume_interval.enabled) else 0

        # ── data (global batch split across GPUs, no gradient accumulation) ─
        raw_train, raw_val, _ = get_dataloaders(cfg)
        nw = int(getattr(cfg.data, "num_workers", 0))
        dl_kw = dict(num_workers=nw, pin_memory=bool(getattr(cfg.data, "pin_memory", True)))
        if nw > 0:
            dl_kw.update(prefetch_factor=int(getattr(cfg.data, "prefetch_factor", 2)), persistent_workers=True)
        if self.ddp_active:
            if cfg.train.batch_size % self.world_size:
                raise ValueError(f"batch_size {cfg.train.batch_size} not divisible by world_size {self.world_size}")
            bs = cfg.train.batch_size // self.world_size
            self.train_loader = torch.utils.data.DataLoader(
                raw_train.dataset, batch_size=bs, drop_last=True, shuffle=False,
                sampler=DistributedSampler(raw_train.dataset, num_replicas=self.world_size, rank=self.rank,
                                           shuffle=True, drop_last=True), **dl_kw)
            self.val_loader = torch.utils.data.DataLoader(
                raw_val.dataset, batch_size=bs, drop_last=True, shuffle=False,
                sampler=DistributedSampler(raw_val.dataset, num_replicas=self.world_size, rank=self.rank,
                                           shuffle=False, drop_last=True), **dl_kw)
        else:
            self.train_loader = torch.utils.data.DataLoader(
                raw_train.dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True, **dl_kw)
            self.val_loader = torch.utils.data.DataLoader(
                raw_val.dataset, batch_size=cfg.train.batch_size, shuffle=False, drop_last=True, **dl_kw)

        # ── model / optimizer / EMA / AMP ───────────────────────────────────
        model = create_model(cfg).to(self.device)
        if self.ddp_active:
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
        if bool(getattr(cfg.train, "use_compile", False)):
            model = torch.compile(model, mode=getattr(cfg.train, "compile_mode", "default"), fullgraph=False)
        self.model = model
        self.ema = EMA(self.model, decay=cfg.train.ema_decay)
        self.opt, self.lr_sched = get_optimizer_and_scheduler(self.model, cfg, 0)
        self.amp_enabled = bool(getattr(cfg.train, "use_fp16", False))
        dtype = str(getattr(cfg.train, "amp_dtype", "bf16")).lower()
        if not self.amp_enabled:
            self.amp_dtype = torch.float32
        elif dtype in {"fp16", "float16"} or not torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.bfloat16
        self.use_scaler = self.amp_enabled and self.device.type == "cuda" and self.amp_dtype == torch.float16
        self.scaler = GradScaler(enabled=self.use_scaler)
        self.grad_clip = float(getattr(cfg.optim, "grad_clip", 1.0))

        # ── noise schedule (one, or one per modality) and loss weighting ────
        self.dual = bool(getattr(cfg.data, "cluster_bits_enabled", False))
        if self.dual != bool(getattr(cfg.train, "entropy_dual_enabled", False)):
            raise ValueError("cluster bits and the per-modality schedule go together")
        self.proc = ContinuousForwardProcess(cfg)
        self.schedule = EntropyScheduleController(self, self.proc, dual=self.dual)
        self.group_norm = None
        if self.dual:
            masks = bit_group_masks(cfg, self.device)
            S = masks["info"].numel()
            self.m_info = masks["info"].view(1, S)
            self.m_cluster = masks["cluster"].view(1, S)
            self.per_position_cin = bool(cfg.train.entropy_dual_per_position_cin)
            self.per_position_mf = bool(cfg.train.entropy_dual_per_position_matched_filter)
        if self.dual and bool(getattr(cfg.train, "cluster_loss_weight_enabled", False)):
            # lambda_2: token and cluster bits averaged separately, L_token + lambda_2 L_cluster
            self.group_norm = dict(info_mask=self.m_info.float(), cluster_mask=self.m_cluster.float(),
                                   alpha=float(cfg.train.cluster_loss_weight_alpha))

        # ── resume ──────────────────────────────────────────────────────────
        self.global_step = 0
        self.start_epoch = self._resume()
        self._next_interval = (self.global_step // self.interval_every + 1) * self.interval_every if self.interval_every else None
        self._next_resume = (self.global_step // self.resume_every + 1) * self.resume_every if self.resume_every else None
        self.schedule.load_tables_if_any()
        if self.is_master:
            n = sum(p.numel() for p in _unwrap_all(self.model).parameters())
            print(f"Experiment {cfg.experiment}: {n / 1e6:.1f}M parameters, world_size={self.world_size}")

    # ──────────────────────────────────────────────────────────────────────
    # sigma_data
    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _estimate_sigma_data(self, num_batches: int = 10):
        """CoBit sets sigma_data to the std of the training bits (10 batches, rank 0) and
        records it in runs/<experiment>/sigma_data.json (the GSM8K evaluation reads it)."""
        n_pix, mean, m2 = 0, 0.0, 0.0
        for k, batch in enumerate(self.train_loader):
            if k >= num_batches:
                break
            x = (batch["x0"] if isinstance(batch, dict) else batch).to(self.device).float()
            if x.max() > 1:
                x = x / x.max()
            flat = x.view(-1)
            n_new = flat.numel()
            n_tot = n_pix + n_new
            delta = flat.mean() - mean
            mean += delta * n_new / max(n_tot, 1)
            m2 += flat.var(unbiased=False) * n_new + (delta ** 2) * n_pix * n_new / max(n_tot, 1)
            n_pix = n_tot
        sigma = float((m2 / max(n_pix, 1)).sqrt().item())
        self.cfg.diffusion.continuous.sigma_data = sigma
        (self.run_dir / "sigma_data.json").write_text(json.dumps({"sigma_data": sigma}))
        print(f"sigma_data estimated: {sigma:.4f}")

    # ──────────────────────────────────────────────────────────────────────
    # One step
    # ──────────────────────────────────────────────────────────────────────
    def _step(self, batch, is_train: bool) -> float:
        cfg = self.cfg
        if isinstance(batch, dict):
            x0, prefix_mask = batch["x0"], batch.get("prefix_mask")
        else:
            x0, prefix_mask = batch, None
        B = x0.size(0)
        x0_clean = x0.to(self.device, non_blocking=True).view(B, -1).to(torch.float32)
        S = x0_clean.size(1)

        # Prompt conditioning (GSM8K): prompt bits stay clean and out of the loss.
        cond = prefix_mask is not None
        if cond:
            prefix_mask = prefix_mask.to(self.device, non_blocking=True).view(B, -1).bool()
            noise_prefix = bool(getattr(cfg.cond, "noise_prefix", False))
            suffix_only = bool(getattr(cfg.cond, "loss_on_suffix_only", True))
            p_uncond = max(0.0, min(1.0, float(getattr(cfg.cond, "p_uncond", 0.0))))

        if self.dual:
            # Per-modality noise levels from one shared base time.
            sigma_info, sigma_cluster = self.schedule.draw(B)
            sigma_pos = sigma_info.view(-1, 1) * self.m_info + sigma_cluster.view(-1, 1) * self.m_cluster  # [B,S]
            model_kw = dict(
                sigma_pos=sigma_pos if self.per_position_cin else None,
                sigma_pos_mf=sigma_pos if self.per_position_mf else None,
            )
            sigma, sigma_weight = sigma_info, sigma_pos  # the time embedding takes the token sigma
        else:
            (sigma,) = self.schedule.draw(B)
            sigma_pos = sigma.view(-1, 1)  # [B,1]
            model_kw = {}
            sigma_weight = sigma

        prefix_used, keep = None, None
        if cond and not noise_prefix:
            # Prompt dropout (classifier-free style; p_uncond = 0 in the paper's runs).
            drop = torch.rand(B, device=self.device) < p_uncond
            prefix_used = x0_clean.clone()
            if drop.any():
                replace = drop.view(B, 1) & prefix_mask
                prefix_used[replace] = 0.5  # cfg.cond.null_strategy = "half"
            keep = ~drop  # prompt-dropped examples do not feed the noise schedule
        xt = torch.empty_like(x0_clean)
        xt.normal_()
        xt.mul_(sigma_pos)
        xt.add_(x0_clean)
        if prefix_used is not None:
            xt[prefix_mask] = prefix_used[prefix_mask]

        # Self-conditioning.
        x0_hat = None
        p_sc = float(getattr(cfg.train, "self_condition_prob", 0.5))
        if bool(getattr(cfg.model, "self_condition", False)) and p_sc > 0.0:
            sc_mask = torch.rand(B, device=self.device) < p_sc
            if sc_mask.any() or prefix_used is not None:
                x0_hat = torch.zeros_like(xt)
            if sc_mask.any():
                with autocast(self.device.type, enabled=self.amp_enabled, dtype=self.amp_dtype):
                    logits_sc = _model_logits_continuous(self.model, cfg, xt, sigma, None, **model_kw)
                est = torch.sigmoid(logits_sc.float()).detach().to(dtype=xt.dtype)
                if sc_mask.all():
                    x0_hat.copy_(est)
                else:
                    x0_hat[sc_mask] = est[sc_mask]
                del logits_sc, est
            if prefix_used is not None:
                x0_hat[prefix_mask] = prefix_used[prefix_mask]

        with autocast(self.device.type, enabled=self.amp_enabled, dtype=self.amp_dtype):
            logits = _model_logits_continuous(self.model, cfg, xt, sigma, x0_hat, **model_kw)
            loss_mask = (~prefix_mask).float() if (cond and suffix_only and not noise_prefix) else None
            if self.dual:
                loss = binary_score_interpolation_loss(
                    logits, x0_clean, sigma_weight, cfg, mask=loss_mask, group_norm=self.group_norm
                )
            else:
                loss, metric = binary_score_interpolation_loss(
                    logits, x0_clean, sigma_weight, cfg, return_entropy_metric=True, mask=loss_mask
                )
                pushes = [("all", sigma, metric)]

        if self.dual:
            # Per-modality errors (no grad), each at its own sigma, for the noise schedule.
            with torch.no_grad():
                pushes = []
                for g, m, s in (("info", self.m_info, sigma_info), ("cluster", self.m_cluster, sigma_cluster)):
                    mask_g = m.expand(B, S)
                    if loss_mask is not None:
                        mask_g = mask_g * loss_mask
                    _, metric_g = binary_score_interpolation_loss(
                        logits.detach(), x0_clean, s, cfg, return_entropy_metric=True, mask=mask_g
                    )
                    pushes.append((g, s, metric_g))
        del logits, xt, x0_hat

        if is_train:
            self.opt.zero_grad(set_to_none=True)
            if self.use_scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()
            self.lr_sched.step()
            self.ema.update(self.model)

        if bool(cfg.train.entropy_compute):
            if keep is None:
                keep = torch.ones(B, dtype=torch.bool, device=self.device)
            if keep.any():
                for g, s, metric_g in pushes:
                    self.schedule.push(g, s[keep], metric_g[keep])
        return loss.item()

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        self.ema.apply(self.model)
        total = torch.tensor(0.0, device=self.device)
        count = torch.tensor(0.0, device=self.device)
        for batch in tqdm(self.val_loader, desc="Validating", leave=False, disable=not self.is_master):
            total += self._step(batch, is_train=False)
            count += 1.0
        if self.ddp_active:
            dist.all_reduce(total)
            dist.all_reduce(count)
        self.ema.restore(self.model)
        return (total / count).item()

    @torch.no_grad()
    def _validate_per_modality(self) -> dict:
        """Validation loss over the token bits and over the cluster bits, separately
        (unconditional H-CoBit). The token-bit loss is comparable to CoBit's val loss.
        Runs at every epoch end, before the checkpoint, as in the paper's runs (its noise
        draws are part of the training run's random stream)."""
        cfg = self.cfg
        self.model.eval()
        self.ema.apply(self.model)
        ready = all(t is not None for t in self.schedule.tables.values())
        names = ("info_bit_loss", "cluster_bit_loss")
        sums = {k: torch.zeros((), device=self.device) for k in names}
        count = torch.zeros((), device=self.device)
        p_sc = float(getattr(cfg.train, "self_condition_prob", 0.5))
        for batch in self.val_loader:
            x0 = batch["x0"] if isinstance(batch, dict) else batch
            B = x0.size(0)
            x0_clean = x0.to(self.device, non_blocking=True).view(B, -1).to(torch.float32)
            S = x0_clean.size(1)
            model_kw = {}
            if ready:
                sigma_info, sigma_cluster = self.schedule.draw(B)
                sigma_pos = sigma_info.view(-1, 1) * self.m_info + sigma_cluster.view(-1, 1) * self.m_cluster
                sigma, sigma_weight = sigma_info, sigma_pos
                model_kw = dict(sigma_pos=sigma_pos if self.per_position_cin else None,
                                sigma_pos_mf=sigma_pos if self.per_position_mf else None)
                xt = x0_clean + sigma_pos * torch.randn_like(x0_clean)
            else:
                sigma = self.schedule._base_sigma(B)
                sigma_weight = sigma
                xt = x0_clean + sigma.view(-1, 1) * torch.randn_like(x0_clean)
            x0_hat = None
            if bool(getattr(cfg.model, "self_condition", False)) and p_sc > 0.0:
                sc_mask = torch.rand(B, device=self.device) < p_sc
                if sc_mask.any():
                    x0_hat = torch.zeros_like(xt)
                    with autocast(self.device.type, enabled=self.amp_enabled, dtype=self.amp_dtype):
                        logits_sc = _model_logits_continuous(self.model, cfg, xt, sigma, None, **model_kw)
                    est = torch.sigmoid(logits_sc.float()).detach().to(xt.dtype)
                    if sc_mask.all():
                        x0_hat.copy_(est)
                    else:
                        x0_hat[sc_mask] = est[sc_mask]
            with autocast(self.device.type, enabled=self.amp_enabled, dtype=self.amp_dtype):
                logits = _model_logits_continuous(self.model, cfg, xt, sigma, x0_hat, **model_kw)
                for k, m in zip(names, (self.m_info, self.m_cluster)):
                    loss = binary_score_interpolation_loss(logits, x0_clean, sigma_weight, cfg, mask=m.expand(B, S))
                    sums[k] += loss.detach().float()
            count += 1.0
        if self.ddp_active:
            for k in names:
                dist.all_reduce(sums[k])
            dist.all_reduce(count)
        self.ema.restore(self.model)
        return {k: (sums[k] / count.clamp_min(1.0)).item() for k in names}

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoints
    # ──────────────────────────────────────────────────────────────────────
    def _rng_state(self):
        state = {"py_random": random.getstate(), "np_random": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _state(self, epoch: int) -> dict:
        ema = self.ema.state_dict()
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": _unwrap_all(self.model).state_dict(),
            "ema": {"decay": float(ema["decay"]), "shadow": {k: v.detach().cpu() for k, v in ema["shadow"].items()}},
            "opt": self.opt.state_dict(),
            "lr_sched": self.lr_sched.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_scaler else None,
            "rng_state": self._rng_state(),
            "best_metric": self.best_metric,
            "best_ckpts": self.best_ckpts,
        }

    def _write(self, state: dict, name: str):
        tmp = self.ckpt_dir / f"{name}.tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_dir / f"{name}.pt")

    def _resume(self) -> int:
        path = self.ckpt_dir / "last.pt"
        if not path.exists():
            if self.is_master:
                print("Starting training from scratch.")
            return 0
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = {k.replace("_orig_mod.", "").removeprefix("module."): v for k, v in ckpt["model"].items()}
        _unwrap_all(self.model).load_state_dict(state, strict=True)
        self.opt.load_state_dict(ckpt["opt"])
        self.lr_sched.load_state_dict(ckpt["lr_sched"])
        if self.use_scaler and ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.ema.load_state_dict(ckpt["ema"])
        self.ema.to(self.device)
        rs = ckpt.get("rng_state")
        if rs is not None:
            random.setstate(rs["py_random"])
            np.random.set_state(rs["np_random"])
            torch.set_rng_state(rs["torch_cpu"])
            if torch.cuda.is_available() and "torch_cuda" in rs:
                torch.cuda.set_rng_state_all(rs["torch_cuda"])
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_metric = ckpt.get("best_metric", self.best_metric)
        self.best_ckpts = ckpt.get("best_ckpts", self.best_ckpts)
        if self.is_master:
            print(f"Resumed from {path} at step {self.global_step}")
        return int(ckpt.get("epoch", -1)) + 1

    def _maybe_step_checkpoints(self, epoch: int):
        if not self.is_master:
            return
        if self._next_resume is not None and self.global_step >= self._next_resume:
            self._write(self._state(epoch), "last")  # rolling resume point
            while self._next_resume <= self.global_step:
                self._next_resume += self.resume_every
        if self._next_interval is not None and self.global_step >= self._next_interval:
            torch.save(self._state(epoch), self.ckpt_dir / f"step={self.global_step:09d}.pt")
            while self._next_interval <= self.global_step:
                self._next_interval += self.interval_every

    def _epoch_checkpoints(self, epoch: int, val_loss: float):
        if not self.is_master:
            return
        new_best = self.save_top_k > 0 and val_loss < self.best_metric
        if new_best:
            self.best_metric = float(val_loss)
            name = f"epoch={epoch:04d}-val={val_loss:.4f}"
            self.best_ckpts.append({"path": f"{name}.pt", "metric": float(val_loss), "epoch": int(epoch)})
            self.best_ckpts.sort(key=lambda d: d["metric"])
            while len(self.best_ckpts) > self.save_top_k:
                (self.ckpt_dir / self.best_ckpts.pop(-1)["path"]).unlink(missing_ok=True)
        state = self._state(epoch)
        self._write(state, "last")
        if new_best:
            torch.save(state, self.ckpt_dir / f"{name}.pt")
            torch.save(state, self.ckpt_dir / "best.pt")

    # ──────────────────────────────────────────────────────────────────────
    # Loop
    # ──────────────────────────────────────────────────────────────────────
    def train(self):
        if self.is_master:
            self._estimate_sigma_data()
        target = int(getattr(self.cfg.optim, "total_steps", 0))
        # STOP_AT_STEP (train.py --stop_at_step) halts early WITHOUT changing the LR horizon:
        # the paper's 600k / 750k checkpoints come from 1M-step cosine schedules.
        stop_at = int(os.environ.get("STOP_AT_STEP", "0") or 0)
        if stop_at > 0:
            target = stop_at if target <= 0 else min(target, stop_at)
        refit_every = int(getattr(self.cfg.train, "entropy_update_every_steps", 2000))
        cond_task = bool(getattr(self.cfg.cond, "enabled", False))
        log_every = int(getattr(self.cfg.logging, "log_freq", 10))

        for epoch in range(self.start_epoch, int(self.cfg.train.epochs)):
            if self.ddp_active:
                self.train_loader.sampler.set_epoch(epoch)
            self.model.train()
            total, n_batches, stop = 0.0, 0, False
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", disable=not self.is_master)
            for batch in pbar:
                if target > 0 and self.global_step >= target:
                    stop = True
                    break
                loss = self._step(batch, is_train=True)
                self.global_step += 1
                total += loss
                n_batches += 1
                if refit_every > 0 and self.global_step % refit_every == 0 and bool(self.cfg.train.entropy_compute):
                    self.schedule.refit()
                self._maybe_step_checkpoints(epoch)
                if self.is_master:
                    pbar.set_postfix(loss=f"{loss:.4f}")
                    if self.use_wandb and self.global_step % log_every == 0:
                        wandb.log({"loss/train": loss, "lr": self.opt.param_groups[0]["lr"]}, step=self.global_step)
                if target > 0 and self.global_step >= target:
                    stop = True
                    break
            if n_batches == 0:
                break
            val_loss = self._validate()
            logs = {"loss/epoch_train": total / n_batches, "loss/epoch_val": val_loss}
            if self.dual and not cond_task:
                logs.update(self._validate_per_modality())
            if self.is_master:
                print(f"Epoch {epoch + 1}: " + ", ".join(f"{k} {v:.4f}" for k, v in logs.items()))
                if self.use_wandb:
                    wandb.log(logs, step=self.global_step)
            self._epoch_checkpoints(epoch, val_loss)
            if self.ddp_active:
                dist.barrier()
            if stop:
                break
        if self.use_wandb:
            wandb.finish()
