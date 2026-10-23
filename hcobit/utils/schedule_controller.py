"""Online information-uniform noise schedule (Sections 3.3 / 4.3, Appendix C).

CoBit learns one schedule over all bits; H-CoBit learns one per modality (token bits,
cluster bits). During training, every example's (sigma, squared error) pair is pushed
into a FIFO buffer per schedule. Every ``entropy_update_every_steps``
the buffers are binned over log sigma and turned into a density

    q(bin) ~ g(sigma; c, n) * mean_rate(bin)^p,   rate = error / sigma^2,
    g(sigma) = sigma^n / (sigma^n + c^n)

(p = 1/2 for 'sqrt-rate'). Training noise levels are drawn from the base log-normal
for the first ``entropy_warmup_steps``; afterwards a fraction gamma (ramped linearly to
``entropy_gamma_max`` over ``entropy_transition_steps``) is drawn from the learned
schedules; with two modalities, both from one shared base time u ~ U[0, 1] (coupled
sampling). In DDP, rank 0 fits the schedules and broadcasts them.

Tables are written to the run directory as entropy_{pdf,cdf,sigmas,edges}.pt (CoBit)
or entropy_{pdf,cdf,sigmas,edges}_{info,cluster}.pt (H-CoBit).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.distributed as dist

def _generalized_regularizer(sigmas: torch.Tensor, c: float, n: float) -> torch.Tensor:
    x = (sigmas.clamp_min(1e-12) / float(max(c, 1e-12))).pow(float(n))
    return x / (1.0 + x)


class EntropyScheduleController:
    def __init__(self, trainer, proc, dual: bool):
        self.trainer = trainer
        self.proc = proc
        self.groups = ("info", "cluster") if dual else ("all",)
        tr = trainer.cfg.train
        self.cap = int(tr.entropy_buffer_size)
        self.num_bins = max(2, int(tr.entropy_num_bins))
        self.min_per_bin = int(getattr(tr, "entropy_min_per_bin", 10))
        self.warmup_steps = int(tr.entropy_warmup_steps)
        self.transition_steps = int(tr.entropy_transition_steps)
        self.gamma_max = float(tr.entropy_gamma_max)
        self.c = float(getattr(tr, "entropy_regularizer_c", 0.1))
        self.n = float(getattr(tr, "entropy_regularizer_n", 3.0))
        target = str(getattr(tr, "entropy_target", "rate")).lower()
        self.power = 0.5 if target in ("sqrt", "sqrt-rate", "sqrt_rate") else 1.0 if target == "rate" else float(target)
        self.buf = {g: {"sig": torch.empty(self.cap), "met": torch.empty(self.cap), "ptr": 0, "len": 0} for g in self.groups}
        self.tables = {g: None for g in self.groups}  # {"pdf", "cdf", "sigmas", "edges"} on device

    def _path(self, name: str, group: str) -> Path:
        suffix = "" if group == "all" else f"_{group}"
        return Path(self.trainer.run_dir) / f"entropy_{name}{suffix}.pt"

    # -- buffers -----------------------------------------------------------------------
    def push(self, group: str, sigmas: torch.Tensor, metric: torch.Tensor) -> None:
        s = sigmas.detach().flatten().to(dtype=torch.float32, device="cpu")
        m = metric.detach().flatten().to(dtype=torch.float32, device="cpu")
        n = s.numel()
        if n == 0:
            return
        b = self.buf[group]
        if n >= self.cap:
            s, m, n = s[-self.cap:], m[-self.cap:], self.cap
        ptr, end = b["ptr"], b["ptr"] + n
        if end <= self.cap:
            b["sig"][ptr:end], b["met"][ptr:end] = s, m
        else:
            k = self.cap - ptr
            b["sig"][ptr:], b["met"][ptr:] = s[:k], m[:k]
            b["sig"][: n - k], b["met"][: n - k] = s[k:], m[k:]
        b["ptr"] = (ptr + n) % self.cap
        b["len"] = min(self.cap, b["len"] + n)

    # -- fitting -----------------------------------------------------------------------
    def _profile_from_pairs(self, sig, metr):
        """(pdf, cdf, bin midpoints, bin edges) over log sigma; uniform if degenerate."""
        sigma_min = max(float(self.proc.sigma_min), 1e-12)
        sigma_max = max(float(self.proc.sigma_max), sigma_min * 1.0000001)
        K = self.num_bins
        edges_log = torch.linspace(math.log(sigma_min), math.log(sigma_max), K + 1, dtype=torch.float32)
        mids = (0.5 * (edges_log[:-1] + edges_log[1:])).exp().clamp(min=1e-12)
        edges_sigma = edges_log.exp().clamp(min=1e-12)
        sig, metr = sig.float().flatten(), metr.float().flatten()
        finite = torch.isfinite(sig) & torch.isfinite(metr)
        pdf = torch.full((K,), 1.0 / K, dtype=torch.float32)
        if finite.any():
            sig = sig[finite].clamp(min=sigma_min, max=sigma_max)
            metr = metr[finite]
            idx = (torch.bucketize(sig.log(), edges_log, right=False) - 1).clamp_(0, K - 1)
            rate = torch.clamp(metr / (sig * sig + 1e-12), min=0.0)
            sum_rate = torch.zeros(K).scatter_add_(0, idx, rate)
            count = torch.zeros(K).scatter_add_(0, idx, torch.ones_like(rate))
            mean_rate = torch.clamp(sum_rate / (count + 1e-8), min=0.0)
            q = _generalized_regularizer(mids, self.c, self.n) * (mean_rate.pow(self.power) if self.power != 0 else 1.0)
            if torch.isfinite(q).all() and float(q.sum()) > 0.0:
                pdf = q / q.sum()
        cdf = torch.cumsum(pdf, dim=0)
        cdf = torch.maximum(cdf, torch.cummax(cdf, dim=0)[0])
        cdf[-1] = 1.0
        return pdf, cdf, mids, edges_sigma

    @torch.no_grad()
    def refit(self) -> None:
        """Refit the schedules from the buffers (same order on every rank)."""
        ddp = dist.is_available() and dist.is_initialized() and self.trainer.world_size > 1
        dev, K = self.trainer.device, self.num_bins
        for g in self.groups:
            enough = self.buf[g]["len"] >= K * self.min_per_bin
            if ddp:
                flag = torch.tensor(int(enough) if self.trainer.is_master else 0, device=dev, dtype=torch.int32)
                dist.broadcast(flag, src=0)
                enough = bool(flag.item())
            if not enough:
                continue
            if ddp and not self.trainer.is_master:
                t = [torch.empty(K + (i == 3), device=dev) for i in range(4)]
            else:
                b = self.buf[g]
                n = min(b["len"], self.cap)
                t = [x.to(dev, dtype=torch.float32) for x in self._profile_from_pairs(b["sig"][:n], b["met"][:n])]
            if ddp:
                for x in t:
                    dist.broadcast(x, src=0)
            self.tables[g] = dict(zip(("pdf", "cdf", "sigmas", "edges"), t))
            if self.trainer.is_master:
                for name, x in self.tables[g].items():
                    torch.save(x.detach().cpu(), self._path(name, g))

    def load_tables_if_any(self) -> None:
        """Restore the schedules saved in the run directory (on resume)."""
        dev = self.trainer.device
        for g in self.groups:
            paths = {k: self._path(k, g) for k in ("pdf", "cdf", "sigmas", "edges")}
            if all(paths[k].exists() for k in ("pdf", "cdf", "sigmas")):
                self.tables[g] = {k: torch.load(p, map_location=dev) for k, p in paths.items() if p.exists()}

    # -- sampling ----------------------------------------------------------------------
    def gamma(self) -> float:
        step = int(self.trainer.global_step)
        if step < self.warmup_steps:
            return 0.0
        frac = min(float(step - self.warmup_steps) / max(1, self.transition_steps), 1.0)
        return max(0.0, min(self.gamma_max * frac, 1.0))

    def _base_sigma(self, bsz: int) -> torch.Tensor:
        strat = str(getattr(self.trainer.cfg.train, "sigma_sampling_strategy", "log-uniform")).lower()
        c = self.trainer.cfg.diffusion.continuous
        if strat == "log-normal":
            s = self.proc.sample_sigma(bsz, strategy="log-normal", p_mean=c.p_mean, p_std=c.p_std)
        else:
            s = self.proc.sample_sigma(bsz, strategy="log-uniform")
        return s.to(self.trainer.device)

    def _from_schedules(self, bsz: int):
        """One shared base time u (and in-bin position u2) through each inverse CDF."""
        dev = self.trainer.device
        u = torch.rand(bsz, device=dev)
        u2 = torch.rand(bsz, device=dev)
        smin, smax = float(self.proc.sigma_min), float(self.proc.sigma_max)
        out = []
        for g in self.groups:
            cdf = self.tables[g]["cdf"].to(dev).float().clone().clamp(min=0.0)
            cdf[-1] = 1.0
            idx = torch.clamp(torch.searchsorted(cdf, u, right=False), 0, cdf.numel() - 1)
            edges = self.tables[g]["edges"].to(dev).float().clamp(min=1e-12)
            lo, hi = edges[idx].clamp(min=smin), edges[idx + 1].clamp(max=smax)
            out.append((lo.log() + u2 * (hi.log() - lo.log())).exp().clamp(min=smin, max=smax))
        return out

    def draw(self, bsz: int):
        """One sigma[B] per schedule for a training batch: [sigma] (CoBit) or
        [sigma_token, sigma_cluster] (H-CoBit)."""
        base = self._base_sigma(bsz)
        if any(self.tables[g] is None for g in self.groups):
            return [base for _ in self.groups]
        gamma = self.gamma()
        if gamma <= 0.0:
            return [base for _ in self.groups]
        use = torch.rand(bsz, device=self.trainer.device) < gamma
        out = [base.clone() for _ in self.groups]
        n = int(use.sum().item())
        if n > 0:
            for s, e in zip(out, self._from_schedules(n)):
                s[use] = e.to(s.dtype)
        return out
