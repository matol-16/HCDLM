"""GSM8K accuracy (Table 2), following S-FLM's protocol.

For each of the 1319 test problems: condition on the clean prompt ([BOS] question \\n),
sample the 512-token answer, decode it with the SmolLM tokenizer, run the generated
Python program in a restricted sandbox and compare its result with the gold answer.

Sampling uses the noise schedule the model was trained with, read next to the
checkpoint (CoBit: entropy_{pdf,cdf,sigmas}.pt; H-CoBit: the per-modality
entropy_{cdf,sigmas}_{info,cluster}.pt), churn gamma * (NFE - 1) over the whole
schedule, and the data scale sigma_data measured at training time (sigma_data.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data.task_codec import bits_to_token_ids
from data.tinygsm import GSM8KTestDataset
from diffusion.continuous.processes import ContinuousForwardProcess
from diffusion.continuous.samplers import DDIMSampler
from evaluation.gsm8k_sandbox import evaluate_samples
from evaluation.utils import load_checkpoint
from models import create_model
from utils.ema import EMA


def _bootstrap_ci(correct: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = np.array([correct[rng.integers(0, len(correct), size=len(correct))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(means.mean()), float(lo), float(hi)


def _configure_churn(cfg, gamma: float, num_steps: int, dual: bool, s_noise: float = 1.003) -> None:
    from ml_collections import config_dict

    st = config_dict.ConfigDict()
    st.enabled = gamma > 0.0
    st.s_churn = float(gamma) * max(1, int(num_steps) - 1) if gamma > 0.0 else 0.0
    st.s_noise = float(s_noise) if gamma > 0.0 else 1.0
    st.window_mode = "entropy_cdf" if gamma > 0.0 else "deterministic"
    st.entropy_quantile_lo = 0.0
    st.entropy_quantile_hi = 1.0
    st.entropy_fallback = "deterministic"
    st.s_tmin = None
    st.s_tmax = None
    st.s_churn_cluster = None
    st.entropy_quantile_lo_cluster = None
    st.entropy_quantile_hi_cluster = None
    if "evaluation" not in cfg:
        cfg.evaluation = config_dict.ConfigDict()
    cfg.evaluation.stochastic = st
    if dual:
        # Per-position c_in and matched filter, as in training (the profiles themselves
        # are the trained tables next to the checkpoint).
        cfg.evaluation.dual_schedule = config_dict.ConfigDict(
            dict(enabled=True, per_position_cin=True, per_position_matched_filter=True)
        )


@torch.no_grad()
def evaluate_gsm8k(cfg, checkpoint: Path, *, nfe: int, gamma: float, out_dir: Path,
                   batch_size: int = 64, limit=None, seed: int = 42, timeout_s: float = 5.0) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(checkpoint).parent
    sigma_data = float(json.loads((model_dir / "sigma_data.json").read_text())["sigma_data"])
    cfg.diffusion.continuous.sigma_data = sigma_data  # the value training measured

    model = create_model(cfg).to(device).eval()
    load_checkpoint(model, EMA(model, decay=0.0), checkpoint, device, apply_ema=True)
    sampler = DDIMSampler(model, ContinuousForwardProcess(cfg), cfg)
    dual = bool(getattr(cfg.data, "cluster_bits_enabled", False))
    _configure_churn(cfg, gamma, nfe, dual)
    use_amp, amp_dtype = device.type == "cuda", torch.bfloat16

    ds = GSM8KTestDataset(cfg)
    tok, bpt, info_bits = ds.tok, ds.bits_per_token, int(ds.info_bits)
    tok_len = len(tok)  # len() of a fast tokenizer recomputes the vocab size on every call
    n = len(ds) if limit is None else min(int(limit), len(ds))
    per_correct, records, n_invalid, n_gen = [], [], 0, 0
    for start in range(0, n, batch_size):
        idxs = list(range(start, min(start + batch_size, n)))
        x0 = torch.stack([ds[i]["x0"] for i in idxs]).float().to(device)
        pm = torch.stack([ds[i]["prefix_mask"] for i in idxs]).to(device)
        plens = [int(ds[i]["prompt_len_tokens"]) for i in idxs]

        torch.manual_seed(int(seed))  # every batch restarts from the same seed, as in CoBit's GSM8K evaluation
        torch.cuda.manual_seed_all(int(seed))
        with torch.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
            _, probs = sampler.sample(
                num_samples=len(idxs), seq_len=x0.shape[1], schedule="entropic_dual" if dual else "entropic",
                num_steps=nfe,
                entropy_run_dir=str(model_dir), conditioning_prefix_full=x0, cond_prefix_mask=pm,
                return_probs=True, progress=False,
            )
        bits = (probs.float() >= 0.5).long()
        bits = bits.view(bits.shape[0], -1, bpt)[..., :info_bits].reshape(bits.shape[0], -1)  # drop any cluster bits
        gen_ids = bits_to_token_ids(bits, info_bits)

        for b, gi in enumerate(idxs):
            suffix = gen_ids[b].tolist()[plens[b]:]
            n_gen += len(suffix)
            n_invalid += sum(1 for t in suffix if t >= tok_len)
            text = tok.decode([t if 0 <= t < tok_len else tok.eos_token_id for t in suffix], skip_special_tokens=True)
            rec = ds[gi]
            ok = evaluate_samples(text, rec["response_ground_truth"], timeout_s)
            per_correct.append(1 if ok else 0)
            if len(records) < 100:
                records.append({"idx": gi, "prompt": rec["prompt"][:200], "response": text[:400], "correct": bool(ok)})
        print(f"[gsm8k] {len(per_correct)}/{n}  acc={100.0 * np.mean(per_correct):.2f}%", flush=True)

    correct = np.asarray(per_correct, dtype=np.float64)
    boot, lo, hi = _bootstrap_ci(correct, seed=seed)
    result = {
        "checkpoint": str(checkpoint), "nfe": int(nfe), "gamma": float(gamma), "s_noise": 1.003 if gamma > 0 else None,
        "sigma_data": sigma_data, "num_examples": int(n), "num_correct": int(correct.sum()),
        "accuracy": float(correct.mean()) if n else 0.0, "bootstrap_accuracy": boot, "ci95_low": lo, "ci95_high": hi,
        "invalid_token_rate": n_invalid / max(1, n_gen), "sample_records": records,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gsm8k_nfe{nfe}_gamma{gamma:g}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"GSM8K accuracy {100 * result['accuracy']:.2f}% ({result['num_correct']}/{n}) -> {path}")
    return result
