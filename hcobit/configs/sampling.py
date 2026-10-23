"""Samplers and sweeps behind the paper's evaluation (Table 1, Figure 4, Table 11).

Every sampler is CoBit's DDIM probability-flow ODE with EDM churn and self-conditioning,
on an information-uniform noise grid, stopping at sigma_min = 0.08. Churn gamma is per
step (S_churn = gamma * (NFE - 1)).

Arms
  cobit              CoBit-S, single schedule, s_noise = 1.003
  cobit_snoise1.05   CoBit-S, s_noise = 1.05
  hcobit             H-CoBit, per-modality schedule, shared churn
  hcobit_pms         H-CoBit + per-modality sampler: cluster churn gamma_2 confined to the
                     late window W_late = [0, 0.5] of the schedule
                     (LM1B gamma_2 = 0.4; OWT gamma_2 = 0.3 with global s_noise = 1.05)

Grids
  frontier   the (NFE, gamma) sweep of Table 1 / Figure 4
  mauve      the evenly spaced sweep of Table 11 (MAUVE)
"""
from pathlib import Path

from ml_collections import config_dict

ROOT_2_MINUS_1 = 0.41421356  # sqrt(2) - 1, the sampler's per-step churn ceiling

ARMS = {
    "lm1b": {
        "cobit": dict(model="cobit", s_noise=1.003),
        "cobit_snoise1.05": dict(model="cobit", s_noise=1.05),
        "hcobit": dict(model="hcobit", s_noise=1.003),
        "hcobit_pms": dict(model="hcobit", s_noise=1.003, gamma_cluster=0.4, cluster_window=(0.0, 0.5)),
    },
    "owt": {
        "cobit": dict(model="cobit", s_noise=1.003),
        "cobit_snoise1.05": dict(model="cobit", s_noise=1.05),
        "hcobit": dict(model="hcobit", s_noise=1.003),
        "hcobit_pms": dict(model="hcobit", s_noise=1.05, gamma_cluster=0.3, cluster_window=(0.0, 0.5)),
    },
}

_LM1B_128 = [0.05, 0.06, 0.075, 0.08, 0.10, 0.125, 0.13, 0.15, 0.17, 0.175, 0.185, 0.20, 0.225, 0.25, 0.275, 0.30]
_LM1B_256 = [0.05, 0.075, 0.10, 0.125, 0.13, 0.15, 0.17, 0.175, 0.185, 0.20, 0.215, 0.225, 0.25, 0.275, 0.30]
_OWT = [0.05, 0.10, 0.13, 0.15, 0.18, 0.21, 0.25, 0.30, 0.35, 0.40, ROOT_2_MINUS_1]
_MAUVE = [0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.23, 0.26, 0.29, 0.32, 0.35, 0.38, ROOT_2_MINUS_1]

FRONTIER = {
    "lm1b": {
        64: [0.04, 0.06, 0.08, 0.10, 0.13, 0.15, 0.17],
        128: _LM1B_128,
        256: _LM1B_256,
        512: _LM1B_256,
        1024: [0.13, 0.15, 0.17, 0.185, 0.20, 0.215],
    },
    "owt": {64: _OWT, 128: _OWT, 256: _OWT, 512: sorted(_OWT + [0.26])},
}
#: H-CoBit's LM1B sweep went further down at NFE 1024, where it had not reached data entropy.
FRONTIER_EXTRA = {("lm1b", "hcobit", 1024): [0.05, 0.07, 0.09, 0.11]}

MAUVE_NFES = {"lm1b": [64, 128, 256, 512, 1024], "owt": [64, 128, 256, 512]}
MAUVE_KMEANS_SEEDS = [25, 0, 1, 2, 3, 4]  # mauve-text's default seed 25, plus five
#: Generation chunk size. Each chunk of samples is drawn with its own seed (a function of
#: its global offset), so this fixes the samples. These are the chunks of the paper runs:
#: most LM1B arms ran on 4 GPUs at micro-batch 512, i.e. 256 samples per GPU and chunk;
#: CoBit-S with s_noise = 1.05 ran on 1 GPU, in chunks of 512. Samples do not depend on the
#: number of GPUs as long as each GPU's share (1024 / GPUs) is a multiple of the chunk.
MICRO_BATCH = {"lm1b": 256, "owt": 128}
MICRO_BATCH_ARM = {("lm1b", "cobit_snoise1.05"): 512}
#: GPT-2 Large featurization batch for MAUVE, as in the paper runs.
MAUVE_BATCH = {"lm1b": 64, "owt": 16}


def grid(dataset, arm, name):
    """[(nfe, gamma), ...] for one arm."""
    if name == "frontier":
        base = arm.replace("_pms", "")  # the per-modality sampler re-sweeps its arm's grid
        out = []
        for nfe, gammas in FRONTIER[dataset].items():
            gammas = sorted(gammas + FRONTIER_EXTRA.get((dataset, base, nfe), []))
            out += [(nfe, g) for g in gammas]
        return out
    if name == "mauve":
        return [(nfe, g) for nfe in MAUVE_NFES[dataset] for g in _MAUVE]
    raise ValueError(f"unknown grid {name!r}")


def make_spec(dataset, arm, nfe, gamma):
    a = ARMS[dataset][arm]
    n = max(1, nfe - 1)
    s = config_dict.ConfigDict()
    s.sampler_name = "ddim_entropic_dual" if a["model"] == "hcobit" else "ddim_entropic"
    s.sc_refresh_modes = ["carry"]
    s.target_nfes = [int(nfe)]
    s.ati_etas = [0.0]
    s.stochastic_enabled = True
    s.gamma_target = float(gamma)
    s.s_churn = float(gamma) * n
    s.s_noise = float(a["s_noise"])
    s.window_mode = "entropy_cdf"
    s.entropy_quantile_lo = 0.0
    s.entropy_quantile_hi = 1.0
    s.entropy_fallback = "deterministic"
    s.s_tmin = None
    s.s_tmax = None
    if "gamma_cluster" in a:
        s.s_churn_cluster = float(a["gamma_cluster"]) * n
        s.entropy_quantile_lo_cluster, s.entropy_quantile_hi_cluster = map(float, a["cluster_window"])
    return s


def eval_config(cfg, *, dataset, arm, checkpoint, out_dir, points):
    """Add the cfg.evaluation block for one arm and a list of (nfe, gamma) points.

    The noise schedule is read from the checkpoint's directory: the two per-modality
    profiles entropy_profile_{info,cluster}.pt for H-CoBit, the schedule learned in
    training, entropy_{pdf,cdf,sigmas}.pt, for CoBit.
    """
    a = ARMS[dataset][arm]
    model_dir = str(Path(checkpoint).parent)
    ev = config_dict.ConfigDict()
    ev.checkpoint_path = str(checkpoint)
    ev.out_dir = str(out_dir)
    ev.samples_dir = f"{out_dir}/samples"
    ev.results_csv = f"{out_dir}/results.csv"
    ev.shared_text_cache_dir = f"{out_dir}/text_cache"
    ev.entropy_run_dir = model_dir
    ev.use_amp = True
    ev.amp_dtype = "bf16"
    ev.use_compile = True
    ev.compile_mode = "default"
    if a["model"] == "hcobit":
        ev.dual_schedule = config_dict.ConfigDict(dict(
            enabled=True,
            info_profile=f"{model_dir}/entropy_profile_info.pt",
            cluster_profile=f"{model_dir}/entropy_profile_cluster.pt",
            per_position_cin=True,
            per_position_matched_filter=True,
        ))
    ev.sampling_sweep = config_dict.ConfigDict()
    ev.sampling_sweep.specs = [make_spec(dataset, arm, nfe, g) for nfe, g in points]

    common = dict(
        num_samples=1024,
        micro_batch_size=MICRO_BATCH_ARM.get((dataset, arm), MICRO_BATCH[dataset]),
        terminal_sigmas=[0.08],
        sigma_max=None,
        seed=42,
        shared_cache_dir=ev.shared_text_cache_dir,
    )
    ev.external_ppl = config_dict.ConfigDict(dict(
        hf_model_name="openai-community/gpt2-large",
        hf_dtype="bfloat16",
        attn_implementation="sdpa",
        compute_real_reference=True,
        **common,
    ))
    ev.mauve = config_dict.ConfigDict(dict(
        featurizer_name="gpt2-large",
        batch_size=MAUVE_BATCH[dataset],
        max_tokens=1024,
        num_buckets="auto",
        kmeans_seeds=MAUVE_KMEANS_SEEDS,
        **common,
    ))
    ev.sigma_loss_profile = config_dict.ConfigDict(dict(
        sigma_sampling="log-uniform",
        mc_samples=4,
        entropy_num_bins=256,
        entropy_regularizer_c=0.1,
        entropy_regularizer_n=3.0,
        entropy_target="rate",
    ))
    cfg.evaluation = ev
    cfg.train.seed = 42
    return cfg
