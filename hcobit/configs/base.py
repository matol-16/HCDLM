"""Settings shared by every CoBit / H-CoBit model in the paper.

Values are those of CoBit-S (Batzolis et al., 2026); H-CoBit changes none of them. The
dataset files (lm1b.py, owt.py, tinygsm.py) set the data, the bit width and the
optimisation horizon, and add the cluster modality.
"""
import os

from ml_collections import config_dict

DATA_ROOT = os.environ.get("HCDLM_DATA", "datasets")


def cluster_bits(cfg, n_clusters, map_dir, info_bits, loss_weight=None):
    """Append ceil(log2 n) cluster bits to each token's bit patch (Section 4.2).

    loss_weight is lambda_2 of equation (5). None keeps the plain loss over all bits, i.e.
    lambda_2 = (cluster bits) / (token bits).
    """
    n_bits = (n_clusters - 1).bit_length()
    cfg.data.cluster_bits_enabled = True
    cfg.data.cluster_bits_n_clusters = n_clusters
    cfg.data.cluster_map_dir = map_dir
    set_bits(cfg, info_bits + n_bits)
    # Per-modality information-uniform schedule (Section 4.3).
    cfg.train.entropy_dual_enabled = True
    cfg.train.entropy_dual_per_position_cin = True
    cfg.train.entropy_dual_per_position_matched_filter = True
    if loss_weight is not None:
        cfg.train.cluster_loss_weight_enabled = True
        cfg.train.cluster_loss_weight_alpha = loss_weight


def set_bits(cfg, bits_per_token):
    cfg.data.bits_per_token = bits_per_token
    cfg.data.sequence_len = cfg.data.sequence_len_tokens * bits_per_token
    cfg.model.patch_size = bits_per_token  # one patch per token


def get_base_config():
    cfg = config_dict.ConfigDict()
    cfg.framework = "continuous_score"
    cfg.device = "cuda"

    cfg.data = config_dict.ConfigDict()
    cfg.data.representation = "binary"
    cfg.data.binarization = "raw_binary"
    cfg.data.token_space = "tokenizer_id"
    cfg.data.vocab_size = 2
    cfg.data.channels = 1
    cfg.data.flatten_order = "flatten"
    cfg.data.num_workers = 12
    cfg.data.prefetch_factor = 4
    cfg.data.pin_memory = True

    # Unconditional generation; tinygsm.py overrides this for GSM8K.
    cfg.cond = config_dict.ConfigDict()
    cfg.cond.enabled = False
    cfg.cond.sample_prompt_len = False
    cfg.cond.cond_len_tokens = 0
    cfg.cond.cond_len_chars = 0
    cfg.cond.p_uncond = 1.0
    cfg.cond.noise_prefix = True
    cfg.cond.loss_on_suffix_only = False
    cfg.cond.null_strategy = "half"

    # 12-layer, 12-head Diffusion Transformer, hidden size 768 (Appendix D).
    cfg.model = config_dict.ConfigDict()
    cfg.model.name = "sdt"
    cfg.model.use_flash_attn = True
    cfg.model.self_condition = True
    cfg.model.center_inputs = True
    cfg.model.embed_dim = 768
    cfg.model.dim_ff = 3072
    cfg.model.n_blocks = 12
    cfg.model.n_heads = 12
    cfg.model.dropout = 0.1
    cfg.model.use_swiglu = True
    cfg.model.use_adaln = True
    cfg.model.use_rope_trunk = True
    cfg.model.rope_base = 10_000.0
    cfg.model.abs_pos_mode = "local_only"
    cfg.model.n_fourier_global = 32
    cfg.model.n_fourier_local = 4
    cfg.model.n_pos_features = 1
    cfg.model.rpb_max_distance = 1
    cfg.model.scale_by_sigma = False
    cfg.model.head_type = "optimal_skip_mlp"
    cfg.model.head_variant = "single"
    cfg.model.head_hidden = 128
    cfg.model.head_embed_dim = 64
    cfg.model.head_use_cross_attn = True
    cfg.model.head_use_local_mixer = True
    cfg.model.head_use_self_attn = False
    cfg.model.head_kernel = 3
    cfg.model.head_dilation = 1
    cfg.model.out_dim = 1
    cfg.model.content_dim_discrete = 64
    cfg.model.content_dim_continuous = 64
    # CoBit's matched-filter residual output head.
    cfg.model.continuous_logit_scaling = "matched_filter_residual"
    cfg.model.matched_filter_center = 0.5
    cfg.model.matched_filter_scale = 1.0
    cfg.model.matched_filter_clip = 30.0

    # EDM / variance-exploding noise, sigma_data = 1/2.
    cfg.diffusion = config_dict.ConfigDict()
    cfg.diffusion.continuous = config_dict.ConfigDict()
    cfg.diffusion.continuous.sigma_min = 0.002
    cfg.diffusion.continuous.sigma_max = 80.0
    cfg.diffusion.continuous.rho = 7.0
    cfg.diffusion.continuous.sigma_data = 0.5
    cfg.diffusion.continuous.data_center = 0.5
    cfg.diffusion.continuous.p_mean = -1.2
    cfg.diffusion.continuous.p_std = 1.2

    cfg.train = config_dict.ConfigDict()
    cfg.train.seed = 42
    cfg.train.deterministic = False
    cfg.train.batch_size = 512  # global; divided across GPUs, no gradient accumulation
    cfg.train.use_compile = True
    cfg.train.compile_mode = "default"
    cfg.train.use_fp16 = True
    cfg.train.amp_dtype = "bf16"
    cfg.train.allow_tf32 = True
    cfg.train.loss_type = "binary_sm"
    cfg.train.loss_weighting = "edm"
    cfg.train.ema_decay = 0.9999
    cfg.train.sigma_sampling_strategy = "log-normal"
    cfg.train.self_condition_prob = 0.5
    # Online entropy-rate schedule: 40k-step warmup, refit every 2k steps from a 128-bin
    # loss histogram, each refit blended in over 10k steps.
    cfg.train.entropy_compute = True
    cfg.train.entropy_use_for_sampling = True
    cfg.train.entropy_offline = config_dict.ConfigDict()
    cfg.train.entropy_offline.enabled = False
    cfg.train.entropy_mode = "regularized"
    cfg.train.entropy_target = "sqrt-rate"
    cfg.train.entropy_regularizer_c = 0.1
    cfg.train.entropy_regularizer_n = 3.0
    cfg.train.entropy_num_bins = 128
    cfg.train.entropy_min_per_bin = 100
    cfg.train.entropy_buffer_size = 800_000
    cfg.train.entropy_warmup_steps = 40_000
    cfg.train.entropy_update_every_steps = 2_000
    cfg.train.entropy_transition_steps = 10_000
    cfg.train.entropy_gamma_max = 1.0
    cfg.train.entropy_plot_every_k_epochs = 5
    cfg.train.sanity = config_dict.ConfigDict()
    cfg.train.sanity.enabled = False
    cfg.train.sanity.run_epoch = -1
    cfg.train.checkpointing = config_dict.ConfigDict()
    cfg.train.checkpointing.save_last = True
    cfg.train.checkpointing.save_top_k = 2
    cfg.train.checkpointing.mode = "min"
    cfg.train.checkpointing.interval = config_dict.ConfigDict()
    cfg.train.checkpointing.interval.enabled = True
    cfg.train.checkpointing.interval.every_steps = 50_000
    cfg.train.checkpointing.interval.keep_last = 0
    cfg.train.checkpointing.resume_interval = config_dict.ConfigDict()
    cfg.train.checkpointing.resume_interval.enabled = True
    cfg.train.checkpointing.resume_interval.every_steps = 5_000

    cfg.optim = config_dict.ConfigDict()
    cfg.optim.optimizer = "AdamW"
    cfg.optim.lr = 3e-4
    cfg.optim.beta1 = 0.9
    cfg.optim.beta2 = 0.99
    cfg.optim.eps = 1e-8
    cfg.optim.weight_decay = 0.01
    cfg.optim.grad_clip = 1.0
    cfg.optim.warmup = 2_500
    cfg.optim.scheduler = "cosine_decay"
    cfg.optim.total_steps = 1_000_000  # LR horizon; stop earlier with --stop_at_step

    cfg.logging = config_dict.ConfigDict()
    cfg.logging.use_wandb = False
    cfg.logging.project = "HCDLM"
    cfg.logging.mode = "online"
    cfg.logging.log_freq = 10
    cfg.logging.run_id = "auto"
    return cfg
