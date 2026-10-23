"""TinyGSM, 512 SmolLM-135M tokens (|V| = 49,153 -> 16 bits per token), evaluated on GSM8K.

    configs/tinygsm.py:hcobit   H-CoBit, n = 16 clusters, lambda_2 = 0.3 (20 bits per token)
    configs/tinygsm.py:cobit    CoBit-S (16 bits per token)

Conditional on the question: the prompt is kept noise-free and the loss is applied to the
answer only. CoBit's TinyGSM recipe: constant learning rate, beta_2 = 0.999, no weight
decay, 250k steps.
"""
from configs.base import DATA_ROOT, cluster_bits, get_base_config, set_bits

INFO_BITS = 16


def get_config(model="hcobit"):
    cfg = get_base_config()
    cfg.experiment = f"tinygsm_{model}"

    d = cfg.data
    d.dataset = "tinygsm"
    d.root = f"{DATA_ROOT}/tinygsm"
    d.tokenizer_name = "HuggingFaceTB/SmolLM-135M"
    d.token_vocab_size = 49153
    d.sequence_len_tokens = 512
    d.separator = "\n"
    d.max_train_examples = None
    d.val_ratio = 0.01
    d.val_seed = 42
    d.gsm8k_test_path = f"{DATA_ROOT}/gsm8k/gsm8k_test.json"
    d.num_workers = 8

    cfg.cond.enabled = True
    cfg.cond.p_uncond = 0.0
    cfg.cond.noise_prefix = False
    cfg.cond.loss_on_suffix_only = True

    cfg.optim.scheduler = "constant"
    cfg.optim.beta2 = 0.999
    cfg.optim.weight_decay = 0.0
    cfg.optim.total_steps = 250_000
    cfg.train.epochs = 100_000
    cfg.train.checkpointing.save_top_k = 1
    cfg.train.checkpointing.interval.every_steps = 25_000

    if model == "hcobit":
        cluster_bits(cfg, 16, "cluster_maps/tinygsm/k_16", INFO_BITS, loss_weight=0.3)
    elif model == "cobit":
        set_bits(cfg, INFO_BITS)
    else:
        raise ValueError(f"model must be 'hcobit' or 'cobit', got {model!r}")
    return cfg
