"""LM1B, 128 BERT-base-uncased tokens (|V| = 30,522 -> 15 bits per token).

    configs/lm1b.py:hcobit   H-CoBit, n = 16 clusters (19 bits per token)
    configs/lm1b.py:cobit    CoBit-S

Both models use a 1M-step cosine horizon. CoBit-S trains to the 74-epoch cap, close to
1M steps (CoBit's released checkpoint: step 992,719); H-CoBit is stopped at 600k
(train.py --stop_at_step 600000).
"""
from configs.base import DATA_ROOT, cluster_bits, get_base_config, set_bits

INFO_BITS = 15


def get_config(model="hcobit"):
    cfg = get_base_config()
    cfg.experiment = f"lm1b_{model}"

    cfg.data.dataset = "LM1B"
    cfg.data.root = f"{DATA_ROOT}/lm1b"
    cfg.data.tokenizer_name = "bert-base-uncased"
    cfg.data.sequence_len_tokens = 128
    cfg.data.val_fraction = 0.005
    cfg.train.epochs = 74

    if model == "hcobit":
        # lambda_2 = 4/15: plain loss over the 4 cluster + 15 token bits.
        cluster_bits(cfg, 16, "cluster_maps/lm1b/k_16", INFO_BITS)
    elif model == "cobit":
        set_bits(cfg, INFO_BITS)
    else:
        raise ValueError(f"model must be 'hcobit' or 'cobit', got {model!r}")
    return cfg
