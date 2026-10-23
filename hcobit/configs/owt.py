"""OpenWebText, 1024 gpt2id_bpe16 code tokens (|V| = 65,536 -> 16 bits per token).

    configs/owt.py:hcobit   H-CoBit, n = 16 clusters, lambda_2 = 0.1 (20 bits per token)
    configs/owt.py:cobit    CoBit-S

gpt2id_bpe16 is CoBit's second-stage BPE over GPT-2 ids; its tokenizer ships in
assets/owt_codec/. The last 100k documents of the OWT train split are held out for
validation. Both models are trained on the 1M-step cosine horizon and stopped at 750k
(train.py --stop_at_step 750000).
"""
from configs.base import DATA_ROOT, cluster_bits, get_base_config, set_bits

INFO_BITS = 16


def get_config(model="hcobit"):
    cfg = get_base_config()
    cfg.experiment = f"owt_{model}"

    d = cfg.data
    d.dataset = "OpenWebText"
    d.root = f"{DATA_ROOT}/openwebtext"
    d.tokenizer_name = "gpt2"
    d.train_split = "train[:-100000]"
    d.valid_split = "train[-100000:]"
    d.sequence_codec = "gpt2id_bpe16"
    d.code_tokenizer_path = "tokenizer_gpt2id_bpe16_65536_base1024.json"
    d.code_tokenizer_meta_path = "tokenizer_gpt2id_bpe16_65536_base1024.meta.json"
    d.sequence_len_tokens = 1024
    d.base_sequence_len_tokens = 1024
    d.wrap = True
    d.insert_train_eos = True
    d.insert_valid_eos = True
    d.cache_num_proc = 8
    d.cache_encode_batch_size = 1000
    d.cache_write_batch_size = 8192
    d.cache_overwrite = False
    d.code_cache_batch_size = 2048
    d.code_cache_overwrite = False
    cfg.train.epochs = 100

    if model == "hcobit":
        cluster_bits(cfg, 16, "cluster_maps/owt/k_16", INFO_BITS, loss_weight=0.1)
    elif model == "cobit":
        set_bits(cfg, INFO_BITS)
    else:
        raise ValueError(f"model must be 'hcobit' or 'cobit', got {model!r}")
    return cfg
