"""Download and pre-tokenize the datasets (run once, before training or evaluation).

    python scripts/prepare_data.py lm1b      # LM1B train/test text, detokenized as in SEDD/MDLM
    python scripts/prepare_data.py owt       # OpenWebText -> GPT-2 ids -> gpt2id_bpe16 code caches
    python scripts/prepare_data.py tinygsm   # TinyGSM train/val caches + the GSM8K test set

Everything is written under $HCDLM_DATA (default: ./datasets). Caches are built in a single
process so that a later multi-GPU run only reads them.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs import load_config  # noqa: E402
from configs.base import DATA_ROOT  # noqa: E402


def lm1b_detokenizer(x):
    x = x.strip()
    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = re.sub(r" \'(\w+)", r"'\1", x)
    x = re.sub(r" (\w+) \. ", r" \1. ", x)
    x = re.sub(r" (\w+) \.$", r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = re.sub(r" \?$", "?", x)
    x = x.replace(" ! ", "! ")
    x = re.sub(r" \!$", "!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
    x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("£ ", "£")
    return x.strip()


def _pack_lm1b(text_path: Path, out: Path, split: str, seq_len: int = 128):
    """BERT-base-uncased ids of every non-empty line (no special tokens), concatenated
    with [SEP] between examples and cut into seq_len-token rows; the tail is dropped."""
    import numpy as np
    from tqdm import tqdm
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    lines = [l.strip() for l in open(text_path, encoding="utf-8", errors="ignore") if l.strip()]
    stream, rows, n_examples = [], [], 0
    for i in tqdm(range(0, len(lines), 2048), desc=f"packing {split}"):
        for ids in tok(lines[i:i + 2048], add_special_tokens=False, padding=False, truncation=False)["input_ids"]:
            if not ids:
                continue
            if n_examples:
                stream.append(tok.sep_token_id)
            stream.extend(ids)
            n_examples += 1
        n_full = len(stream) // seq_len
        if n_full:
            rows.append(np.asarray(stream[: n_full * seq_len], dtype=np.uint16).reshape(n_full, seq_len))
            del stream[: n_full * seq_len]
    arr = np.concatenate(rows)
    arr.tofile(out / f"cache_{split}_tokens.uint16")
    meta = dict(
        cache_format="packed_token_blocks", seq_len_tokens=seq_len, n_sequences=int(arr.shape[0]),
        tokenizer_name="bert-base-uncased", vocab_size=int(tok.vocab_size), add_special_tokens=False,
        packing_boundary_mode="sep", packing_boundary_token_id=int(tok.sep_token_id), keep_remainder=False,
        dropped_tail_tokens=len(stream), total_stream_tokens=int(arr.size + len(stream)),
        n_nonempty_examples=n_examples,
    )
    (out / f"cache_{split}_tokens.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{out / f'cache_{split}_tokens.uint16'}: {arr.shape[0]} x {seq_len}")


LM1B_REPO = "jdeschena/lm1b"  # parquet copy of LM1B; CoBit's download script falls back to it
LM1B_SHARDS = {"train": [f"data/train-{i:05d}-of-00009.parquet" for i in range(9)],
               "test": ["data/test-00000-of-00001.parquet"]}


def prepare_lm1b():
    """LM1B (the split CoBit and prior work use) from its Hugging Face parquet copy, read
    in shard order, detokenized as in SEDD / MDLM, then packed into BERT token caches.
    The text is line-for-line identical, in order, to the text our models were trained
    and evaluated on (30,301,028 train / 306,688 test sentences); TensorFlow is not needed."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    out = Path(DATA_ROOT) / "lm1b"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        text_path = out / f"{split}.txt"
        n = 0
        with open(text_path, "w", encoding="utf-8") as f:
            for shard in tqdm(LM1B_SHARDS[split], desc=split):
                path = hf_hub_download(LM1B_REPO, shard, repo_type="dataset")
                for text in pq.read_table(path, columns=["text"]).column("text").to_pylist():
                    txt = lm1b_detokenizer(text or "")
                    if not txt:
                        continue
                    f.write(txt + "\n")
                    n += 1
        print(f"{text_path}: {n} sentences")
        _pack_lm1b(text_path, out, split)


def prepare_owt():
    root = Path(DATA_ROOT) / "openwebtext"
    root.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "assets/owt_codec").iterdir():
        shutil.copy(f, root / f.name)
    from data.openwebtext import ensure_openwebtext_caches_ready
    cfg = load_config(str(ROOT / "configs/owt.py") + ":hcobit")
    for split in ("train", "val"):
        ensure_openwebtext_caches_ready(cfg, split=split)


def prepare_tinygsm():
    from datasets import load_dataset
    from data.tinygsm import TinyGSMDataset

    out = Path(DATA_ROOT) / "gsm8k"
    out.mkdir(parents=True, exist_ok=True)
    test = load_dataset("openai/gsm8k", "main", split="test")
    records = [{"prompt": r["question"], "response_ground_truth": r["answer"]} for r in test]
    (out / "gsm8k_test.json").write_text(json.dumps(records, indent=1))
    print(f"{out / 'gsm8k_test.json'}: {len(records)} problems")
    cfg = load_config(str(ROOT / "configs/tinygsm.py"))
    for split in ("train", "val"):
        print(f"tinygsm {split}: {len(TinyGSMDataset(cfg, split=split))} examples")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["lm1b", "owt", "tinygsm"])
    args = ap.parse_args()
    {"lm1b": prepare_lm1b, "owt": prepare_owt, "tinygsm": prepare_tinygsm}[args.dataset]()


if __name__ == "__main__":
    main()
