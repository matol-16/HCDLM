# H-CoBit

Code to reproduce the H-CoBit results of the paper: Table 1 and Figure 4 (GenPPL / entropy on LM1B and OpenWebText), Table 11 (MAUVE) and Table 2 (GSM8K). It is CoBit's implementation ([Batzolis et al., 2026](https://arxiv.org/abs/2605.07013)) with the cluster modality added, reduced to what these results use. Both CoBit-S and H-CoBit can be trained and evaluated.

```
configs/        base.py (CoBit-S settings), lm1b.py, owt.py, tinygsm.py, sampling.py (samplers and sweeps)
cluster_maps/   the n = 16 cluster maps (Appendix A); scripts/cluster_maps/ rebuilds them
assets/         the OpenWebText gpt2id_bpe16 tokenizer
results/        the paper's own numbers, one row per operating point
train.py        training
evaluate.py     evaluation
scripts/        data preparation, cluster maps, checkpoint download, release packaging, report
```

## Setup

```bash
conda create -n hcobit python=3.11 -y
conda activate hcobit
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Data goes to `./datasets` (or `$HCDLM_DATA`):

```bash
python scripts/prepare_data.py lm1b      # LM1B from Hugging Face (parquet), BERT token caches
python scripts/prepare_data.py owt       # OpenWebText from Hugging Face, gpt2id_bpe16 code caches
python scripts/prepare_data.py tinygsm   # TinyGSM caches and the GSM8K test set
```

## Reproduce the paper's results from the released checkpoints

### 1. Download the checkpoints

```bash
python scripts/download_checkpoints.py                # all five -> checkpoints/<name>/
python scripts/download_checkpoints.py hcobit_lm1b    # or only some of them
```

| Name | Model | Hugging Face repo |
|---|---|---|
| `hcobit_lm1b` | H-CoBit, LM1B, 600k steps | [matol-16/hcdlm](https://huggingface.co/matol-16/hcdlm) |
| `hcobit_owt` | H-CoBit, OpenWebText, 750k steps | [matol-16/hcdlm](https://huggingface.co/matol-16/hcdlm) |
| `hcobit_tinygsm` | H-CoBit, TinyGSM (GSM8K), 250k steps | [matol-16/hcdlm](https://huggingface.co/matol-16/hcdlm) |
| `cobit_lm1b` | CoBit-S, LM1B, 1M steps[^cobit1m] | [gbatzolis/CoBit](https://huggingface.co/gbatzolis/CoBit) |
| `cobit_owt` | CoBit-S, OpenWebText, 750k steps | [gbatzolis/CoBit](https://huggingface.co/gbatzolis/CoBit) |

Both repositories are public, so no Hugging Face login is needed. Files go through the Hugging Face cache (`$HF_HOME`) and are copied into `checkpoints/<name>/`. Files already there are skipped, and `--out` sets another destination.

Each folder holds the EMA weights (`model.pt`, about 0.5 GB) and the noise schedule the model is sampled with:

- CoBit-S: `entropy_{pdf,cdf,sigmas,edges}.pt`.
- H-CoBit on LM1B and OpenWebText: `entropy_profile_{info,cluster}.pt`.
- H-CoBit on TinyGSM: `entropy_*_{info,cluster}.pt` and `sigma_data.json`.

**Compute nodes without internet access.** Do these steps on a machine that has access:

1. Run the download above.
2. Run `prepare_data.py`, which also caches the tokenizers and the GSM8K test set.
3. Fetch the GPT-2 Large judge:

   ```bash
   hf download openai-community/gpt2-large && hf download gpt2-large
   ```

Then set `HF_HUB_OFFLINE=1` on the compute nodes.

### 2. Evaluate

`evaluate.py <metric> <dataset> <arm>` reads `checkpoints/<model>_<dataset>/` by default, and `--checkpoint` points it elsewhere. Each run writes one row per operating point to `runs/eval/results/`, in the format of `results/`.

Arms are:

- `cobit` and `cobit_snoise1.05` (CoBit-S);
- `hcobit`;
- `hcobit_pms` (H-CoBit + per-modality sampler).

Their samplers and sweeps are in `configs/sampling.py`.

The paper's H-CoBit operating points:

| Result | Command | Paper |
|---|---|---|
| Table 1, LM1B, H-CoBit | `python evaluate.py genppl lm1b hcobit --nfe 256 --gamma 0.225` | GenPPL 57.3, $H$ 4.34[^lm1b] |
| Table 1, LM1B, + per-modality sampler | `python evaluate.py genppl lm1b hcobit_pms --nfe 512 --gamma 0.3` | GenPPL 49.4, $H$ 4.34 |
| Table 1, OWT, H-CoBit | `python evaluate.py genppl owt hcobit --nfe 128 --gamma 0.18` | GenPPL 67.8, $H$ 5.45 |
| Table 1, OWT, + per-modality sampler | `python evaluate.py genppl owt hcobit_pms --nfe 256 --gamma 0.25` | GenPPL 50.4, $H$ 5.46 |
| Table 11, LM1B | `python evaluate.py mauve lm1b hcobit --nfe 128 --gamma 0.05` | MAUVE 0.903[^lm1b] |
| Table 11, OWT | `python evaluate.py mauve owt hcobit --nfe 128 --gamma 0.17` | MAUVE 0.580 |
| Table 2, GSM8K | `python evaluate.py gsm8k tinygsm hcobit --nfe 512 --gamma 0.38` | 27.4% (362 / 1319)[^gsm8k] |

[^gsm8k]: A single sampling run (seed 42, batch size 64), following S-FLM and CoBit. Measured on one NVIDIA A100-SXM 64 GB GPU; another batch size or GPU gives a different draw of samples.

[^cobit1m]: The 1M-step learning-rate schedule; training ends at the 74-epoch cap, at step 992,719.


The CoBit-S rows use the same commands with arm `cobit` or `cobit_snoise1.05`; their operating points are in `results/`.

#### Full sweeps

Without `--nfe` and `--gamma`, `genppl` and `mauve` run the arm's full (NFE, churn) sweep from `configs/sampling.py`. These are the sweeps behind Figure 4 and Tables 1 and 11:

```bash
for ds in lm1b owt; do
  for arm in cobit cobit_snoise1.05 hcobit hcobit_pms; do python evaluate.py genppl $ds $arm; done
  for arm in cobit hcobit; do python evaluate.py mauve $ds $arm; done
done
python scripts/report.py --results runs/eval/results --plot frontiers.png
```

GSM8K has no built-in sweep. This loop runs the (NFE, churn) grid of Table 2:

```bash
sweep() { nfe=$1; shift; for g in "$@"; do python evaluate.py gsm8k tinygsm hcobit --nfe $nfe --gamma $g; done; }
sweep 256  0 0.21 0.26 0.31 0.36
sweep 384  0 0.3 0.36 0.4 0.4142
sweep 512  0 0.34 0.38 0.4 0.4142
sweep 1024 0 0.15 0.2 0.26 0.3 0.34 0.38 0.4 0.4142
```

`python scripts/report.py` without arguments prints the paper's numbers from `results/`, for comparison.

No CoBit-S checkpoint is released for TinyGSM; [train](#train) one to evaluate it on GSM8K.


## Train

```bash
# H-CoBit
torchrun --standalone --nproc_per_node=4 train.py --config configs/lm1b.py:hcobit --stop_at_step 600000
torchrun --standalone --nproc_per_node=4 train.py --config configs/owt.py:hcobit --stop_at_step 750000
torchrun --standalone --nproc_per_node=4 train.py --config configs/tinygsm.py:hcobit
# CoBit-S
torchrun --standalone --nproc_per_node=4 train.py --config configs/lm1b.py:cobit
torchrun --standalone --nproc_per_node=4 train.py --config configs/owt.py:cobit --stop_at_step 750000
torchrun --standalone --nproc_per_node=4 train.py --config configs/tinygsm.py:cobit
```

The batch size is global, so the number of GPUs does not change training. Relaunching a command resumes from `runs/<experiment>/checkpoints/last.pt`.

To evaluate a trained model, package it like the released ones (`--step` picks `step=<step>.pt`, default `last.pt`). H-CoBit on LM1B and OpenWebText also needs its noise profiles (`evaluate.py profile`):

```bash
python scripts/make_release.py --run runs/lm1b_hcobit --step 600000 --out checkpoints/hcobit_lm1b
python evaluate.py profile lm1b hcobit
python scripts/make_release.py --run runs/tinygsm_cobit --step 250000 --out checkpoints/cobit_tinygsm --task
python evaluate.py gsm8k tinygsm cobit --nfe 512 --gamma 0.34
```

## Cluster maps

Each token's cluster is its k-means++ cluster (n = 16) over the token embeddings of a pretrained model with the same tokenizer: [LangFlow-LM1B](https://huggingface.co/Continuous-Rivals-Discrete/langflow-lm1b) for LM1B, [LangFlow-OWT](https://huggingface.co/Continuous-Rivals-Discrete/langflow-owt) for OpenWebText and [S-FLM](https://huggingface.co/jdeschena/s-flm) for TinyGSM. The released checkpoints were trained with the shipped maps in `cluster_maps/`; use those with them.

To rebuild the maps (OpenWebText and TinyGSM first need `python scripts/prepare_data.py owt` / `tinygsm`):

```bash
python scripts/build_cluster_maps.py               # all three datasets (or: lm1b, owt, tinygsm)
```

The maps are written to `runs/cluster_maps/<dataset>/k_16/`, and the script reports whether each matches the shipped one. OpenWebText and TinyGSM match exactly; LM1B's k-means reaches a different local optimum on other hardware, so its rebuilt map differs.
