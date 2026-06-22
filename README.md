# Audio Deepfake Detection

Binary classification of real vs. fake speech with deep learning. The project compares
several model families under a single **cross-dataset protocol**: train and validate on
**Fake-or-Real (FoR)** (in-domain), then test on **In-the-Wild (ITW)** (a different,
unseen dataset). Models are compared mainly by **AUC** and **accuracy** (EER,
precision/recall and F1 are also reported).

Model families covered:

| Family | Where | Input | Headline idea |
|--------|-------|-------|----------------|
| **MLP** | `scripts/nn_manual_features/` | hand-crafted features (MFCC, chroma, …) | cheap baseline on tabular features |
| **CNN / RNN / CRNN** | `scripts/cnn_rnn/` | mel spectrograms (`.npy`) | spectrogram baselines  |
| **WavLM probe** | `scripts/wavlm/wavlm_train.py` | raw waveform | frozen WavLM + weighted-layer-sum + linear head |
| **WavLM fine-tune** | `scripts/wavlm/wavlm_finetune.py` | raw waveform | partial backbone fine-tune + online augmentation |

> **Key result:** a *frozen* WavLM probe generalises to ITW better than backbone
> fine-tuning.

---

## Quick start

### 1. Prerequisites

- **Python 3.12** (pinned in `pyproject.toml`)
- [**uv**](https://docs.astral.sh/uv/) for dependency management
- An NVIDIA GPU is strongly recommended for WavLM. The lockfile pulls the **CUDA 12.8**
  PyTorch build (`torch`/`torchaudio` from the `pytorch-cu128` index).

### 2. Install

```bash
git clone <repo-url>
cd audio-deepfake-detection-DL
uv sync            # creates .venv and installs the exact locked dependencies
```

Run anything either by prefixing with `uv run`, or by activating the venv:

```bash
# option A — per command
uv run python -m scripts.wavlm.wavlm_train --config training_configs/wavlm_base_plus.yaml

# option B — activate once
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m scripts.wavlm.wavlm_train --config training_configs/wavlm_base_plus.yaml
```

> **Important:** every script uses package-relative imports (`from scripts...`), so run
> them as **modules** from the repo root (`python -m scripts.<pkg>.<module>`), *not* as
> file paths (`python scripts/.../train.py`), which raises `ModuleNotFoundError`.

---

## Datasets

The `data/` and `results/` folders are **shared** (not committed — both are git-ignored).
Mount them at the **repo root** so the default config paths resolve:

```
audio-deepfake-detection-DL/
├── data/        ← shared datasets, mounted here
└── results/     ← shared run outputs / checkpoints, mounted here
```

On Windows you can mount with a directory symlink, e.g.:

```powershell
New-Item -ItemType SymbolicLink -Path .\data    -Target "<path-to-shared>\data"
New-Item -ItemType SymbolicLink -Path .\results -Target "<path-to-shared>\results"
```

(Linux/macOS: `ln -s <path-to-shared>/data data` and the same for `results`.)

The datasets below live under the mounted `data/`. Point the config paths at them.

| Dataset | Used by | Purpose | Default config path |
|---------|---------|---------|---------------------|
| **Fake-or-Real (FoR)** — `for-norm` variant | all | train + validation (in-domain) | `data/FoR_dataset/for-norm/for-norm/{training,testing}` |
| **In-the-Wild (ITW)** | all | test (cross-dataset) | `data/in-the-wild-audio-deepfake/release_in_the_wild_trimmed_normalized` |
| **MUSAN** | WavLM augmentation | additive-noise source | `data/musan` |
| **RIRS_NOISES** (simulated RIRs) | WavLM augmentation | reverberation source | `data/RIRS_NOISES/simulated_rirs` |

Each audio split is organised as one folder per class:

```
<split>/
├── real/   *.wav      (label 0)
└── fake/   *.wav      (label 1)
```

The WavLM scripts read `.wav` directly. The CNN/RNN/CRNN scripts read **pre-extracted
mel spectrograms** as `.npy` (see below). The MLP reads **Parquet** feature tables.

---

## Models

| Model | Input | Architecture |
|-------|-------|-------------|
| **DeepfakeDetectorMLP** | hand-crafted features `(~390 values)` | Linear → BatchNorm1d → ReLU → Dropout stack, Sigmoid output |
| **CNNPoolingBaseline** | Mel spectrogram `(128 freq × ~531 time)` | Conv2d → ReLU → AvgPool2d → Dropout blocks, AdaptiveAvgPool, Linear head |
| **RNNBaseline** | Mel spectrogram as time series `(531 steps × 128 freq)` | Stacked bidirectional LSTMs with halving hidden size, last-step Linear head |
| **CRNNBaseline** | Mel spectrogram | CNN front-end compresses to `(66 steps × 2048)`, then stacked BiLSTMs |
| **WavLMClassifier** | raw waveform (16 kHz) | `microsoft/wavlm-base-plus` → softmax-weighted sum over the 12 transformer layers → mean-pool → linear head |

---

## Running the experiments

### A. MLP — hand-crafted features

Expects three Parquet files (one per split) with pre-extracted features (MFCCs, chroma,
spectral contrast, ZCR, …). Update the `data:` paths in the config, then run the sweep:

```bash
python -m scripts.nn_manual_features.train_model --config training_configs/simple_nn.yaml
```

> The feature-extraction step itself was run on Kaggle; `simple_nn.py` is a standalone
> reference script and is not part of the sweep pipeline.

### B. CNN / RNN / CRNN — mel spectrograms

**1. Extract mel spectrograms** (offline, once). Edit the `CONFIG` paths at the top of the
script, then:

```bash
python -m scripts.cnn_rnn.cnn_features_extraction
```

This reads `.wav` files, extracts 128-bin mel spectrograms at 16 kHz, normalises them with
a `StandardScaler` fitted on the training set, and writes `.npy` files (one per clip) into
the directory layout shown above.

**2. Point the config** `data.{train,val,test}_path` at those `.npy` directories, then run
each sweep (each config holds a full experiment grid and runs all entries in sequence):

```bash
python -m scripts.cnn_rnn.train_cnn  --config training_configs/cnn_pooling.yaml
python -m scripts.cnn_rnn.train_rnn  --config training_configs/rnn.yaml
python -m scripts.cnn_rnn.train_crnn --config training_configs/crnn.yaml
```

### C. WavLM — frozen backbone

Frozen backbone; features are extracted once and **cached** to `{output_dir}/cache/`, then
a linear head + per-layer weights are trained on the cached features. Reads raw
`.wav`.

```bash
python -m scripts.wavlm.wavlm_train --config training_configs/wavlm_base_plus.yaml
```

### D. WavLM — fine-tune (with or without augmentation)

Partially un-freezes the backbone (conv feature encoder frozen, top-N transformer layers
trainable — set via `finetune.n_trainable_layers` / `freeze_feature_encoder`). Applies
**online waveform augmentation to the train split only**: MUSAN additive noise + simulated
RIR reverb (and optionally pre-rendered codec variants). Uses bf16 autocast, gradient
checkpointing, discriminative learning rates and early stopping.

```bash
# (optional) pre-render codec-degraded train variants so p_codec > 0 has something to swap in
python -m scripts.wavlm.wavlm_precompute_codec --config training_configs/wavlm_base_plus.yaml

python -m scripts.wavlm.wavlm_finetune --config training_configs/wavlm_base_plus.yaml
```

To reproduce the **aug vs. no-aug** comparison, toggle `augment.enabled` in the config.

---

## Outputs

### Spectrogram / MLP sweeps

Each run writes under `training.output_dir`:

```
{output_dir}/
├── experiments/
│   └── <experiment_tag>/
│       ├── checkpoints/   best_<tag>.pt, history_<tag>.json, metrics_<tag>.txt
│       ├── plots/         training_curves_<tag>.png, roc_auc_<tag>.png
│       └── model_class_<tag>.py    ← standalone model class for this checkpoint
└── comparison/
    ├── roc_auc_comparison.png      ← all experiments ranked by AUC
    ├── accuracy_comparison.png
    └── metrics_summary.csv         ← sortable table of all results
```

The `.pt` checkpoint is self-contained: weights + constructor args + the model class
source. To reload:

```python
import torch
from scripts.cnn_rnn.models import CNNPoolingBaseline   # or whichever model

ckpt = torch.load("best_<tag>.pt", map_location="cpu")
model = CNNPoolingBaseline(**ckpt["model_config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
# (or import the standalone model_class_<tag>.py saved alongside it — no repo import needed)
```

### WavLM runs

Each run writes to its `*.output_dir`:

- `best_head.pt` (frozen) or `best_model.pt` (fine-tune) — best-val checkpoint
- `metrics_*.json` — full FoR-val + ITW-test metrics (loss/acc/AUC/EER/precision/recall/F1),
  the resolved config, and the learned per-layer softmax weights
- `cache/{train,val,test}.pt` — cached features (probe only)
- When `analysis.collect_embedding: true`: per-split embedding dumps plus a **UMAP** plot
  and a **score-histogram** under `analysis.output_dir`

> `results/` is git-ignored and shared (mounted at the repo root, see
> [Datasets](#datasets)) — run outputs and checkpoints live there, not in the repo.

---

## Configuration

All experiments are driven by YAML in `training_configs/`. The sweep configs
(`cnn_pooling`, `rnn`, `crnn`, `simple_nn`) share a `model / data / training / experiments`
shape, where `experiments:` is a list of runs varied one knob at a time (ablation style):

```yaml
training:
  seed: 42
  batch_size: 32
  epochs: 30
  early_stop_patience: 5      # stop if val loss doesn't improve for N epochs
  lr_reduce_patience: 3       # reduce LR if val loss plateaus for N epochs
  lr_reduce_factor: 0.5
  output_dir: "results/cnn"

experiments:
  - {depth: 3, base_channels: 32, lr: 5.0e-4, dropout: 0.5, weight_decay: 1.0e-4}  # baseline
  - {depth: 2, base_channels: 32, lr: 5.0e-4, dropout: 0.5, weight_decay: 1.0e-4}  # shallower
  # ...
```

The WavLM config (`wavlm_base_plus.yaml`) instead has
`model / data / training / finetune / augment / analysis` sections — each is read by the
corresponding script. Important configs:

- `data.train_dir` / `val_dir` / `test_dir` — the FoR→ITW protocol (train/val = FoR, test = ITW)
- `finetune.n_trainable_layers`, `freeze_feature_encoder` — how much backbone to un-freeze
- `augment.{p_noise, p_rir, p_codec, snr_db_range}` — online train-time augmentation
- `analysis.collect_embedding`, `analysis.splits` — embedding dumps + UMAP/score plots


---

## Repository structure

```
audio-deepfake-detection-DL/
├── scripts/
│   ├── nn_manual_features/      # MLP on hand-crafted features (Parquet)
│   │   ├── dataset.py  model.py  utils.py  train_model.py  simple_nn.py
│   ├── cnn_rnn/                 # CNN / RNN / CRNN on mel spectrograms (.npy)
│   │   ├── cnn_features_extraction.py   # offline mel-spectrogram extraction
│   │   ├── dataset.py  models.py  utils.py
│   │   └── train_cnn.py  train_rnn.py  train_crnn.py
│   └── wavlm/                   # WavLM probe / fine-tune
│       ├── wavlm_dataset.py            # raw-waveform Dataset (+ offline codec swap)
│       ├── wavlm_augment.py            # online MUSAN/RIR/codec augmentation pipeline
│       ├── wavlm_precompute_codec.py   # offline codec-variant renderer (PyAV)
│       ├── wavlm_extract.py            # feature extraction + caching
│       ├── wavlm_train.py              # frozen probe (weighted-layer-sum + head)
│       ├── wavlm_finetune.py           # partial backbone fine-tune + augmentation
│       └── utils.py                    # config, dataloaders, optimizers, EER, UMAP/plots
├── training_configs/           # one YAML per experiment family
│   ├── simple_nn.yaml  cnn_pooling.yaml  rnn.yaml  crnn.yaml
│   └── wavlm_base_plus.yaml            # probe + fine-tune + augment
├── pyproject.toml  uv.lock     # dependencies (managed with uv)
├── results/                    # run outputs (git-ignored)
└── data/                       # datasets (git-ignored)
```

---

## Dependencies

Managed entirely through `pyproject.toml` + `uv.lock` (`uv sync`). The main libraries:

- **PyTorch** (`torch`, `torchaudio`, CUDA 12.8 build) — all models
- **transformers** — WavLM backbone (`wavlm-base-plus`)
- **librosa**, **soundfile**, **av** (PyAV) — audio I/O, feature extraction, codec augmentation
- **scikit-learn**, **numpy**, **scipy**, **pandas**, **pyarrow** — metrics + tabular features
- **umap-learn**, **matplotlib**, **seaborn** — embedding visualisation
- **tensorboard**, **tqdm**, **pyyaml** — logging, progress, config
</content>
