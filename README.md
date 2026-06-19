# Audio Deepfake Detection

Binary classification of real vs. fake audio using deep learning. Four model families are explored through ablation experiments: CNN, RNN, CRNN, and a simple MLP on hand-crafted features.

---

## Repository Structure

```
audio-deepfake-detection-DL/
├── scripts/
│   ├── cnn_rnn/
│   │   ├── cnn_features_extraction.py   # offline preprocessing — extract mel spectrograms to .npy
│   │   ├── dataset.py                   # PyTorch Dataset for pre-extracted .npy spectrograms
│   │   ├── models.py                    # CNNPoolingBaseline, RNNBaseline, CRNNBaseline
│   │   ├── utils.py                     # Trainer, save_rich_checkpoint, plot helpers, make_tag_*
│   │   ├── train_cnn.py                 # experiment runner — CNN sweep
│   │   ├── train_rnn.py                 # experiment runner — RNN sweep
│   │   └── train_crnn.py                # experiment runner — CRNN sweep
│   └── nn_manual_features/
│       ├── dataset.py                   # PyTorch Dataset for parquet feature files
│       ├── model.py                     # DeepfakeDetectorMLP
│       ├── utils.py                     # data loading, plot helpers, make_tag_nn
│       ├── train_model.py               # experiment runner — MLP sweep
│       └── simple_nn.py                 # standalone reference script (not part of pipeline)
├── training_configs/
│   ├── cnn_pooling.yaml                 # CNN experiment grid + training config
│   ├── rnn.yaml                         # RNN experiment grid + training config
│   ├── crnn.yaml                        # CRNN experiment grid + training config
│   └── simple_nn.yaml                   # MLP experiment grid + training config
├── results/                             # committed best-model checkpoints
│   ├── cnn/
│   ├── rnn/
│   ├── crnn/
│   └── simple_nn/
└── data/                                # raw audio (not committed)
```

---

## Models

| Model | Input | Architecture |
|-------|-------|-------------|
| **CNNPoolingBaseline** | Mel spectrogram `(128 freq × ~531 time)` | Stack of Conv2d → ReLU → AvgPool2d → Dropout blocks, AdaptiveAvgPool, Linear head |
| **RNNBaseline** | Mel spectrogram as time series `(531 steps × 128 freq)` | Stacked bidirectional LSTMs with halving hidden size, last-step Linear head |
| **CRNNBaseline** | Mel spectrogram | CNN front-end compresses to `(66 steps × 2048)`, then stacked BiLSTMs |
| **DeepfakeDetectorMLP** | Hand-crafted features `(~390 values)` | Linear → BatchNorm1d → ReLU → Dropout stack, Sigmoid output |

---

## Data Setup

### CNN / RNN / CRNN

These models consume pre-extracted mel spectrograms saved as `.npy` files. Expected directory layout:

```
<split>/
├── real/
│   ├── sample_001.npy
│   └── ...
└── fake/
    ├── sample_001.npy
    └── ...
```

Where `<split>` is one of `training/`, `validation/`, `testing/`.

**To extract features from raw audio:**

```bash
# Edit CONFIG paths at the top of the script first
python scripts/cnn_rnn/cnn_features_extraction.py
```

This reads `.wav` files from `data/for-norm/`, extracts 128-bin mel spectrograms at 16 kHz, normalises them with a StandardScaler fitted on the training set, and writes `.npy` files to `data/cnn_mel_features/`.

### Simple NN (MLP)

Expects three Parquet files with pre-extracted features (MFCCs, chroma, spectral contrast, ZCR, etc.) — one per split. Update the `data:` paths in `training_configs/simple_nn.yaml` to point to your files.

---

## Running Experiments

Each YAML config contains the full experiment grid. Running the corresponding script will execute **all experiments in sequence** and generate comparison plots at the end.

### 1. Update data paths

Open the relevant YAML and set `data.train_path`, `data.val_path`, and `data.test_path` to your local feature directories (or Parquet files for the MLP).

```yaml
# training_configs/cnn_pooling.yaml
data:
  train_path: "/path/to/cnn_mel_features/training"
  val_path:   "/path/to/cnn_mel_features/validation"
  test_path:  "/path/to/cnn_mel_features/testing"
  feature_extension: ".npy"
```

### 2. Run from the repo root

```bash
# CNN — 10 experiments
python scripts/cnn_rnn/train_cnn.py --config training_configs/cnn_pooling.yaml

# RNN — 10 experiments
python scripts/cnn_rnn/train_rnn.py --config training_configs/rnn.yaml

# CRNN — 10 experiments
python scripts/cnn_rnn/train_crnn.py --config training_configs/crnn.yaml

# MLP — 10 experiments
python scripts/nn_manual_features/train_model.py --config training_configs/simple_nn.yaml
```

### 3. Outputs

Each run produces the following under `training.output_dir` (set in the YAML):

```
{output_dir}/
├── experiments/
│   ├── CNN_d3_ch32_lr5e-4_do0.5_wd1e-4/    ← one folder per experiment
│   │   ├── checkpoints/
│   │   │   ├── best_<tag>.pt                ← model weights + config + metadata
│   │   │   ├── history_<tag>.json           ← loss/accuracy per epoch
│   │   │   └── metrics_<tag>.txt            ← test accuracy, AUC, classification report
│   │   ├── plots/
│   │   │   ├── training_curves_<tag>.png
│   │   │   └── roc_auc_<tag>.png
│   │   └── model_class_<tag>.py             ← standalone model class for this checkpoint
│   └── ...
└── comparison/
    ├── roc_auc_comparison.png               ← all experiments ranked by AUC
    ├── accuracy_comparison.png              ← all experiments ranked by accuracy
    └── metrics_summary.csv                  ← sortable table of all results
```

---

## Understanding the YAML Config

Each YAML has three sections:

```yaml
model:
  name: "CNNPoolingBaseline"   # used for display only

data:
  train_path: "..."
  val_path:   "..."
  test_path:  "..."
  feature_extension: ".npy"

training:
  seed: 42
  batch_size: 32
  epochs: 30
  early_stop_patience: 5        # stop if val loss doesn't improve for N epochs
  lr_reduce_patience: 3         # reduce LR if val loss plateaus for N epochs
  lr_reduce_factor: 0.5         # multiply LR by this on plateau
  num_workers: 2
  output_dir: "results/cnn"

experiments:
  # Each entry is one run. Vary one parameter at a time (ablation style).
  - {depth: 3, base_channels: 32, lr: 5.0e-4, dropout: 0.5, weight_decay: 1.0e-4}  # baseline
  - {depth: 2, base_channels: 32, lr: 5.0e-4, dropout: 0.5, weight_decay: 1.0e-4}  # shallower
  - {depth: 4, base_channels: 32, lr: 5.0e-4, dropout: 0.5, weight_decay: 1.0e-4}  # deeper
  # ...
```

**To add or remove experiments**, edit the `experiments:` list. The script runs them in order and always generates the comparison plots at the end.

---

## Experiment Parameters by Model

### CNN (`cnn_pooling.yaml`)
| Parameter | What it controls |
|-----------|-----------------|
| `depth` | Number of Conv2d blocks (each halves spatial resolution) |
| `base_channels` | Channels in the first block (doubles each block) |
| `lr` | Adam learning rate |
| `dropout` | Dropout probability after each block |
| `weight_decay` | L2 regularisation |

### RNN (`rnn.yaml`)
| Parameter | What it controls |
|-----------|-----------------|
| `num_layers` | Number of stacked BiLSTM layers |
| `hidden_size` | Hidden units in the first BiLSTM layer (halves each layer) |
| `lr` | Adam learning rate |
| `dropout` | Dropout between LSTM layers |
| `weight_decay` | L2 regularisation |

### CRNN (`crnn.yaml`)
| Parameter | What it controls |
|-----------|-----------------|
| `cnn_depth` | Number of CNN blocks before the RNN |
| `base_channels` | Starting channels in the CNN (doubles each block) |
| `num_rnn_layers` | Number of stacked BiLSTM layers |
| `rnn_hidden` | Hidden units in the first BiLSTM layer (halves each layer) |
| `lr` | Adam learning rate |
| `dropout` | Dropout in both CNN and RNN parts |
| `weight_decay` | L2 regularisation |

### MLP (`simple_nn.yaml`)
| Parameter | What it controls |
|-----------|-----------------|
| `hidden_sizes` | List of hidden layer widths — length sets depth, values set width |
| `lr` | Adam learning rate |
| `dropout` | Dropout after each hidden layer |
| `weight_decay` | L2 regularisation |

---

## Committing the Best Model

After running experiments, pick the best from `comparison/metrics_summary.csv` and commit its files:

```
results/{model}/
├── checkpoints/
│   ├── best_<tag>.pt          ← weights + config + metadata
│   └── metrics_<tag>.txt      ← test accuracy and AUC (human-readable)
└── plots/
    ├── training_curves_<tag>.png
    └── roc_auc_<tag>.png
```

The `.pt` checkpoint is self-contained: it stores the model weights, the constructor arguments needed to rebuild the architecture, and the model class source code.

**To load a saved model:**

```python
import torch
from scripts.cnn_rnn.models import CNNPoolingBaseline  # or whichever model

ckpt = torch.load("best_CNN_d3_ch32_lr5e-4_do0.5_wd1e-4.pt", map_location="cpu")
model = CNNPoolingBaseline(**ckpt["model_config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Alternatively, use the standalone class file saved alongside the checkpoint:
# model_class_<tag>.py — no repo import needed
```

---

## Dependencies

```
torch
numpy
scikit-learn
matplotlib
pyyaml
tqdm
librosa          # feature extraction only
pandas           # MLP pipeline only
pyarrow          # MLP pipeline only (parquet reading)
```
