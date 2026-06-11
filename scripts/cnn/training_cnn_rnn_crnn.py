"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AUDIO DEEPFAKE DETECTION — PRE-SCALED FEATURE PIPELINE               ║
║       Loads pre-scaled numpy spectrogram features directly from disk         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ==============================================================================
# ▼▼▼  CONFIG — Adjust your paths and feature extension here  ▼▼▼
# ==============================================================================

CONFIG = {
    # ── Paths ─────────────────────────────────────────────────────────────────
    "TRAIN_PATH": "/kaggle/input/datasets/kostaskaimakis/cnn-rnn-mel-features/cnn_mel_features/training",
    "VAL_PATH":   "/kaggle/input/datasets/kostaskaimakis/cnn-rnn-mel-features/cnn_mel_features/validation",
    "TEST_PATH":  "/kaggle/input/datasets/kostaskaimakis/cnn-rnn-mel-features/cnn_mel_features/testing",

    "OUTPUT_DIR": "/kaggle/working/deepfake_detection",

    # ── Features ──────────────────────────────────────────────────────────────
    "FEATURE_EXTENSION": ".npy",  # Choices: ".npy" | ".pt" | ".npz"
    "SPEC_TYPE": "mel",           # Used solely for naming saved outputs

    # ── Model ─────────────────────────────────────────────────────────────────
    # Choices: "CNNPoolingBaseline" | "CRNNBaseline" | "RNNBaseline"
    "MODEL_NAME": "CRNNBaseline",

    # ── Training ──────────────────────────────────────────────────────────────
    "BATCH_SIZE":            64,
    "EPOCHS":                100,
    "LEARNING_RATE":         5e-4,
    "EARLY_STOP_PATIENCE":   5,
    "LR_REDUCE_PATIENCE":     5,
    "LR_REDUCE_FACTOR":      0.5,
    "NUM_WORKERS":            2,   
}

# ==============================================================================
# ▲▲▲  end of CONFIG  ▲▲▲
# ==============================================================================

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

_out      = Path(CONFIG["OUTPUT_DIR"])
_spec     = CONFIG["SPEC_TYPE"]
_tag      = f"{CONFIG['MODEL_NAME']}_{_spec}"

CKPT_DIR  = _out / "checkpoints" / _tag
PLOTS_DIR = _out / "plots"       / _tag

for _d in [CKPT_DIR, PLOTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {"real": 0, "fake": 1}

print(f"Device : {DEVICE}")
print(f"Tag    : {_tag}")
print(f"Output : {_out}\n")

def seed_everything(seed: int = 42):
    import random
    import os
    
    # 1. Set Python core and environment seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Set NumPy seed
    np.random.seed(seed)
    
    # 3. Set PyTorch seeds (CPU and all GPUs)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # safe for multi-GPU setups
    
    # 4. Enforce deterministic behaviors in cuDNN algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ==============================================================================
# 1.  DATASET & LOADING UTILITY
# ==============================================================================

def load_precalculated_feature(path: str) -> np.ndarray:
    ext = CONFIG["FEATURE_EXTENSION"].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".pt":
        arr = torch.load(path, map_location="cpu").numpy()
    elif ext == ".npz":
        with np.load(path) as data:
            arr = data[data.files[0]]
    else:
        raise ValueError(f"Unsupported extension format: {ext}")
    return arr.astype(np.float32)


class PrecalculatedFeatureDataset(Dataset):
    def __init__(self, root: str, split_name: str):
        ext = CONFIG["FEATURE_EXTENSION"]
        self.samples: list = []
        
        for label_dir in sorted(Path(root).iterdir()):
            if not label_dir.is_dir():
                continue
            lname = label_dir.name.lower()
            if lname not in LABEL_MAP:
                continue
            for feat_file in sorted(label_dir.glob(f"*{ext}")):
                self.samples.append((feat_file, LABEL_MAP[lname]))

        if not self.samples:
            raise RuntimeError(f"No feature files with extension '{ext}' found under {root}")

        counts = {0: 0, 1: 0}
        for _, lbl in self.samples:
            counts[lbl] += 1
        print(f"  [{split_name:>10}]  {len(self.samples)} files  "
              f"(real: {counts[0]}, fake: {counts[1]})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, label = self.samples[idx]
        
        # Load the already-scaled array from disk
        spec = load_precalculated_feature(str(feat_path))
        return torch.tensor(spec), torch.tensor(label)


def get_dataloaders():
    print("=" * 60)
    print("STEP 1 — BUILDING DATALOADERS")
    print("=" * 60)

    pin = torch.cuda.is_available()
    loaders = {}
    
    for split, path in [
        ("training",   CONFIG["TRAIN_PATH"]),
        ("validation", CONFIG["VAL_PATH"]),
        ("testing",    CONFIG["TEST_PATH"]),
    ]:
        ds = PrecalculatedFeatureDataset(path, split_name=split)
        loaders[split] = DataLoader(
            ds,
            batch_size  = CONFIG["BATCH_SIZE"],
            shuffle     = (split == "training"),
            num_workers = CONFIG["NUM_WORKERS"],
            pin_memory  = pin,
        )

    print()
    return loaders["training"], loaders["validation"], loaders["testing"]


# ==============================================================================
# 2.  MODEL ARCHITECTURES
# ==============================================================================

class CNNPoolingBaseline(nn.Module):
    def __init__(self, num_classes=1): 
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class CRNNBaseline(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.rnn_input_dim = 128 * 16 
        self.lstm1 = nn.LSTM(input_size=self.rnn_input_dim, hidden_size=128, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.cnn(x)
        
        x = x.permute(0, 3, 1, 2).contiguous()
        batch_size, seq_len, channels, freq = x.size()
        x = x.view(batch_size, seq_len, channels * freq)
        
        x, _ = self.lstm1(x)
        x = self.relu(x)
        x = self.drop1(x)
        
        x, _ = self.lstm2(x)
        x = self.relu(x)
        x = self.drop2(x)
        
        out_last_step = x[:, -1, :]
        return self.classifier(out_last_step)


class RNNBaseline(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = self.relu(x)
        x = self.drop1(x)
        
        x, _ = self.lstm2(x)
        x = self.relu(x)
        x = self.drop2(x)
        
        out_last_step = x[:, -1, :]
        return self.classifier(out_last_step)


MODEL_REGISTRY = {
    "CNNPoolingBaseline": CNNPoolingBaseline,
    "CRNNBaseline":       CRNNBaseline,
    "RNNBaseline":        RNNBaseline,
}


# ==============================================================================
# 3.  TRAINER
# ==============================================================================

class Trainer:
    def __init__(self, model: nn.Module):
        self.model     = model.to(DEVICE)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"],weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode     = "min",
            factor   = CONFIG["LR_REDUCE_FACTOR"],
            patience = CONFIG["LR_REDUCE_PATIENCE"],
        )
        self.history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}

    @staticmethod
    def _preds(logits: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) > 0.5).long()

    def _run_epoch(self, loader: DataLoader, train: bool):
        self.model.train() if train else self.model.eval()
        total_loss = correct = total = 0
        desc = "  Train" if train else "  Val  "
        ctx  = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for specs, labels in tqdm(loader, desc=desc, leave=False):
                specs  = specs.to(DEVICE)
                labs_f = labels.float().unsqueeze(1).to(DEVICE)

                if train:
                    self.optimizer.zero_grad()

                logits = self.model(specs)
                loss   = self.criterion(logits, labs_f)

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                correct    += (self._preds(logits) == labs_f.long()).sum().item()
                total      += labs_f.size(0)

        return total_loss / len(loader), correct / total

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Path:
        best_val_loss    = float("inf")
        patience_counter = 0
        ckpt_path        = CKPT_DIR / f"best_{_tag}.pt"

        print(f"\n{'Ep':>4}  {'T-Loss':>8}  {'T-Acc':>7}  {'V-Loss':>8}  {'V-Acc':>7}")
        print("─" * 48)

        for epoch in range(CONFIG["EPOCHS"]):
            t_loss, t_acc = self._run_epoch(train_loader, train=True)
            v_loss, v_acc = self._run_epoch(val_loader,   train=False)
            self.scheduler.step(v_loss)

            self.history["train_loss"].append(t_loss)
            self.history["train_acc"].append(t_acc)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)

            marker = ""
            if v_loss < best_val_loss:
                best_val_loss    = v_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), ckpt_path)
                marker = "  ✓ best"
            else:
                patience_counter += 1

            print(f"{epoch+1:>4}  {t_loss:>8.4f}  {t_acc:>7.4f}  "
                  f"{v_loss:>8.4f}  {v_acc:>7.4f}{marker}")

            if patience_counter >= CONFIG["EARLY_STOP_PATIENCE"]:
                print(f"\n  Early stopping at epoch {epoch + 1}.")
                break

        print(f"\n  Best val loss : {best_val_loss:.4f}")
        print(f"  Checkpoint    : {ckpt_path}\n")
        return ckpt_path

    def evaluate(self, test_loader: DataLoader):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for specs, labels in tqdm(test_loader, desc="  Test ", leave=False):
                specs  = specs.to(DEVICE)
                logits = self.model(specs)
                probs  = torch.sigmoid(logits).squeeze(1)
                preds  = (probs > 0.5).long()

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

        acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        return acc, all_preds, all_labels, all_probs


# ==============================================================================
# 4.  PLOTS & METRICS
# ==============================================================================

def _save_plot(fig: plt.Figure, name: str):
    
    p = PLOTS_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {p}")


def plot_training_curves(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training curves — {_tag}", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val",   linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train", linewidth=2)
    ax.plot(epochs, history["val_acc"],   label="Val",   linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1); ax.set_title("Accuracy"); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    _save_plot(fig, f"training_curves_{_tag}.png")


def plot_roc_auc(labels: list, probs: list) -> float:
    fpr, tpr, _ = roc_curve(labels, probs)
    auc          = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2.5, color="steelblue", label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.10, color="steelblue")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {_tag}", fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, f"roc_auc_{_tag}.png")
    return auc


def save_results(history: dict, test_acc: float,
                 preds: list, labels: list, probs: list, auc: float):
    hist_path = CKPT_DIR / f"history_{_tag}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History  → {hist_path}")

    report  = classification_report(labels, preds, target_names=["Real", "Fake"], digits=4)
    metrics = (
        f"Model        : {_tag}\n"
        f"Test accuracy: {test_acc:.4f}\n"
        f"AUC-ROC      : {auc:.4f}\n\n"
        f"{report}"
    )
    print("\n" + metrics)

    report_path = CKPT_DIR / f"metrics_{_tag}.txt"
    with open(report_path, "w") as f:
        f.write(metrics)
    print(f"  Metrics  → {report_path}")


# ==============================================================================
# 5.  MAIN
# ==============================================================================

def main():
    # Fix the random seeds before doing ANYTHING else
    seed_everything(seed=42)
    # ── Step 1: DataLoaders ───────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders()

    # ── Step 2: Model ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 2 — MODEL")
    print("=" * 60)
    if CONFIG["MODEL_NAME"] not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{CONFIG['MODEL_NAME']}'. Choose from: {list(MODEL_REGISTRY)}")
        
    model    = MODEL_REGISTRY[CONFIG["MODEL_NAME"]]()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {CONFIG['MODEL_NAME']}  —  {n_params:,} parameters\n")

    # ── Step 3: Training ──────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 3 — TRAINING")
    print("=" * 60)
    trainer   = Trainer(model)
    ckpt_path = trainer.fit(train_loader, val_loader)

    # ── Step 4: Training curves ───────────────────────────────────────────────
    print("=" * 60)
    print("STEP 4 — TRAINING CURVES")
    print("=" * 60)
    plot_training_curves(trainer.history)

    # ── Step 5: Test evaluation ───────────────────────────────────────────────
    print("=" * 60)
    print("STEP 5 — TEST EVALUATION")
    print("=" * 60)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print(f"  Loaded best weights from {ckpt_path}")
    test_acc, preds, labels, probs = trainer.evaluate(test_loader)

    # ── Step 6: ROC-AUC curve & Results ───────────────────────────────────────
    print("=" * 60)
    print("STEP 6 — SAVING RESULTS")
    print("=" * 60)
    auc = plot_roc_auc(labels, probs)
    save_results(trainer.history, test_acc, preds, labels, probs, auc)

    print("\n" + "=" * 60)
    print(f"  DONE.  All outputs in: {_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()