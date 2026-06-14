
import random
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import sys

# ─────────────────────────────────────────────
# CONFIG & DIRECTORIES
# ─────────────────────────────────────────────
TRAIN_PATH    = "/kaggle/input/datasets/kostaskaimakis/for-features/features/training_features_20_128_256_128.parquet"
VAL_PATH      = "/kaggle/input/datasets/kostaskaimakis/for-features/features/testing_features_20_128_256_128.parquet"
TEST_PATH     = "/kaggle/input/datasets/kostaskaimakis/for-features/itw_features_20_128_256_128.parquet"

# New Structured Paths
BASE_DIR       = "/kaggle/working/results"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
PLOT_DIR       = os.path.join(BASE_DIR, "plots")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

SAVE_PATH      = os.path.join(CHECKPOINT_DIR, "best_simpleNN.pt")
HISTORY_PATH   = os.path.join(CHECKPOINT_DIR, "history_simpleNN.json")
METRICS_PATH   = os.path.join(CHECKPOINT_DIR, "metrics_simpleNN.txt")

BATCH_SIZE    = 64
LEARNING_RATE = 5e-4
MAX_EPOCHS    = 100
PATIENCE      = 10      
THRESHOLD     = 0.5    
SEED          = 42
MODEL_CONFIG  = {
    "hidden_sizes": [256, 128, 64],
    "dropout": 0.5,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Using seed: {SEED}")
print(f"Model config: {MODEL_CONFIG}")


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class AudioFeatureDataset(Dataset):
    def __init__(self, dataframe):
        self.features = torch.tensor(
            dataframe.drop(columns=["label", "filename"]).values.astype("float32")
        )
        self.labels = torch.tensor(
            dataframe["label"].values.astype("float32")
        ).unsqueeze(1)  

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model = DeepfakeDetectorMLP(**checkpoint["model_config"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class DeepfakeDetectorMLP(nn.Module):
    def __init__(self, input_size=390, hidden_sizes=None, dropout=0.3):
        super(DeepfakeDetectorMLP, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_features = hidden_size

        layers.extend([
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        ])

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────────
# HELPERS & PLOTTING
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_probs, all_labels = [], []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            probs = model(X)
            loss  = criterion(probs, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss  += loss.item() * len(y)
            all_probs.append(probs.detach().cpu())
            all_labels.append(y.detach().cpu())

    all_probs  = torch.cat(all_probs).numpy().squeeze()
    all_labels = torch.cat(all_labels).numpy().squeeze()
    preds      = (all_probs >= THRESHOLD).astype(int)
    accuracy   = (preds == all_labels.astype(int)).mean()
    avg_loss   = total_loss / len(all_labels)

    return avg_loss, accuracy, all_labels, all_probs


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(axes, ["loss", "acc"], ["Loss", "Accuracy"]):
        ax.plot(history[f"train_{metric}"], label="Train")
        ax.plot(history[f"val_{metric}"],   label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    save_fig_path = os.path.join(PLOT_DIR, "training_curves_simpleNN.png")
    plt.savefig(save_fig_path, dpi=150)
    plt.show()
    print(f"→ Saved {save_fig_path}")


def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    save_fig_path = os.path.join(PLOT_DIR, "roc_auc_simpleNN.png")
    plt.savefig(save_fig_path, dpi=150)
    plt.show()
    print(f"→ Saved {save_fig_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    set_seed(SEED)

    # ── 1. Load data ───────────────────────────
    print("Loading parquet files...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df   = pd.read_parquet(VAL_PATH)
    test_df  = pd.read_parquet(TEST_PATH)
    
    label_map = {'real': 0, 'fake': 1}
    for df in [train_df, val_df, test_df]:
        df.dropna(inplace=True)
        df['label'] = df['label'].map(label_map)

    feature_cols = [c for c in train_df.columns if c not in ("label", "filename")]
    input_size   = len(feature_cols)

    # ── 2. Scale ───────────────────────────────
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    val_df[feature_cols]   = val_df[feature_cols].fillna(0)
    test_df[feature_cols]  = test_df[feature_cols].fillna(0)

    # ── 3. DataLoaders ─────────────────────────
    worker_gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        AudioFeatureDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=worker_gen,
    )
    val_loader = DataLoader(AudioFeatureDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(AudioFeatureDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # ── 4. Model, loss, optimizer ──────────────
    model_config = {
        "input_size": input_size,
        "hidden_sizes": MODEL_CONFIG["hidden_sizes"],
        "dropout": MODEL_CONFIG["dropout"],
    }
    model     = DeepfakeDetectorMLP(**model_config).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # ── 5. Training loop ───────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss      = float("inf")
    best_weights       = None
    epochs_no_improve  = 0

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}  {'LR':>9}")
    print("─" * 65)

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc, _, _  = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _  = run_epoch(model, val_loader,   criterion)
        scheduler.step(vl_loss)

        # Cast explicitly to native float to ensure JSON compatibility
        history["train_loss"].append(float(tr_loss))
        history["val_loss"].append(float(vl_loss))
        history["train_acc"].append(float(tr_acc))
        history["val_acc"].append(float(vl_acc))

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>8.2%}  {vl_loss:>9.4f}  {vl_acc:>7.2%}  {lr_now:>9.2e}")

        if vl_loss < best_val_loss:
            best_val_loss    = vl_loss
            best_weights     = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    # ── 6. Save best weights + metadata ─────────
    torch.save({
        "model_state_dict": best_weights,
        "model_config":     model_config,
        "feature_cols":     feature_cols,
        "seed":             SEED,
        "pytorch_version":  torch.__version__,
        "python_version":   sys.version,
        "val_loss":         best_val_loss,
        "scaler_mean":      scaler.mean_,
        "scaler_scale":     scaler.scale_,
    }, SAVE_PATH)
    print(f"\n→ Best model saved to {SAVE_PATH}")

    # Save History JSON
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)
    print(f"→ Saved history log to {HISTORY_PATH}")

    # ── 7. Plots ───────────────────────────────
    plot_history(history)

    # ── 8. Evaluate on test set ────────────────
    model.load_state_dict(best_weights)
    te_loss, te_acc, y_true, y_probs = run_epoch(model, test_loader, criterion)
    y_pred = (y_probs >= THRESHOLD).astype(int)
    auc_score = roc_auc_score(y_true, y_probs)

    # Generate Metrics String for print and file write
    clf_report = classification_report(y_true, y_pred, target_names=["Real", "Fake"])
    metrics_output = (
        f"=============================================\n"
        f"TEST SET RESULTS\n"
        f"=============================================\n"
        f"Loss      : {te_loss:.4f}\n"
        f"Accuracy  : {te_acc:.2%}\n"
        f"ROC-AUC   : {auc_score:.4f}\n\n"
        f"Classification Report:\n{clf_report}"
    )
    
    print("\n" + metrics_output)
    
    # Save Metrics TXT
    with open(METRICS_PATH, "w") as f:
        f.write(metrics_output)
    print(f"→ Saved performance metrics to {METRICS_PATH}")

    # Plot & Save ROC AUC Curve
    plot_roc_curve(y_true, y_probs)


if __name__ == "__main__":
    main()