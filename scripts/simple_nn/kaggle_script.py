import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import copy

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_PATH    = "/kaggle/input/datasets/kostaskaimakis/for-features/features/training_features_20_128_256_128.parquet"
VAL_PATH      = "/kaggle/input/datasets/kostaskaimakis/for-features/features/testing_features_20_128_256_128.parquet"
TEST_PATH     = "/kaggle/input/datasets/kostaskaimakis/for-features/itw_features_20_128_256_128.parquet"
SAVE_PATH     = "/kaggle/working/best_model.pt"

BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS    = 100
PATIENCE      = 10      # stop after N epochs with no val loss improvement
THRESHOLD     = 0.5     # sigmoid threshold for Real(0) vs Fake(1)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class AudioFeatureDataset(Dataset):
    def __init__(self, dataframe):
        self.features = torch.tensor(
            dataframe.drop(columns=["label", "filename"]).values.astype("float32")
        )
        # float32 labels required by BCELoss
        self.labels = torch.tensor(
            dataframe["label"].values.astype("float32")
        ).unsqueeze(1)   # shape: (N, 1) to match model output

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ─────────────────────────────────────────────
# MODEL  (your architecture, unchanged)
# ─────────────────────────────────────────────
class DeepfakeDetectorMLP(nn.Module):
    def __init__(self, input_size=390):
        super(DeepfakeDetectorMLP, self).__init__()
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Layer 3
            nn.Linear(128, 64),
            nn.ReLU(),
            # Output: 1 neuron  (Real=0, Fake=1)
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    """
    One full pass over loader.
    Pass optimizer=None for validation / test (eval mode, no gradients).
    Returns (avg_loss, accuracy, all_labels, all_probs).
    """
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
    plt.savefig("/kaggle/working/training_history.png", dpi=150)
    plt.show()
    print("→ Saved training_history.png")


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix – Test Set")
    plt.tight_layout()
    plt.savefig("/kaggle/working/confusion_matrix.png", dpi=150)
    plt.show()
    print("→ Saved confusion_matrix.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── 1. Load data ───────────────────────────
    print("Loading parquet files...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df   = pd.read_parquet(VAL_PATH)
    test_df  = pd.read_parquet(TEST_PATH)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    
    #map labels to integers
    label_map = {'real': 0, 'fake': 1}
    for df in [train_df, val_df, test_df]:
        df.dropna()
        df['label'] = df['label'].map(label_map)

    
    
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    feature_cols = [c for c in train_df.columns if c not in ("label", "filename")]
    input_size   = len(feature_cols)
    print(f"  Input features: {input_size}")
    print(f"  Label distribution (train):\n{train_df['label'].value_counts().to_string()}\n")

    # ── 2. Scale (fit ONLY on train) ───────────
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    # CRITICAL: Fill NaNs created by StandardScaler (division by zero)
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    val_df[feature_cols]   = val_df[feature_cols].fillna(0)
    test_df[feature_cols]  = test_df[feature_cols].fillna(0)

    # ── 3. DataLoaders ─────────────────────────
    train_loader = DataLoader(AudioFeatureDataset(train_df), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(AudioFeatureDataset(val_df),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(AudioFeatureDataset(test_df),  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── 4. Model, loss, optimizer ──────────────
    model     = DeepfakeDetectorMLP(input_size=input_size).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # halve LR if val loss doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # ── 5. Training loop ───────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss      = float("inf")
    best_weights       = None
    epochs_no_improve  = 0

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  {'LR':>9}")
    print("─" * 65)

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc, _, _  = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _  = run_epoch(model, val_loader,   criterion)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>8.2%}  "
              f"{vl_loss:>9.4f}  {vl_acc:>7.2%}  {lr_now:>9.2e}")

        # early stopping
        if vl_loss < best_val_loss:
            best_val_loss    = vl_loss
            best_weights     = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs).")
                break

    # ── 6. Save best weights ───────────────────
    torch.save({
        "model_state_dict": best_weights,
        "input_size":       input_size,
        "val_loss":         best_val_loss,
        "scaler_mean":      scaler.mean_,
        "scaler_scale":     scaler.scale_,
    }, SAVE_PATH)
    print(f"\n→ Best model saved to {SAVE_PATH}  (val loss: {best_val_loss:.4f})")

    # ── 7. Plots ───────────────────────────────
    plot_history(history)

    # ── 8. Evaluate on test set ────────────────
    model.load_state_dict(best_weights)
    te_loss, te_acc, y_true, y_probs = run_epoch(model, test_loader, criterion)
    y_pred = (y_probs >= THRESHOLD).astype(int)
    auc    = roc_auc_score(y_true, y_probs)

    print("\n" + "═" * 45)
    print("TEST SET RESULTS")
    print("═" * 45)
    print(f"  Loss     : {te_loss:.4f}")
    print(f"  Accuracy : {te_acc:.2%}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    plot_confusion(y_true, y_pred)


if __name__ == "__main__":
    main()
