import argparse
import os
import json
import copy
import sys
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score

# Notice: We no longer import from model.py!
from dataset import AudioFeatureDataset
from utils import load_config, seed_everything, build_dataloaders, plot_history, plot_roc_curve


# ==============================================================================
# 1. MODEL ARCHITECTURE
# ==============================================================================
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


# ==============================================================================
# 2. TRAINING EPOCH HELPER
# ==============================================================================
def run_epoch(model, loader, criterion, threshold, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_probs, all_labels = [], []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
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
    preds      = (all_probs >= threshold).astype(int)
    accuracy   = (preds == all_labels.astype(int)).mean()
    avg_loss   = total_loss / len(all_labels)

    return avg_loss, accuracy, all_labels, all_probs


# ==============================================================================
# 3. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to pipeline configuration file")
    args = parser.parse_args()
 
    cfg = load_config(args.config)
    seed_everything(cfg.training.seed)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running pipeline on: {device}")

    # Set up Result Directories
    checkpoint_dir = os.path.join(cfg.training.output_dir, "checkpoints")
    plot_dir = os.path.join(cfg.training.output_dir, "plots")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
 
    print("\n--- Initializing Data Loaders ---")
    train_loader, val_loader, test_loader, input_size, scaler, feature_cols = build_dataloaders(cfg, AudioFeatureDataset)
    
    print("\n--- Initializing Network ---")
    model_config = {
        "input_size": input_size,
        "hidden_sizes": cfg.model.hidden_sizes,
        "dropout": cfg.model.dropout,
    }
    # Instantiate the model defined directly in this script
    model = DeepfakeDetectorMLP(**model_config).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=cfg.training.lr_factor, patience=cfg.training.lr_patience)
 
    best_path = os.path.join(checkpoint_dir, "best_simpleNN.pt")
    history_path = os.path.join(checkpoint_dir, "history_simpleNN.json")
    metrics_path = os.path.join(checkpoint_dir, "metrics_simpleNN.txt")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0
 
    print("\n--- Starting Training Loop ---")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}  {'LR':>9}")
    print("─" * 65)

    for epoch in range(1, cfg.training.epochs + 1):
        tr_loss, tr_acc, _, _  = run_epoch(model, train_loader, criterion, cfg.model.threshold, device, optimizer)
        vl_loss, vl_acc, _, _  = run_epoch(model, val_loader,   criterion, cfg.model.threshold, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(float(tr_loss))
        history["val_loss"].append(float(vl_loss))
        history["train_acc"].append(float(tr_acc))
        history["val_acc"].append(float(vl_acc))

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>8.2%}  {vl_loss:>9.4f}  {vl_acc:>7.2%}  {lr_now:>9.2e}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.training.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {cfg.training.patience} epochs).")
                break

    # Save Checkpoint & History
    torch.save({
        "model_state_dict": best_weights,
        "model_config":     model_config,
        "feature_cols":     feature_cols,
        "seed":             cfg.training.seed,
        "pytorch_version":  torch.__version__,
        "python_version":   sys.version,
        "val_loss":         best_val_loss,
        "scaler_mean":      scaler.mean_,
        "scaler_scale":     scaler.scale_,
    }, best_path)
    print(f"\n→ Best model saved to {best_path}")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"→ Saved history log to {history_path}")

    # Generate Plots
    plot_history(history, plot_dir)

    print("\n--- Running Final Evaluation ---")
    model.load_state_dict(best_weights)
    te_loss, te_acc, y_true, y_probs = run_epoch(model, test_loader, criterion, cfg.model.threshold, device)
    y_pred = (y_probs >= cfg.model.threshold).astype(int)
    auc_score = roc_auc_score(y_true, y_probs)

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
    with open(metrics_path, "w") as f:
        f.write(metrics_output)
    print(f"→ Saved performance metrics to {metrics_path}")

    plot_roc_curve(y_true, y_probs, plot_dir)


if __name__ == "__main__":
    main()