import argparse
import copy
import inspect
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import classification_report

from scripts.nn_manual_features.model import DeepfakeDetectorMLP
from scripts.nn_manual_features.dataset import AudioFeatureDataset
from scripts.nn_manual_features.utils import (
    load_config, seed_everything, build_dataloaders,
    make_tag_nn, plot_training_curves, plot_roc_auc, plot_comparisons,
)


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


def run_experiment(exp, train_loader, val_loader, test_loader,
                   output_dir, cfg, device, input_size, scaler, feature_cols):
    tag = make_tag_nn(exp)
    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT: {tag}")
    print(f"{'=' * 70}")

    exp_dir   = Path(output_dir) / "experiments" / tag
    ckpt_dir  = exp_dir / "checkpoints"
    plots_dir = exp_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_config = {
        "input_size":   input_size,
        "hidden_sizes": exp["hidden_sizes"],
        "dropout":      exp["dropout"],
    }
    model = DeepfakeDetectorMLP(**model_config).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    try:
        model_class_src = inspect.getsource(DeepfakeDetectorMLP)
    except OSError:
        model_class_src = "# Source unavailable (interactive session)"

    model_py = exp_dir / f"model_class_{tag}.py"
    with open(model_py, "w") as f:
        f.write("import torch\nimport torch.nn as nn\n\n\n")
        f.write(model_class_src)
        f.write(f"\n\n# Reconstruction for this experiment:\n")
        f.write(f"# model = DeepfakeDetectorMLP(**{model_config})\n")
        f.write(f"# checkpoint = torch.load('best_{tag}.pt', map_location='cpu')\n")
        f.write(f"# model.load_state_dict(checkpoint['model_state_dict'])\n")
    print(f"  Model class → {model_py}")

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=exp["lr"], weight_decay=exp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg.training.lr_reduce_factor,
        patience=cfg.training.lr_reduce_patience,
    )
    threshold = cfg.training.threshold

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss    = float("inf")
    best_weights     = None
    patience_counter = 0

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  {'LR':>9}")
    print("─" * 65)

    for epoch in range(1, cfg.training.epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, threshold, device, optimizer)
        vl_loss, vl_acc, _, _ = run_epoch(model, val_loader,   criterion, threshold, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(float(tr_loss))
        history["val_loss"].append(float(vl_loss))
        history["train_acc"].append(float(tr_acc))
        history["val_acc"].append(float(vl_acc))

        lr_now = optimizer.param_groups[0]["lr"]
        marker = ""
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_weights  = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = "  ✓"
        else:
            patience_counter += 1
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>8.2%}  "
              f"{vl_loss:>9.4f}  {vl_acc:>7.2%}  {lr_now:>9.2e}{marker}")
        if patience_counter >= cfg.training.early_stop_patience:
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    epochs_trained = epoch

    weights_path = ckpt_dir / f"best_{tag}.pt"
    torch.save({
        "model_state_dict":   best_weights,
        "model_config":       model_config,
        "model_class_name":   "DeepfakeDetectorMLP",
        "model_class_source": model_class_src,
        "experiment":         exp,
        "feature_cols":       feature_cols,
        "val_loss":           best_val_loss,
        "epochs_trained":     epochs_trained,
        "scaler_mean":        scaler.mean_,
        "scaler_scale":       scaler.scale_,
    }, weights_path)
    print(f"  Checkpoint → {weights_path}")

    plot_training_curves(history, plots_dir, tag)

    model.load_state_dict(best_weights)
    _, te_acc, y_true, y_probs = run_epoch(model, test_loader, criterion, threshold, device)
    y_pred = (y_probs >= threshold).astype(int)
    auc = plot_roc_auc(y_true, y_probs, plots_dir, tag)

    hist_path = ckpt_dir / f"history_{tag}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History  → {hist_path}")

    report = classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4)
    metrics_txt = (
        f"Model        : {tag}\n"
        f"Test accuracy: {te_acc:.4f}\n"
        f"AUC-ROC      : {auc:.4f}\n\n"
        f"{report}"
    )
    print("\n" + metrics_txt)
    metrics_path = ckpt_dir / f"metrics_{tag}.txt"
    with open(metrics_path, "w") as f:
        f.write(metrics_txt)
    print(f"  Metrics  → {metrics_path}")

    return {
        "tag":            tag,
        "test_acc":       round(te_acc, 6),
        "auc":            round(auc, 6),
        "best_val_loss":  round(best_val_loss, 6),
        "epochs_trained": epochs_trained,
        **exp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args, _ = parser.parse_known_args()

    cfg = load_config(args.config)
    seed_everything(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device     : {device}")
    print(f"Experiments: {len(cfg.experiments)}\n")

    train_loader, val_loader, test_loader, input_size, scaler, feature_cols = \
        build_dataloaders(cfg, AudioFeatureDataset)

    results = []
    for i, exp in enumerate(cfg.experiments, 1):
        print(f"\nEXPERIMENT {i}/{len(cfg.experiments)}")
        seed_everything(cfg.training.seed)
        result = run_experiment(
            exp, train_loader, val_loader, test_loader,
            cfg.training.output_dir, cfg, device, input_size, scaler, feature_cols,
        )
        results.append(result)

    print(f"\n{'=' * 70}")
    print("  COMPARISON PLOTS")
    print(f"{'=' * 70}")
    plot_comparisons(results, cfg.training.output_dir)


if __name__ == "__main__":
    main()
