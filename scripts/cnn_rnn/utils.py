import os
import json
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
from types import SimpleNamespace
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

def _to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    return d

def load_config(path):
    with open(path) as f:
        return _to_namespace(yaml.safe_load(f))

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_output_paths(cfg):
    out_dir = Path(cfg.training.output_dir)
    tag = f"{cfg.model.name}_{cfg.model.spec_type}"
    
    ckpt_dir = out_dir / "checkpoints" / tag
    plots_dir = out_dir / "plots" / tag
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    return ckpt_dir, plots_dir, tag

def _save_plot(plots_dir, fig: plt.Figure, name: str):
    p = plots_dir / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {p}")

def plot_training_curves(plots_dir, tag, history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training curves — {tag}", fontsize=13, fontweight="bold")

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
    _save_plot(plots_dir, fig, f"training_curves_{tag}.png")

def plot_roc_auc(plots_dir, tag, labels: list, probs: list) -> float:
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2.5, color="steelblue", label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.10, color="steelblue")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {tag}", fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_plot(plots_dir, fig, f"roc_auc_{tag}.png")
    return auc

class Trainer:
    def __init__(self, model, lr, weight_decay, cfg, device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min",
            factor=cfg["lr_reduce_factor"], patience=cfg["lr_reduce_patience"],
        )
        self.history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}

    def _preds(self, logits):
        return (torch.sigmoid(logits) > 0.5).long()

    def _run_epoch(self, loader, train):
        self.model.train() if train else self.model.eval()
        total_loss = correct = total = 0
        desc = "  Train" if train else "  Val  "
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for specs, labels in tqdm(loader, desc=desc, leave=False):
                specs = specs.to(self.device)
                labs_f = labels.float().unsqueeze(1).to(self.device)
                if train:
                    self.optimizer.zero_grad()
                logits = self.model(specs)
                loss = self.criterion(logits, labs_f)
                if train:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
                correct += (self._preds(logits) == labs_f.long()).sum().item()
                total += labs_f.size(0)
        return total_loss / len(loader), correct / total

    def fit(self, train_loader, val_loader, ckpt_path):
        best_val_loss = float("inf")
        patience_counter = 0
        print(f"\n{'Ep':>4}  {'T-Loss':>8}  {'T-Acc':>7}  {'V-Loss':>8}  {'V-Acc':>7}")
        print("─" * 48)
        for epoch in range(self.cfg["epochs"]):
            t_loss, t_acc = self._run_epoch(train_loader, True)
            v_loss, v_acc = self._run_epoch(val_loader, False)
            self.scheduler.step(v_loss)
            self.history["train_loss"].append(t_loss)
            self.history["train_acc"].append(t_acc)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)
            marker = ""
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), ckpt_path)
                marker = "  ✓ best"
            else:
                patience_counter += 1
            print(f"{epoch+1:>4}  {t_loss:>8.4f}  {t_acc:>7.4f}  {v_loss:>8.4f}  {v_acc:>7.4f}{marker}")
            if patience_counter >= self.cfg["early_stop_patience"]:
                print(f"\n  Early stopping at epoch {epoch + 1}.")
                break
        print(f"\n  Best val loss : {best_val_loss:.4f}")
        print(f"  Checkpoint    : {ckpt_path}\n")
        return ckpt_path, best_val_loss, len(self.history["train_loss"])

    def evaluate(self, test_loader):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for specs, labels in tqdm(test_loader, desc="  Test ", leave=False):
                specs = specs.to(self.device)
                logits = self.model(specs)
                probs = torch.sigmoid(logits).squeeze(1)
                preds = (probs > 0.5).long()
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())
        acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        return acc, all_preds, all_labels, all_probs


def save_rich_checkpoint(ckpt_dir, tag, model, model_config, model_class,
                         best_val_loss, epochs_trained, experiment=None):
    import inspect

    try:
        model_class_src = inspect.getsource(model_class)
    except OSError:
        model_class_src = "# Source unavailable (interactive session)"

    ckpt_path = ckpt_dir / f"best_{tag}.pt"
    torch.save({
        "model_state_dict":   model.state_dict(),
        "model_config":       model_config,
        "model_class_name":   model_class.__name__,
        "model_class_source": model_class_src,
        "experiment":         experiment,
        "val_loss":           best_val_loss,
        "epochs_trained":     epochs_trained,
    }, ckpt_path)
    print(f"  Checkpoint → {ckpt_path}")

    model_py = ckpt_dir / f"model_class_{tag}.py"
    with open(model_py, "w") as f:
        f.write("import torch\nimport torch.nn as nn\n\n\n")
        f.write(model_class_src)
        f.write(f"\n\n# Reconstruction for this experiment:\n")
        f.write(f"# model = {model_class.__name__}(**{model_config})\n")
        f.write(f"# checkpoint = torch.load('best_{tag}.pt', map_location='cpu')\n")
        f.write(f"# model.load_state_dict(checkpoint['model_state_dict'])\n")
    print(f"  Model class → {model_py}")

    return ckpt_path


def _sci(v):
    return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")


def make_tag_cnn(exp):
    return (
        f"CNN_d{exp['depth']}_ch{exp['base_channels']}"
        f"_lr{_sci(exp['lr'])}_do{exp['dropout']}_wd{_sci(exp['weight_decay'])}"
    )


def make_tag_rnn(exp):
    return (
        f"RNN_l{exp['num_layers']}_h{exp['hidden_size']}"
        f"_lr{_sci(exp['lr'])}_do{exp['dropout']}_wd{_sci(exp['weight_decay'])}"
    )


def make_tag_crnn(exp):
    return (
        f"CRNN_cd{exp['cnn_depth']}_ch{exp['base_channels']}"
        f"_rl{exp['num_rnn_layers']}_rh{exp['rnn_hidden']}"
        f"_lr{_sci(exp['lr'])}_do{exp['dropout']}_wd{_sci(exp['weight_decay'])}"
    )


def plot_comparisons(results, output_dir, title_prefix):
    import csv
    comp_dir = Path(output_dir) / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    def _hbar(data_pairs, xlabel, title, color_best, color_rest, filename):
        pairs  = sorted(data_pairs, key=lambda x: x[1], reverse=True)
        tags   = [p[0] for p in pairs]
        vals   = [p[1] for p in pairs]
        colors = [color_best if i == 0 else color_rest for i in range(len(tags))]
        fig, ax = plt.subplots(figsize=(12, max(4, len(tags) * 0.55)))
        bars = ax.barh(tags, vals, color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0, min(max(vals) * 1.15, 1.05))
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        p = comp_dir / filename
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Comparison → {p}")

    _hbar(
        [(r["tag"], r["auc"]) for r in results],
        "ROC-AUC", f"{title_prefix} Experiments — ROC-AUC Comparison",
        "steelblue", "lightsteelblue", "roc_auc_comparison.png",
    )
    _hbar(
        [(r["tag"], r["test_acc"]) for r in results],
        "Test Accuracy", f"{title_prefix} Experiments — Accuracy Comparison",
        "coral", "lightsalmon", "accuracy_comparison.png",
    )

    csv_path = comp_dir / "metrics_summary.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r["auc"], reverse=True))
    print(f"  CSV      → {csv_path}")


def save_results(ckpt_dir, tag, history: dict, test_acc: float,
                 preds: list, labels: list, probs: list, auc: float):
    hist_path = ckpt_dir / f"history_{tag}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History  → {hist_path}")

    report = classification_report(labels, preds, target_names=["Real", "Fake"], digits=4)
    metrics = (
        f"Model        : {tag}\n"
        f"Test accuracy: {test_acc:.4f}\n"
        f"AUC-ROC      : {auc:.4f}\n\n"
        f"{report}"
    )
    print("\n" + metrics)

    report_path = ckpt_dir / f"metrics_{tag}.txt"
    with open(report_path, "w") as f:
        f.write(metrics)
    print(f"  Metrics  → {report_path}")