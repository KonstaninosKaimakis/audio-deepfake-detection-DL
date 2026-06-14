import os
import json
import random
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
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