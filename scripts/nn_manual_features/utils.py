import os
import csv
import random
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import DataLoader

def load_config(path):
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    def _to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
        return d
    return _to_namespace(config_dict)

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloaders(cfg, dataset_class):
    print("Loading parquet files...")
    train_df = pd.read_parquet(cfg.data.train_path)
    val_df   = pd.read_parquet(cfg.data.val_path)
    test_df  = pd.read_parquet(cfg.data.test_path)
    
    label_map = {'real': 0, 'fake': 1}
    for df in [train_df, val_df, test_df]:
        df.dropna(inplace=True)
        df['label'] = df['label'].map(label_map)

    feature_cols = [c for c in train_df.columns if c not in ("label", "filename")]
    input_size   = len(feature_cols)

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    val_df[feature_cols]   = val_df[feature_cols].fillna(0)
    test_df[feature_cols]  = test_df[feature_cols].fillna(0)

    print(f"  [Train] {len(train_df)} files")
    print(f"  [Val]   {len(val_df)} files")
    print(f"  [Test]  {len(test_df)} files")

    pin = torch.cuda.is_available()
    worker_gen = torch.Generator().manual_seed(cfg.training.seed)
    
    train_loader = DataLoader(dataset_class(train_df), batch_size=cfg.training.batch_size, shuffle=True,  num_workers=cfg.data.num_workers, pin_memory=pin)
    val_loader   = DataLoader(dataset_class(val_df),   batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.data.num_workers, pin_memory=pin)
    test_loader  = DataLoader(dataset_class(test_df),  batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.data.num_workers, pin_memory=pin)
    
    return train_loader, val_loader, test_loader, input_size, scaler, feature_cols

def plot_history(history, plot_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(axes, ["loss", "acc"], ["Loss", "Accuracy"]):
        ax.plot(history[f"train_{metric}"], label="Train")
        ax.plot(history[f"val_{metric}"],   label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    save_fig_path = os.path.join(plot_dir, "training_curves_simpleNN.png")
    plt.savefig(save_fig_path, dpi=150)
    plt.close()
    print(f"→ Saved {save_fig_path}")

def plot_roc_curve(y_true, y_probs, plot_dir):
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

    save_fig_path = os.path.join(plot_dir, "roc_auc_simpleNN.png")
    plt.savefig(save_fig_path, dpi=150)
    plt.close()
    print(f"→ Saved {save_fig_path}")


def _sci(v):
    return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")


def make_tag_nn(exp):
    sizes_str = "-".join(str(s) for s in exp["hidden_sizes"])
    return (
        f"NN_h{sizes_str}"
        f"_lr{_sci(exp['lr'])}_do{exp['dropout']}_wd{_sci(exp['weight_decay'])}"
    )


def plot_training_curves(history, plots_dir, tag):
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
    p = Path(plots_dir) / f"training_curves_{tag}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {p}")


def plot_roc_auc(y_true, y_probs, plots_dir, tag):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2.5, color="steelblue", label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.10, color="steelblue")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {tag}", fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = Path(plots_dir) / f"roc_auc_{tag}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {p}")
    return auc


def plot_comparisons(results, output_dir):
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
        "ROC-AUC", "NN Experiments — ROC-AUC Comparison",
        "steelblue", "lightsteelblue", "roc_auc_comparison.png",
    )
    _hbar(
        [(r["tag"], r["test_acc"]) for r in results],
        "Test Accuracy", "NN Experiments — Accuracy Comparison",
        "coral", "lightsalmon", "accuracy_comparison.png",
    )

    csv_path = comp_dir / "metrics_summary.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r["auc"], reverse=True))
    print(f"  CSV      → {csv_path}")