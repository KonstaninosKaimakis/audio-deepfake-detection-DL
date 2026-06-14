import os
import random
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
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
    # Ensure dataloader workers are explicitly seeded
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

    # Fit scaler on train, transform on val/test
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
    
    train_loader = DataLoader(dataset_class(train_df), batch_size=cfg.training.batch_size, shuffle=True,  num_workers=cfg.data.num_workers, pin_memory=pin, worker_init_fn=seed_worker, generator=worker_gen)
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