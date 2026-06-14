import os
import random

import yaml
import numpy as np
import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader


def _to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    return d


def load_config(path):
    with open(path) as f:
        return _to_namespace(yaml.safe_load(f))


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(cfg, dataset_class):
    def _ds(split_dir):
        return dataset_class(split_dir,
                             max_duration_sec=cfg.data.max_duration_sec,
                             sample_rate=cfg.data.sample_rate)

    train_ds = _ds(cfg.data.train_dir)   # FoR (in-domain)
    val_ds   = _ds(cfg.data.val_dir)     # FoR (in-domain)
    test_ds  = _ds(cfg.data.test_dir)    # ITW (cross-dataset)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.training.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


def build_model(cfg, model_class, device):
    return model_class(cfg.model.name, hidden_size=cfg.model.hidden, dropout=cfg.model.dropout).to(device)


# def build_optimizer(cfg, model):
#     return torch.optim.AdamW([
#         {"params": model.wavlm.parameters(), "lr": cfg.training.lr * 0.1},
#         {"params": model.head.parameters(),  "lr": cfg.training.lr},
#     ], weight_decay=1e-2)

def build_optimizer(cfg, model):
    return torch.optim.AdamW(model.head.parameters(), lr=cfg.training.lr, weight_decay=1e-2)

def build_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)