import os
import random
import yaml
import numpy as np
import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader
from scripts.wavlm.wavlm_augment import WavLMAugmentationPipeline
from transformers import get_cosine_schedule_with_warmup

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


def compute_eer(labels, scores):
    # Equal Error Rate: the operating point where FPR == FNR. Threshold-independent, so it's the
    # labels: 0/1 (1 = fake / positive); scores: positive-class probability.
    from sklearn.metrics import roc_curve
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def build_augmentation(cfg):
    # construct the online augmentation pipeline from cfg.augment (or None if disabled/absent)
    aug_cfg = getattr(cfg, "augment", None)
    if aug_cfg is None or not getattr(aug_cfg, "enabled", False):
        return None
    
    return WavLMAugmentationPipeline(
        sample_rate=cfg.data.sample_rate,
        musan_dir=aug_cfg.musan_dir,
        rir_dir=aug_cfg.rir_dir,
        snr_db_range=tuple(aug_cfg.snr_db_range),
        p_noise=aug_cfg.p_noise,
        p_rir=aug_cfg.p_rir,
        p_codec=0.0,   # codec is applied offline now (pre-rendered variants via the dataset)
    )


def build_dataloaders(cfg, dataset_class, max_duration_sec=None, batch_size=None,
                      augment_train=False, num_workers=None):
    # overrides let the fine-tune track use shorter clips / smaller batch than the probe
    max_duration_sec = max_duration_sec if max_duration_sec is not None else cfg.data.max_duration_sec
    batch_size       = batch_size       if batch_size       is not None else cfg.training.batch_size

    # num_workers=None keeps the probe's 8/4/4; the fine-tune lowers it so fewer codec
    train_workers = num_workers if num_workers is not None else 8
    eval_workers  = max(1, train_workers // 2) if num_workers is not None else 4

    # augmentation is TRAIN-only; val/test are never augmented
    train_aug = build_augmentation(cfg) if augment_train else None

    # offline codec is TRAIN-only too: pre-rendered variants swapped in by the dataset (prob p_codec)
    aug_cfg = getattr(cfg, "augment", None)
    if augment_train and aug_cfg is not None and getattr(aug_cfg, "enabled", False):
        codec_cache_dir = getattr(aug_cfg, "codec_cache_dir", None)
        p_codec         = getattr(aug_cfg, "p_codec", 0.0)
    else:
        codec_cache_dir, p_codec = None, 0.0

    def _ds(split_dir, augment=None, codec_cache_dir=None, p_codec=0.0):
        return dataset_class(
            split_dir,
            max_duration_sec=max_duration_sec,
            sample_rate=cfg.data.sample_rate,
            augment=augment,
            codec_cache_dir=codec_cache_dir,
            p_codec=p_codec
        )

    train_ds = _ds(
        cfg.data.train_dir, 
        augment=train_aug,           
        codec_cache_dir=codec_cache_dir, 
        p_codec=p_codec
        )                                                              # FoR (in-domain)
    val_ds = _ds(cfg.data.val_dir)                                   # FoR (in-domain)
    test_ds = _ds(cfg.data.test_dir)                                  # ITW (cross-dataset)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=train_workers, pin_memory=True)
    val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=eval_workers,  pin_memory=True)
    test_loader = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=eval_workers,  pin_memory=True)
    return train_loader, val_loader, test_loader


def build_model(cfg, model_class, device, freeze_backbone=True):
    model = model_class(
        cfg.model.name, 
        hidden_size=cfg.model.hidden,
        dropout=cfg.model.dropout, 
        freeze_backbone=freeze_backbone
        ).to(device)
    return model


def apply_partial_freeze(model, n_trainable_layers=None, freeze_feature_encoder=True):
    wavlm = model.wavlm_model

    if freeze_feature_encoder:
        wavlm.freeze_feature_encoder()   # conv feature encoder: low-level acoustics, overfits fast

    if n_trainable_layers is not None:
        layers = wavlm.encoder.layers
        n_freeze = max(0, len(layers) - n_trainable_layers)
        for layer in layers[:n_freeze]:
            for p in layer.parameters():
                p.requires_grad = False
    return model


# def build_optimizer(cfg, model):
#     return torch.optim.AdamW([
#         {"params": model.wavlm.parameters(), "lr": cfg.training.lr * 0.1},
#         {"params": model.head.parameters(),  "lr": cfg.training.lr},
#     ], weight_decay=1e-2)

def build_optimizer(cfg, model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=cfg.training.lr, weight_decay=1e-2)

def build_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)


def build_finetune_optimizer(cfg, model):
    # discriminative LRs: low LR for the pretrained backbone, higher LR for the fresh head
    backbone_params = [p for p in model.wavlm_model.parameters() if p.requires_grad]
    head_params = list(model.head.parameters()) + [model.layer_weights]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.finetune.lr_backbone},
        {"params": head_params, "lr": cfg.finetune.lr_head},
    ], weight_decay=cfg.finetune.weight_decay)


def build_finetune_scheduler(cfg, optimizer, num_training_steps):
    
    num_warmup_steps = int(cfg.finetune.warmup_ratio * num_training_steps)
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)