import json
import contextlib

import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from datetime import datetime

from scripts.wavlm.wavlm_dataset import WavLMDataset
from scripts.wavlm.wavlm_train import WavLMClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from scripts.wavlm.utils import (
    load_config, 
    seed_everything, 
    build_dataloaders, 
    build_model,
    apply_partial_freeze,
    build_finetune_optimizer,
    build_finetune_scheduler,
    compute_eer,
    config_to_dict,
)


def _autocast(device):
    # bf16 autocast on CUDA only (no GradScaler needed for bf16); a no-op on CPU
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_loss = 0.0
    correct = 0
    n = 0

    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        with _autocast(device):
            logits = model(x, attention_mask=mask)
            loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        with _autocast(device):
            logits = model(x, attention_mask=mask)
        total_loss += criterion(logits.float(), y).item() * y.size(0)
        n += y.size(0)

        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1])
    auc = roc_auc_score(all_labels, all_probs)
    eer = compute_eer(all_labels, all_probs)

    return {
        "loss":           total_loss / n,
        "acc":            acc,
        "precision_real": precision[0],
        "recall_real":    recall[0],
        "f1_real":        f1[0],
        "precision_fake": precision[1],
        "recall_fake":    recall[1],
        "f1_fake":        f1[1],
        "auc":            auc,
        "eer":            eer,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training_configs/wavlm_base_plus.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.training.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # raw-audio dataloaders (NO cache: backbone weights change every step)
    # augment_train=True applies online augmentation to the TRAIN split only
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg,
        WavLMDataset,
        max_duration_sec=cfg.finetune.max_duration_sec,
        batch_size=cfg.finetune.batch_size,
        augment_train=True,
        num_workers=getattr(cfg.finetune, "num_workers", 4),
    )

    aug = train_loader.dataset.augment
    if aug is not None:
        print("\nAugmentation (train only):")
        for k, v in aug.as_dict().items():
            print(f"  {k}: {v}")

    model = build_model(cfg, WavLMClassifier, device, freeze_backbone=cfg.finetune.freeze_backbone)
    # partial fine-tune: freeze the conv feature encoder + bottom transformer layers (anti-overfit)
    if not cfg.finetune.freeze_backbone:
        n_trainable_layers = getattr(cfg.finetune, "n_trainable_layers", None)
        freeze_feature_encoder = getattr(cfg.finetune, "freeze_feature_encoder", True)
        apply_partial_freeze(model, n_trainable_layers, freeze_feature_encoder)
        print(f"Partial freeze: top {n_trainable_layers} transformer layers trainable, "
              f"freeze_feature_encoder={freeze_feature_encoder}")
    if cfg.finetune.grad_checkpointing:
        model.wavlm_model.gradient_checkpointing_enable()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable / 1e6:.1f} M  (freeze_backbone={cfg.finetune.freeze_backbone})")

    optimizer = build_finetune_optimizer(cfg, model)
    num_training_steps = cfg.finetune.epochs * len(train_loader)
    scheduler = build_finetune_scheduler(cfg, optimizer, num_training_steps)

    Path(cfg.finetune.output_dir).mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_path = f"{cfg.finetune.output_dir}/best_model.pt"

    patience = getattr(cfg.finetune, "patience", None)   # None / <=0 disables early stopping
    epochs_no_improve = 0

    for epoch in range(1, cfg.finetune.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f}  acc {tr_acc:.3f} | "
              f"val loss {val['loss']:.4f}  acc {val['acc']:.3f}  auc {val['auc']:.3f}")
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"  saved best (val_loss={best_loss:.4f})")
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                print(f"  early stop: no val-loss improvement for {patience} epochs "
                      f"(best={best_loss:.4f} @ epoch {epoch - epochs_no_improve})")
                break

    # final evaluation with the best-val checkpoint: in-domain (FoR) + cross-dataset (ITW)
    model.load_state_dict(torch.load(best_path, map_location=device))

    raw = model.layer_weights.tolist()
    norm = torch.softmax(model.layer_weights, dim=0).tolist()
    print("\nLearned layer weights:")
    for i, (r, w) in enumerate(zip(raw, norm)):
        print(f"  layer {i:02d}: raw {r:+.4f}  softmax {w:.4f}")

    def _report(name, m):
        print(f"\n{name}:")
        print(f"  loss {m['loss']:.4f} | acc {m['acc']:.3f} | auc {m['auc']:.4f} | eer {m['eer']:.4f}")
        print(f"  real:  precision {m['precision_real']:.3f} | recall {m['recall_real']:.3f} | f1 {m['f1_real']:.3f}")
        print(f"  fake:  precision {m['precision_fake']:.3f} | recall {m['recall_fake']:.3f} | f1 {m['f1_fake']:.3f}")

    val_metrics  = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    _report("FoR validation (in-domain)", val_metrics)
    _report("ITW (cross-dataset) test",   test_metrics)

    results = {
        "config_file":           args.config,
        "config":                config_to_dict(cfg),
        "freeze_backbone":       cfg.finetune.freeze_backbone,
        "best_val_loss":         best_loss,
        "augmentation":          aug.as_dict() if aug is not None else None,
        "layer_weights_raw":     raw,
        "layer_weights_softmax": norm,
        "for_val":               val_metrics,
        "itw_test":              test_metrics,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = f"{cfg.finetune.output_dir}/metrics_{stamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=float)   # default=float casts any numpy scalars
    print(f"\nSaved metrics -> {metrics_path}")
