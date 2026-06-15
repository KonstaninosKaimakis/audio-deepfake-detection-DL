import torch
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from transformers import WavLMModel
from scripts.wavlm.wavlm_dataset import WavLMDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from scripts.wavlm.utils import load_config, seed_everything, build_dataloaders, build_model, build_optimizer, build_scheduler
from scripts.wavlm.wavlm_extract import load_or_extract, cached_loader


class WavLMClassifier(nn.Module):
    def __init__(
        self,
        model='microsoft/wavlm-base-plus',
        hidden_size = 256,
        dropout = 0.5,
        freeze_backbone = True
    ):
        super().__init__()

        self.wavlm_model = WavLMModel.from_pretrained(model)

        if freeze_backbone:
            # frozen probe (cached features): train only head + layer_weights
            for p in self.wavlm_model.parameters():
                p.requires_grad = False
        # else: full fine-tune -> everything trainable, including the conv feature encoder

        # extract wavlm hidden size from model configuration
        wavlm_hidden_size = self.wavlm_model.config.hidden_size

        # one learnable weight per hidden state (embedding + 12 transformer layers = 13)
        num_layers = self.wavlm_model.config.num_hidden_layers + 1
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        # self.head = nn.Sequential(
        #     nn.Linear(wavlm_hidden_size, hidden_size),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size, 2),
        # )

        self.head = nn.Linear(wavlm_hidden_size, 2)

    def mean_pooling(self, hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        mean = torch.sum(hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return mean

    def weighted_sum(self, layer_feats):
        # layer_feats: (B, num_layers, 768) -> (B, 768)
        norm_weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1)
        return torch.sum(layer_feats * norm_weights, dim=1)

    def forward(self, input_values, attention_mask=None):
        hidden_states = self.wavlm_model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states

        if attention_mask is not None:
            frame_mask = self.wavlm_model._get_feature_vector_attention_mask(
                hidden_states[0].shape[1],
                attention_mask=attention_mask
            )
            pooled = [self.mean_pooling(h, frame_mask) for h in hidden_states]
        else:
            pooled = [h.mean(dim=1) for h in hidden_states]

        layer_feats = torch.stack(pooled, dim=1)   # (B, num_layers, 768)
        logits = self.head(self.weighted_sum(layer_feats))

        return logits


def train_model(model, loader, optimizer, device):
    model.head.train()
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_loss = 0.0
    correct = 0
    n = 0

    for feats, y in loader:
        feats, y = feats.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model.head(model.weighted_sum(feats))
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.head.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for feats, y in loader:
        feats, y = feats.to(device), y.to(device)
        logits = model.head(model.weighted_sum(feats))
        total_loss += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
 
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
 
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
 
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1])
    auc = roc_auc_score(all_labels, all_probs)

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
 
    train_loader, val_loader, test_loader = build_dataloaders(cfg, WavLMDataset)
    model = build_model(cfg, WavLMClassifier, device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    Path(cfg.training.output_dir).mkdir(parents=True, exist_ok=True)

    cache_dir = Path(cfg.training.output_dir) / "cache"
    print("\nPreparing feature cache...")
    train_feats, train_labels = load_or_extract(model, train_loader, cache_dir / "train.pt", device)
    val_feats,   val_labels   = load_or_extract(model, val_loader,   cache_dir / "val.pt",   device)
    test_feats,  test_labels  = load_or_extract(model, test_loader,  cache_dir / "test.pt",  device)

    train_loader = cached_loader(train_feats, train_labels, cfg.training.batch_size, shuffle=True)
    val_loader   = cached_loader(val_feats,   val_labels,   cfg.training.batch_size, shuffle=False)
    test_loader  = cached_loader(test_feats,  test_labels,  cfg.training.batch_size, shuffle=False)

    best_acc = 0.0
    best_path = f"{cfg.training.output_dir}/best_head.pt"

    for epoch in range(1, cfg.training.epochs + 1):
        tr_loss, tr_acc = train_model(model, train_loader, optimizer, device)
        val = evaluate_model(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f}  acc {tr_acc:.3f} | "
              f"val loss {val['loss']:.4f}  acc {val['acc']:.3f}  auc {val['auc']:.3f}")
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            torch.save({"head": model.head.state_dict(),
                        "layer_weights": model.layer_weights.detach().cpu()}, best_path)
            print(f"  saved best (val_acc={best_acc:.3f})")

    # final evaluation with the best-val checkpoint: in-domain (FoR) + cross-dataset (ITW)
    ckpt = torch.load(best_path, map_location=device)
    model.head.load_state_dict(ckpt["head"])
    model.layer_weights.data = ckpt["layer_weights"].to(device)

    # learned per-layer importance: raw logits and the softmax weights actually used
    raw = model.layer_weights.tolist()
    norm = torch.softmax(model.layer_weights, dim=0).tolist()
    print("\nLearned layer weights:")
    for i, (r, w) in enumerate(zip(raw, norm)):
        print(f"  layer {i:02d}: raw {r:+.4f}  softmax {w:.4f}")

    def _report(name, m):
        print(f"\n{name}:")
        print(f"  loss {m['loss']:.4f} | acc {m['acc']:.3f} | auc {m['auc']:.4f}")
        print(f"  real:  precision {m['precision_real']:.3f} | recall {m['recall_real']:.3f} | f1 {m['f1_real']:.3f}")
        print(f"  fake:  precision {m['precision_fake']:.3f} | recall {m['recall_fake']:.3f} | f1 {m['f1_fake']:.3f}")

    _report("FoR validation (in-domain)", evaluate_model(model, val_loader, device))
    _report("ITW (cross-dataset) test",   evaluate_model(model, test_loader, device))