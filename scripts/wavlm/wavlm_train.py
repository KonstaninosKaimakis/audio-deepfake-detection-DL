import torch
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from transformers import WavLMModel
from scripts.wavlm.wavlm_dataset import WavLMDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from scripts.wavlm.utils import load_config, seed_everything, build_dataloaders, build_model, build_optimizer, build_scheduler


class WavLMClassifier(nn.Module):
    def __init__(
        self,
        model='microsoft/wavlm-base-plus',
        hidden_size = 256,
        dropout = 0.5
    ):
        super().__init__()

        self.wavlm_model = WavLMModel.from_pretrained(model)
        
        # freeze wavlm weights -> only train classifier head
        # self.wavlm_model.freeze_feature_encoder()
        
        # freeze everything
        for p in self.wavlm_model.parameters():
            p.requires_grad = False

        # extract wavlm hidden size from model configuration
        wavlm_hidden_size = self.wavlm_model.config.hidden_size
        
        self.head = nn.Sequential(
            nn.Linear(wavlm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def mean_pooling(self, hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        mean = torch.sum(hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return mean
    
    def forward(self, input_values, attention_mask=None):
        last_transformer_layer_output = self.wavlm_model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).last_hidden_state

        if attention_mask is not None:
            frame_mask = self.wavlm_model._get_feature_vector_attention_mask(
                last_transformer_layer_output.shape[1],
                attention_mask=attention_mask
            )
            pooled = self.mean_pooling(last_transformer_layer_output, frame_mask)
        
        else:
            pooled = last_transformer_layer_output.mean(dim=1)
        
        logits = self.head(pooled)

        return logits


def train(model, loader, optimizer, device):
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    n = 0

    for x, mask, y in loader:
        x=x.to(device)
        mask=mask.to(device)
        y=y.to(device)

        optimizer.zero_grad()
        logits = model(x, attention_mask=mask)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
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
        logits = model(x, attention_mask=mask)
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
    best_acc = 0.0
 
    best_path = f"{cfg.training.output_dir}/best.pt"
    for epoch in range(1, cfg.training.epochs + 1):
        tr_loss, tr_acc = train(model, train_loader, optimizer, device)
        val = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.3f} | "
              f"val loss {val['loss']:.4f} acc {val['acc']:.3f} auc {val['auc']:.3f}")
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            torch.save(model.state_dict(), best_path)
            print(f"  saved best (val_acc={best_acc:.3f})")

    # final evaluation with the best-val checkpoint: in-domain (FoR) + cross-dataset (ITW)
    model.load_state_dict(torch.load(best_path, map_location=device))

    def _report(name, m):
        print(f"\n{name}:")
        print(f"  loss {m['loss']:.4f} | acc {m['acc']:.3f} | auc {m['auc']:.4f}")
        print(f"  real:  precision {m['precision_real']:.3f} | recall {m['recall_real']:.3f} | f1 {m['f1_real']:.3f}")
        print(f"  fake:  precision {m['precision_fake']:.3f} | recall {m['recall_fake']:.3f} | f1 {m['f1_fake']:.3f}")

    _report("FoR validation (in-domain)", evaluate(model, val_loader, device))
    _report("ITW (cross-dataset) test",   evaluate(model, test_loader, device))