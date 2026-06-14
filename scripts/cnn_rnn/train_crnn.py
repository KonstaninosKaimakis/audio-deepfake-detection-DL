import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path

from scripts.cnn_rnn.utils import (
    load_config, seed_everything, get_output_paths,
    plot_training_curves, plot_roc_auc, save_results
)
from scripts.cnn_rnn.dataset import build_dataloaders

class CRNNBaseline(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.rnn_input_dim = 128 * 16 
        self.lstm1 = nn.LSTM(input_size=self.rnn_input_dim, hidden_size=128, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.cnn(x)
        
        x = x.permute(0, 3, 1, 2).contiguous()
        batch_size, seq_len, channels, freq = x.size()
        x = x.view(batch_size, seq_len, channels * freq)
        
        x, _ = self.lstm1(x)
        x = self.relu(x)
        x = self.drop1(x)
        
        x, _ = self.lstm2(x)
        x = self.relu(x)
        x = self.drop2(x)
        
        out_last_step = x[:, -1, :]
        return self.classifier(out_last_step)

class Trainer:
    def __init__(self, model, cfg, device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=cfg.training.lr_reduce_factor, patience=cfg.training.lr_reduce_patience
        )
        self.history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}

    def _preds(self, logits):
        return (torch.sigmoid(logits) > 0.5).long()

    def _run_epoch(self, loader, train: bool):
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

    def fit(self, train_loader, val_loader, ckpt_dir, tag):
        best_val_loss = float("inf")
        patience_counter = 0
        ckpt_path = ckpt_dir / f"best_{tag}.pt"

        print(f"\n{'Ep':>4}  {'T-Loss':>8}  {'T-Acc':>7}  {'V-Loss':>8}  {'V-Acc':>7}")
        print("─" * 48)

        for epoch in range(self.cfg.training.epochs):
            t_loss, t_acc = self._run_epoch(train_loader, train=True)
            v_loss, v_acc = self._run_epoch(val_loader, train=False)
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

            if patience_counter >= self.cfg.training.early_stop_patience:
                print(f"\n  Early stopping at epoch {epoch + 1}.")
                break

        print(f"\n  Best val loss : {best_val_loss:.4f}")
        print(f"  Checkpoint    : {ckpt_path}\n")
        return ckpt_path

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training_configs/crnn.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir, plots_dir, tag = get_output_paths(cfg)
    print(f"Device : {device}\nTag    : {tag}\nOutput : {cfg.training.output_dir}\n")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    print("=" * 60)
    print("STEP 2 — MODEL")
    print("=" * 60)
    model = CRNNBaseline()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {cfg.model.name}  —  {n_params:,} parameters\n")

    print("=" * 60)
    print("STEP 3 — TRAINING")
    print("=" * 60)
    trainer = Trainer(model, cfg, device)
    ckpt_path = trainer.fit(train_loader, val_loader, ckpt_dir, tag)

    print("=" * 60)
    print("STEP 4 — TRAINING CURVES")
    print("=" * 60)
    plot_training_curves(plots_dir, tag, trainer.history)

    print("=" * 60)
    print("STEP 5 — TEST EVALUATION")
    print("=" * 60)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"  Loaded best weights from {ckpt_path}")
    test_acc, preds, labels, probs = trainer.evaluate(test_loader)

    print("=" * 60)
    print("STEP 6 — SAVING RESULTS")
    print("=" * 60)
    auc = plot_roc_auc(plots_dir, tag, labels, probs)
    save_results(ckpt_dir, tag, trainer.history, test_acc, preds, labels, probs, auc)