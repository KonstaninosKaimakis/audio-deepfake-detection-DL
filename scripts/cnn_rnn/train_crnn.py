import argparse
import torch
from pathlib import Path

from scripts.cnn_rnn.models import CRNNBaseline
from scripts.cnn_rnn.utils import (
    load_config, seed_everything,
    plot_training_curves, plot_roc_auc, save_results,
    save_rich_checkpoint, Trainer,
    make_tag_crnn, plot_comparisons,
)
from scripts.cnn_rnn.dataset import build_dataloaders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training_configs/crnn.yaml")
    args, _ = parser.parse_known_args()

    cfg = load_config(args.config)
    seed_everything(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = cfg.training.output_dir

    print(f"Device     : {device}")
    print(f"Experiments: {len(cfg.experiments)}\n")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    results = []
    for i, exp in enumerate(cfg.experiments, 1):
        tag = make_tag_crnn(exp)
        print(f"\n{'=' * 70}")
        print(f"  EXPERIMENT {i}/{len(cfg.experiments)}: {tag}")
        print(f"{'=' * 70}")

        seed_everything(cfg.training.seed)

        exp_dir   = Path(output_dir) / "experiments" / tag
        ckpt_dir  = exp_dir / "checkpoints"
        plots_dir = exp_dir / "plots"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        model_config = {
            "cnn_depth":      exp["cnn_depth"],
            "base_channels":  exp["base_channels"],
            "num_rnn_layers": exp["num_rnn_layers"],
            "rnn_hidden":     exp["rnn_hidden"],
            "dropout":        exp["dropout"],
            "freq_bins":      128,
        }
        model = CRNNBaseline(**model_config)
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        training_cfg = {
            "epochs":              cfg.training.epochs,
            "early_stop_patience": cfg.training.early_stop_patience,
            "lr_reduce_factor":    cfg.training.lr_reduce_factor,
            "lr_reduce_patience":  cfg.training.lr_reduce_patience,
        }
        trainer = Trainer(model, exp["lr"], exp["weight_decay"], training_cfg, device)
        ckpt_path, best_val_loss, epochs_trained = trainer.fit(
            train_loader, val_loader, ckpt_dir / f"best_{tag}.pt"
        )

        plot_training_curves(plots_dir, tag, trainer.history)

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        test_acc, preds, labels, probs = trainer.evaluate(test_loader)

        auc = plot_roc_auc(plots_dir, tag, labels, probs)
        save_results(ckpt_dir, tag, trainer.history, test_acc, preds, labels, probs, auc)
        save_rich_checkpoint(ckpt_dir, tag, model, model_config, CRNNBaseline,
                             best_val_loss, epochs_trained, experiment=exp)

        results.append({
            "tag":            tag,
            "test_acc":       round(test_acc, 6),
            "auc":            round(auc, 6),
            "best_val_loss":  round(best_val_loss, 6),
            "epochs_trained": epochs_trained,
            **exp,
        })

    print(f"\n{'=' * 70}")
    print("  COMPARISON PLOTS")
    print(f"{'=' * 70}")
    plot_comparisons(results, output_dir, "CRNN")
