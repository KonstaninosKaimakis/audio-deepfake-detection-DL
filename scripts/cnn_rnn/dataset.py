import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

LABEL_MAP = {"real": 0, "fake": 1}

def load_precalculated_feature(path: str, extension: str) -> np.ndarray:
    ext = extension.lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".pt":
        arr = torch.load(path, map_location="cpu").numpy()
    elif ext == ".npz":
        with np.load(path) as data:
            arr = data[data.files[0]]
    else:
        raise ValueError(f"Unsupported extension format: {ext}")
    return arr.astype(np.float32)

class PrecalculatedFeatureDataset(Dataset):
    def __init__(self, root: str, split_name: str, extension: str):
        self.extension = extension
        self.samples = []
        
        root_path = Path(root)
        for label_dir in sorted(root_path.iterdir()):
            if not label_dir.is_dir():
                continue
            lname = label_dir.name.lower()
            if lname not in LABEL_MAP:
                continue
            for feat_file in sorted(label_dir.glob(f"*{self.extension}")):
                self.samples.append((feat_file, LABEL_MAP[lname]))

        if not self.samples:
            raise RuntimeError(f"No feature files with extension '{self.extension}' found under {root}")

        counts = {0: 0, 1: 0}
        for _, lbl in self.samples:
            counts[lbl] += 1
        print(f"  [{split_name:>10}]  {len(self.samples)} files  (real: {counts[0]}, fake: {counts[1]})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, label = self.samples[idx]
        spec = load_precalculated_feature(str(feat_path), self.extension)
        return torch.tensor(spec), torch.tensor(label)

def build_dataloaders(cfg):
    print("=" * 60)
    print("STEP 1 — BUILDING DATALOADERS")
    print("=" * 60)

    pin = torch.cuda.is_available()
    ext = cfg.data.feature_extension
    
    train_ds = PrecalculatedFeatureDataset(cfg.data.train_path, "training", ext)
    val_ds   = PrecalculatedFeatureDataset(cfg.data.val_path, "validation", ext)
    test_ds  = PrecalculatedFeatureDataset(cfg.data.test_path, "testing", ext)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, pin_memory=pin)

    print()
    return train_loader, val_loader, test_loader