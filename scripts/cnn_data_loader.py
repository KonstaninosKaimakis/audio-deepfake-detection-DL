"""
PyTorch DataLoader for CNN training with prepared spectrogram data.

Usage:
    from cnn_data_loader import SpectrogramDataset, get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders(
        prepared_data_dir="./prepared_data_cnn",
        batch_size=32,
        num_workers=4
    )

    for batch_specs, batch_labels in train_loader:
        # batch_specs shape: (batch_size, freq_bins, time_steps)
        # batch_labels shape: (batch_size,)
        pass
"""

import json
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SpectrogramDataset(Dataset):
    """
    PyTorch Dataset for spectrogram data.
    Handles variable-length spectrograms by padding them inside the collate function.
    """

    def __init__(self, npz_file, pad_to=None, normalize=False, scaler_file=None):
        """
        Args:
            npz_file (str): Path to the NPZ file containing spectrograms, labels, filenames.
            pad_to (int, optional): Pad all spectrograms to this time length.
                                    If None, pads to the longest in the batch.
            normalize (bool): Whether to apply additional per-sample normalization.
            scaler_file (str, optional): Path to scaler pickle file for normalization.
        """
        self.npz_file = npz_file
        self.pad_to = pad_to
        self.normalize = normalize
        self.scaler = None

        data = np.load(npz_file, allow_pickle=True)
        self.spectrograms = data['spectrograms']
        self.labels = data['labels']
        self.filenames = data['filenames']

        print(f"Loaded {len(self.spectrograms)} spectrograms from {npz_file}")

        if scaler_file and normalize:
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Loaded scaler from {scaler_file}")

        self.max_time_length = max(spec.shape[1] for spec in self.spectrograms)
        print(f"Max time length: {self.max_time_length}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.spectrograms[idx].astype(np.float32)
        label = int(self.labels[idx])

        if self.normalize and self.scaler is not None:
            spec_shape = spec.shape
            spec_flat = spec.reshape(-1, 1)
            spec_flat = self.scaler.transform(spec_flat)
            spec = spec_flat.reshape(spec_shape).astype(np.float32)

        return torch.tensor(spec), torch.tensor(label)

    def get_spectrogram_shape(self):
        """Return the shape of the first spectrogram (useful for model inspection)."""
        return self.spectrograms[0].shape


def collate_fn_pad(batch, pad_to=None):
    """
    Custom collate function to pad variable-length spectrograms.

    Args:
        batch:  List of (spectrogram, label) tuples.
        pad_to: Pad to this fixed length, or use the max length in the batch if None.

    Returns:
        specs  (batch_size, freq_bins, time_steps)
        labels (batch_size,)
    """
    specs, labels = zip(*batch)

    max_time = pad_to if pad_to else max(s.shape[1] for s in specs)

    padded_specs = []
    for spec in specs:
        pad_amount = max_time - spec.shape[1]
        if pad_amount > 0:
            spec = torch.nn.functional.pad(spec, (0, pad_amount), mode='constant', value=0)
        padded_specs.append(spec)

    specs_batch = torch.stack(padded_specs)
    labels_batch = torch.stack([torch.tensor(l) for l in labels])
    return specs_batch, labels_batch


def get_dataloaders(prepared_data_dir, batch_size=32, num_workers=0, pad_to=None):
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        prepared_data_dir (str): Path to directory created by prepare_data_for_cnn.py.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of workers for data loading.
        pad_to (int, optional): Pad spectrograms to this fixed time length.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    prepared_path = Path(prepared_data_dir)

    config_file = prepared_path / "config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)

    spec_type = config['spectrogram_type']
    scaler_file = prepared_path / f"scaler_{spec_type}.pkl"

    # FIX #8: use functools.partial instead of a lambda defined inside a loop.
    # A lambda inside a loop captures variables by reference; if the loop variable
    # changed later the lambda would silently use the wrong value.  partial binds
    # the value immediately and is also picklable (required for num_workers > 0).
    collate = partial(collate_fn_pad, pad_to=pad_to)

    # FIX #9: pin_memory is only beneficial (and safe) when a CUDA device is
    # available.  Passing pin_memory=True on a CPU-only machine prints a warning
    # in recent PyTorch versions and wastes time allocating pinned memory pages.
    pin = torch.cuda.is_available()

    loaders = {}

    for split in ['training', 'validation', 'testing']:
        split_dir = prepared_path / split
        npz_file = split_dir / f"{split}_spectrograms.npz"

        if npz_file.exists():
            dataset = SpectrogramDataset(
                str(npz_file),
                pad_to=pad_to,
                normalize=False,   # Already normalized during preparation
                scaler_file=None,
            )

            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'training'),
                num_workers=num_workers,
                collate_fn=collate,
                pin_memory=pin,
            )
            print(f"Created {split} DataLoader with {len(dataset)} samples")
        else:
            print(f"Warning: {npz_file} not found, skipping {split} split")

    return (
        loaders.get('training'),
        loaders.get('validation'),
        loaders.get('testing'),
    )


def inspect_batch(loader, num_batches=1):
    """Inspect a few batches to understand data shapes and label distribution."""
    print("\nBatch Inspection:")
    print("-" * 50)
    for batch_idx, (specs, labels) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        print(f"Batch {batch_idx + 1}:")
        print(f"  Specs shape:  {specs.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels (0=Real, 1=Fake): {labels.tolist()}")
        print(f"  Specs dtype:  {specs.dtype}")
        print(f"  Specs range:  [{specs.min():.3f}, {specs.max():.3f}]")
        print()


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        prepared_data_dir="./prepared_data_cnn",
        batch_size=32,
        num_workers=0,
    )
    if train_loader:
        inspect_batch(train_loader, num_batches=2)
