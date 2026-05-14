"""
Prepare WAV files for CNN training.
Converts WAV files to spectrograms and saves them in a format ready for PyTorch DataLoaders.

Directory structure expected (for each split separately):
    training_folder/
    ├── real/
    │   ├── file1.wav
    │   └── file2.wav
    └── fake/
        ├── file3.wav
        └── file4.wav

Usage - Run three times, once for each split:
    # Process training data (fits and saves scaler)
    python prepare_data_for_cnn.py --split_path ./training --split_name training --spectrogram_type mel --output_dir ./prepared_data

    # Process validation data (loads scaler from training run)
    python prepare_data_for_cnn.py --split_path ./validation --split_name validation --spectrogram_type mel --output_dir ./prepared_data

    # Process testing data (loads scaler from training run)
    python prepare_data_for_cnn.py --split_path ./testing --split_name testing --spectrogram_type mel --output_dir ./prepared_data
"""

import os
import argparse
import pickle
import json
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SpectrogramGenerator:
    """Generate different types of spectrograms from audio files."""

    def __init__(self, sr=16000, n_fft=1024, hop_length=256, n_mels=128, n_mfcc=20):
        self.sr = sr
        # FIX #3: n_fft raised from 128 → 1024 for proper frequency resolution.
        # At 16kHz, n_fft=128 gives ~125 Hz/bin (too coarse); 1024 gives ~15.6 Hz/bin.
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

    def mel_spectrogram(self, audio_path):
        """Generate mel-spectrogram from audio file."""
        y, sr = librosa.load(audio_path, sr=self.sr)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db.astype(np.float32)

    def mfcc(self, audio_path):
        """Generate MFCC from audio file."""
        y, sr = librosa.load(audio_path, sr=self.sr)
        mfcc_feature = librosa.feature.mfcc(
            y=y, sr=sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mfcc=self.n_mfcc
        )
        return mfcc_feature.astype(np.float32)

    def spectrogram(self, audio_path):
        """Generate regular (STFT) spectrogram from audio file."""
        y, sr = librosa.load(audio_path, sr=self.sr)
        # FIX #4: was `librosa.feature.stft(...)` which does not exist.
        # Correct function is `librosa.stft(...)`.
        spec = librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_length)
        spec_db = librosa.power_to_db(np.abs(spec) ** 2, ref=np.max)
        return spec_db.astype(np.float32)

    def get_spectrogram(self, audio_path, spec_type):
        """Get spectrogram based on type."""
        if spec_type.lower() == "mel":
            return self.mel_spectrogram(audio_path)
        elif spec_type.lower() == "mfcc":
            return self.mfcc(audio_path)
        elif spec_type.lower() == "spectrogram":
            return self.spectrogram(audio_path)
        else:
            raise ValueError(f"Unknown spectrogram type: {spec_type}")


class DataPreparer:
    """Prepare audio data for CNN training from a single split folder."""

    def __init__(self, split_path, split_name, output_dir, spectrogram_type="mel", sr=16000):
        self.split_path = Path(split_path)
        self.split_name = split_name
        self.output_dir = Path(output_dir)
        self.spectrogram_type = spectrogram_type
        self.sr = sr
        self.spec_gen = SpectrogramGenerator(sr=sr)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.label_map = {"real": 0, "fake": 1}
        self.reverse_label_map = {0: "real", 1: "fake"}
        self.metadata = []

    def find_wav_files(self):
        """Find all WAV files organized by label (real/fake) in the split folder."""
        files_by_label = {}
        for label_dir in self.split_path.iterdir():
            if label_dir.is_dir():
                label_name = label_dir.name
                if label_name.lower() not in self.label_map:
                    print(f"Warning: Unknown label '{label_name}', skipping...")
                    continue
                wav_files = list(label_dir.glob("*.wav"))
                files_by_label[label_name] = wav_files
                print(f"Found {len(wav_files)} files in {label_name}/")
        return files_by_label

    def _load_scaler(self):
        """Load the scaler saved during the training split, if it exists."""
        scaler_file = self.output_dir / f"scaler_{self.spectrogram_type}.pkl"
        if scaler_file.exists():
            with open(scaler_file, "rb") as f:
                scaler = pickle.load(f)
            print(f"Loaded scaler from {scaler_file}")
            return scaler
        return None

    def process_split(self, files_by_label):
        """
        Process all files in the split and save as NPZ.

        FIX #1 & #2: Previously the scaler was re-fit on each individual file
        (one scaler per file, only the first was saved), and val/test splits
        never loaded the saved training scaler.

        Correct approach:
          - Training: collect ALL spectrograms first, fit ONE scaler across the
            full training set, then normalize and save.
          - Val / Test: load the scaler saved from the training run and apply it.
        """
        split_output_dir = self.output_dir / self.split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # --- Pass 1: generate raw spectrograms for every file ---
        raw_spectrograms = []
        labels = []
        filenames = []

        for label_name, wav_files in files_by_label.items():
            label_id = self.label_map[label_name]
            for wav_file in tqdm(wav_files, desc=f"{self.split_name}/{label_name} (generating)"):
                try:
                    spec = self.spec_gen.get_spectrogram(str(wav_file), self.spectrogram_type)
                    raw_spectrograms.append(spec)
                    labels.append(label_id)
                    filenames.append(wav_file.name)
                except Exception as e:
                    print(f"Error processing {wav_file}: {e}")
                    continue

        if not raw_spectrograms:
            return 0

        # --- Pass 2: fit or load scaler, then normalize ---
        if self.split_name == "training":
            # Fit a single scaler on the entire training set at once.
            print("Fitting scaler on full training set...")
            all_values = np.concatenate(
                [spec.reshape(-1) for spec in raw_spectrograms]
            ).reshape(-1, 1)
            scaler = StandardScaler()
            scaler.fit(all_values)

            # Persist scaler for val/test runs
            scaler_file = self.output_dir / f"scaler_{self.spectrogram_type}.pkl"
            with open(scaler_file, "wb") as f:
                pickle.dump(scaler, f)
            print(f"Saved scaler to {scaler_file}")
        else:
            # Load the scaler that was fit during the training run.
            scaler = self._load_scaler()
            if scaler is None:
                print(
                    "Warning: No training scaler found. "
                    "Run the training split first. Skipping normalization."
                )

        normalized_spectrograms = []
        for spec in tqdm(raw_spectrograms, desc=f"{self.split_name} (normalizing)"):
            if scaler is not None:
                original_shape = spec.shape
                spec_flat = spec.reshape(-1, 1)
                spec_flat = scaler.transform(spec_flat)
                spec = spec_flat.reshape(original_shape).astype(np.float32)
            normalized_spectrograms.append(spec)

        # --- Save metadata ---
        for fname, label_id, spec in zip(filenames, labels, normalized_spectrograms):
            label_name = self.reverse_label_map[label_id]
            self.metadata.append({
                "split": self.split_name,
                "filename": fname,
                "label": label_name,
                "label_id": label_id,
                "spec_type": self.spectrogram_type,
                "shape": spec.shape,
            })

        # --- Save NPZ ---
        output_file = split_output_dir / f"{self.split_name}_spectrograms.npz"
        np.savez_compressed(
            output_file,
            spectrograms=np.array(normalized_spectrograms, dtype=object),
            labels=np.array(labels, dtype=np.int64),
            filenames=np.array(filenames),
        )
        print(f"Saved {len(normalized_spectrograms)} spectrograms to {output_file}")
        return len(normalized_spectrograms)

    def prepare(self):
        """Main preparation pipeline for a single split."""
        print(f"Split: {self.split_name}")
        print(f"Split path: {self.split_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Spectrogram type: {self.spectrogram_type}\n")

        files_by_label = self.find_wav_files()
        if not files_by_label:
            print("No data found! Check your split path.")
            return

        total_processed = self.process_split(files_by_label)

        # Save metadata CSV
        if self.metadata:
            metadata_file = self.output_dir / f"metadata_{self.split_name}.csv"
            pd.DataFrame(self.metadata).to_csv(metadata_file, index=False)
            print(f"Saved metadata to {metadata_file}")

        # Save config (only for training split)
        if self.split_name == "training":
            config = {
                "spectrogram_type": self.spectrogram_type,
                "sample_rate": self.sr,
                "n_fft": self.spec_gen.n_fft,
                "hop_length": self.spec_gen.hop_length,
                "n_mels": self.spec_gen.n_mels,
                "n_mfcc": self.spec_gen.n_mfcc,
                "total_samples": total_processed,
                "label_map": self.label_map,
            }
            config_file = self.output_dir / "config.json"
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Saved config to {config_file}")

        print(f"\n✓ Preparation complete! Processed {total_processed} files for {self.split_name}.")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare WAV files for CNN training (single split)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_data_for_cnn.py --split_path ./training  --split_name training  --spectrogram_type mel --output_dir ./prepared_data
  python prepare_data_for_cnn.py --split_path ./validation --split_name validation --spectrogram_type mel --output_dir ./prepared_data
  python prepare_data_for_cnn.py --split_path ./testing   --split_name testing   --spectrogram_type mel --output_dir ./prepared_data
        """
    )
    parser.add_argument("--split_path", type=str, required=True,
                        help="Path to split folder (should contain real/ and fake/ subfolders)")
    parser.add_argument("--split_name", type=str, required=True,
                        choices=["training", "validation", "testing"],
                        help="Name of the split")
    parser.add_argument("--output_dir", type=str, default="./prepared_data_cnn",
                        help="Output directory for prepared data (default: ./prepared_data_cnn)")
    parser.add_argument("--spectrogram_type", type=str,
                        choices=["mel", "mfcc", "spectrogram"], default="mel",
                        help="Type of spectrogram to generate (default: mel)")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Sample rate for audio loading (default: 16000)")

    args = parser.parse_args()

    preparer = DataPreparer(
        split_path=args.split_path,
        split_name=args.split_name,
        output_dir=args.output_dir,
        spectrogram_type=args.spectrogram_type,
        sr=args.sr,
    )
    preparer.prepare()


if __name__ == "__main__":
    main()
