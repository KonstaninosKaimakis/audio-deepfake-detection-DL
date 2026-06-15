import torch
import librosa
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class WavLMDataset(Dataset):
    def __init__(
        self,
        split_directory, 
        labels = {"real": 0, "fake": 1},
        extractor = None,
        max_duration_sec = None,
        padding = False,
        return_attention_mask = False,
        sample_rate = 16000,
        augment = None,
        ):
        self.extractor = extractor
        self.labels = labels
        self.max_duration_sec = max_duration_sec
        self.padding = padding
        self.return_attention_mask = return_attention_mask
        self.sample_rate = sample_rate
        self.augment = augment   # callable(waveform)->waveform, TRAIN split only; None = off
        split_dir = Path(split_directory)
        if max_duration_sec is not None:
            self.max_samples = int(max_duration_sec * sample_rate)
        else:
            self.max_samples = None
        self.samples: list[tuple[Path, int]] = []

        for class_name, label in self.labels.items():
            class_directory = split_dir / class_name
            if not class_directory.exists():
                raise FileNotFoundError(f'Expected class directory {class_directory} not found!')
            wavs = sorted(class_directory.glob("*.wav"))
            if not wavs:
                raise FileNotFoundError(f'No wav files found in {class_directory}')
            for wav_path in wavs:
                self.samples.append((wav_path, label))
        real_count = sum(1 for _, l in self.samples if l == 0)
        fake_count = sum(1 for _, l in self.samples if l == 1)
        print(f"\n  [ {split_directory} ]  {len(self.samples)} files  (real: {real_count}, fake: {fake_count})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        if self.augment is not None:
            wav = self.augment(wav)          # numpy in -> numpy out (noise / RIR / codec)
        wav = torch.from_numpy(wav)
 
        if wav.shape[0] >= self.max_samples:
            start = torch.randint(0, wav.shape[0] - self.max_samples + 1, (1,)).item()
            wav = wav[start : start + self.max_samples]
            mask = torch.ones(self.max_samples, dtype=torch.long)
        else:
            real_len = wav.shape[0]
            wav = torch.nn.functional.pad(wav, (0, self.max_samples - real_len))
            mask = torch.zeros(self.max_samples, dtype=torch.long)
            mask[:real_len] = 1
 
        return wav, mask, torch.tensor(label, dtype=torch.long)