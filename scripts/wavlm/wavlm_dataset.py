import random
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
        codec_cache_dir = None,
        p_codec = 0.0,
        ):
        self.extractor = extractor
        self.labels = labels
        self.max_duration_sec = max_duration_sec
        self.padding = padding
        self.return_attention_mask = return_attention_mask
        self.sample_rate = sample_rate
        self.augment = augment   # callable(waveform)->waveform, TRAIN split only; None = off
        self.p_codec = p_codec   # prob of loading a pre-rendered codec variant instead of the original
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

        # offline codec: map (class_name, original_stem) -> [pre-rendered variant paths]
        self.codec_variants = {}
        if codec_cache_dir is not None and p_codec > 0:
            self._index_codec_variants(Path(codec_cache_dir))
            n_var = sum(len(v) for v in self.codec_variants.values())
            if n_var == 0:
                print(f"  WARNING: p_codec={p_codec} but no codec variants found under {codec_cache_dir}; "
                      f"run wavlm_precompute_codec.py first. Codec swap disabled.")
            else:
                print(f"  codec variants: {n_var} files for {len(self.codec_variants)} originals "
                      f"(p_codec={p_codec})")

    def _index_codec_variants(self, codec_cache_dir):
        # variants are named "{original_stem}__{codec}.wav" under codec_cache_dir/{class}/
        for class_name in self.labels:
            cdir = codec_cache_dir / class_name
            if not cdir.exists():
                continue
            for vp in cdir.glob("*__*.wav"):
                orig_stem = vp.stem.rsplit("__", 1)[0]
                self.codec_variants.setdefault((class_name, orig_stem), []).append(vp)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # offline codec: with prob p_codec, load a pre-rendered codec-degraded variant of this file
        # (no ffmpeg in workers). Online noise/RIR (self.augment) is then applied on top.
        if self.codec_variants and random.random() < self.p_codec:
            variants = self.codec_variants.get((path.parent.name, path.stem))
            if variants:
                path = random.choice(variants)

        # enforce sample rate at load for exact dimensional consistency
        wav, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        if self.augment is not None:
            wav = self.augment(wav)          # numpy in -> numpy out (noise / RIR)
        wav = torch.from_numpy(wav)

        # removes per-domain level/gain differences (FoR vs ITW) that widen the domain gap.
        # guard 1-sample/empty clips: var() needs >=2 samples or it returns NaN -> poisons features.
        if wav.numel() > 1:
            wav = (wav - wav.mean()) / torch.sqrt(wav.var() + 1e-7)

        if wav.shape[0] >= self.max_samples:
            # reproducible eval and stable cached features (no random-offset slices)
            start = (wav.shape[0] - self.max_samples) // 2
            wav = wav[start : start + self.max_samples]
            mask = torch.ones(self.max_samples, dtype=torch.long)
        else:
            real_len = wav.shape[0]
            wav = torch.nn.functional.pad(wav, (0, self.max_samples - real_len))
            mask = torch.zeros(self.max_samples, dtype=torch.long)
            mask[:real_len] = 1
 
        return wav, mask, torch.tensor(label, dtype=torch.long)