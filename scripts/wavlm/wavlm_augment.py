"""Online waveform augmentation for WavLM anti-spoofing.

Replicates the augmentation strategy from "Exploring WavLM Back-ends for Speech
Spoofing and Deepfake Detection" (ASVspoof 5): per-sample, online application of
MUSAN additive noise, simulated-RIR reverberation, and codec compression.

Usage (wire into your existing Dataset, TRAIN split only):

    aug = WavLMAugmentationPipeline(
        sample_rate=16000,
        musan_dir="data/musan",
        rir_dir="data/rirs_noises/simulated_rirs",
        p_noise=0.5, p_rir=0.5, p_codec=0.5,
    )
    # in __getitem__ (train only):
    wav = aug(wav)          # numpy[float] or torch.Tensor in -> same type out

    # before the training loop, for experiment tracking:
    mlflow.log_params(aug.as_dict())
    wandb.config.update(aug.as_dict())

Randomization, per __call__:
  1. acoustic effects - two independent Bernoulli draws (p_noise, p_rir) which
     jointly realize {neither, noise, RIR, both};
  2. codec - with prob p_codec, one codec chain is drawn uniformly, else none.

Notes:
  - Codecs use PyAV (`av`), whose wheel bundles the ffmpeg libraries, so no
    system ffmpeg install is needed. If `av` is missing, codec augmentation is
    skipped (warned once) so training is never interrupted.
  - This runs on CPU inside DataLoader workers; it holds no learnable state.
"""

import io
import math
import random
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio.functional as F

try:  # PyAV bundles the ffmpeg libs in its wheel (in-memory mp3/ogg transcode)
    import av
    _AV_IMPORTABLE = True
except Exception:  # pragma: no cover - depends on the install
    _AV_IMPORTABLE = False


# Codec chains: each step is (container_format, encoder, bit_rate_bps). Lower bit_rate
# = lower quality / more compression artifacts. A chain with >1 step is trans-codec
# (encode, then re-encode in another format).
# NB: the bundled ffmpeg has no libvorbis, so the OGG branch uses libopus (Opus),
# the well-supported OGG-family lossy codec - same purpose (codec artifacts).
DEFAULT_CODEC_CONFIGS = [
    {"name": "mp3_high",  "chain": [("mp3", "libmp3lame", 256000)]},
    {"name": "mp3_low",   "chain": [("mp3", "libmp3lame", 64000)]},
    {"name": "opus_high", "chain": [("ogg", "libopus", 128000)]},
    {"name": "opus_low",  "chain": [("ogg", "libopus", 32000)]},
    {"name": "mp3high_to_opushigh",
     "chain": [("mp3", "libmp3lame", 256000), ("ogg", "libopus", 128000)]},
]


class WavLMAugmentationPipeline:
    def __init__(
        self,
        sample_rate=16000,
        musan_dir=None,
        rir_dir=None,
        snr_db_range=(0.0, 15.0),
        p_noise=0.5,
        p_rir=0.5,
        p_codec=0.5,
        musan_subsets=("noise", "music"),
        codec_configs=None,
        verbose=True,
    ):
        self.sample_rate = sample_rate
        self.snr_db_range = tuple(snr_db_range)
        self.p_noise = p_noise
        self.p_rir = p_rir
        self.p_codec = p_codec
        self.musan_dir = Path(musan_dir) if musan_dir is not None else None
        self.rir_dir = Path(rir_dir) if rir_dir is not None else None
        self.musan_subsets = tuple(musan_subsets) if musan_subsets else None
        self.codec_configs = codec_configs if codec_configs is not None else DEFAULT_CODEC_CONFIGS

        self.noise_files = self._index_musan()
        self.rir_files = self._index_rir()
        self._codec_warned = False

        # codec is usable iff PyAV imported (its wheel bundles the ffmpeg libs)
        self.codec_available = _AV_IMPORTABLE

        if verbose:
            self._log_setup()

    # ---- dataset indexing -----------------------------------------------------
    def _index_musan(self):
        if self.musan_dir is None:
            return []
        files = []
        for sub in (self.musan_subsets or ["."]):
            files.extend((self.musan_dir / sub).rglob("*.wav"))
        return sorted(files)

    def _index_rir(self):
        if self.rir_dir is None:
            return []
        return sorted(self.rir_dir.rglob("*.wav"))

    def _log_setup(self):
        print("WavLMAugmentationPipeline")
        print(f"  sample_rate     : {self.sample_rate}")
        print(f"  p_noise/p_rir   : {self.p_noise} / {self.p_rir}  (SNR {self.snr_db_range} dB)")
        print(f"  p_codec         : {self.p_codec}  (available={self.codec_available})")
        print(f"  MUSAN files     : {len(self.noise_files)}  from {self.musan_dir}")
        print(f"  RIR files       : {len(self.rir_files)}  from {self.rir_dir}")
        if self.p_noise > 0 and not self.noise_files:
            warnings.warn("p_noise > 0 but no MUSAN files found; noise will be skipped.")
        if self.p_rir > 0 and not self.rir_files:
            warnings.warn("p_rir > 0 but no RIR files found; reverb will be skipped.")
        if self.p_codec > 0 and not self.codec_available:
            warnings.warn("p_codec > 0 but PyAV (av) is not importable; codecs will be skipped.")

    # ---- io helpers -----------------------------------------------------------
    def _load_random(self, files):
        path = random.choice(files)
        # librosa resamples to target sr and downmixes to mono (avoids torchaudio's
        # TorchCodec backend dependency, which isn't installed)
        wav, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)
        return torch.from_numpy(wav).unsqueeze(0)   # (1, T) float32

    @staticmethod
    def _match_length(x, target_len):
        L = x.shape[-1]
        if L == target_len:
            return x
        if L > target_len:
            start = random.randint(0, L - target_len)
            return x[..., start:start + target_len]
        reps = math.ceil(target_len / L)
        return x.repeat(1, reps)[..., :target_len]

    # ---- individual effects ---------------------------------------------------
    def _add_noise(self, wav):               # wav (1, T)
        noise = self._match_length(self._load_random(self.noise_files), wav.shape[-1])
        snr_db = random.uniform(*self.snr_db_range)
        snr = torch.tensor([snr_db], dtype=wav.dtype)
        return F.add_noise(wav, noise, snr)

    def _add_reverb(self, wav):              # wav (1, T)
        rir = self._load_random(self.rir_files)
        rir = rir / torch.linalg.vector_norm(rir, ord=2)
        reverbed = F.fftconvolve(wav, rir)
        return reverbed[..., :wav.shape[-1]]

    def _encode_once(self, wav, fmt, encoder, bit_rate):
        """Encode a (1, T) float32 waveform with PyAV, decode back to (1, T) float32 mono."""
        samples = wav.squeeze(0).clamp(-1, 1).contiguous().numpy().astype(np.float32)

        # encode to in-memory bytes
        out_buf = io.BytesIO()
        container = av.open(out_buf, mode="w", format=fmt)
        stream = container.add_stream(encoder, rate=self.sample_rate)
        stream.bit_rate = bit_rate
        # pin to one thread: codec contexts default to auto (~n_cpu) threads, and with
        # several DataLoader workers each opening an encoder this exhausts the process
        # thread/memory budget -> ffmpeg raises ENOMEM ([Errno 12]).
        stream.codec_context.thread_count = 1

        in_frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="fltp", layout="mono")
        in_frame.sample_rate = self.sample_rate

        # convert to the encoder's required sample format, then feed fixed-size chunks
        resampler = av.AudioResampler(format=stream.format.name, layout="mono", rate=self.sample_rate)
        fifo = av.AudioFifo()
        for rf in resampler.resample(in_frame):
            fifo.write(rf)

        frame_size = stream.codec_context.frame_size or 1024
        while fifo.samples >= frame_size:
            chunk = fifo.read(frame_size)
            chunk.pts = None
            for pkt in stream.encode(chunk):
                container.mux(pkt)
        if fifo.samples > 0:
            chunk = fifo.read()
            chunk.pts = None
            for pkt in stream.encode(chunk):
                container.mux(pkt)
        for pkt in stream.encode(None):      # flush the encoder
            container.mux(pkt)
        container.close()

        # decode back to float32 mono at the target sample rate
        out_buf.seek(0)
        in_container = av.open(out_buf, mode="r")
        in_container.streams.audio[0].thread_count = 1   # single-thread the decoder too (ENOMEM guard)
        dec_res = av.AudioResampler(format="fltp", layout="mono", rate=self.sample_rate)
        pieces = []
        for frame in in_container.decode(audio=0):
            for rf in dec_res.resample(frame):
                pieces.append(rf.to_ndarray().reshape(-1))
        in_container.close()

        if not pieces:
            return wav
        decoded = np.concatenate(pieces).astype(np.float32)
        return torch.from_numpy(decoded).unsqueeze(0)

    def _apply_codec(self, wav):             # wav (1, T)
        cfg = random.choice(self.codec_configs)
        try:
            x = wav
            for fmt, encoder, bit_rate in cfg["chain"]:
                x = self._encode_once(x, fmt, encoder, bit_rate)
            return self._match_length(x, wav.shape[-1])
        except Exception as e:               # codec/encoder unavailable or input too short
            if not self._codec_warned:
                warnings.warn(f"codec '{cfg['name']}' failed ({e}); skipping codec augmentation.")
                self._codec_warned = True
            return wav

    # ---- callable -------------------------------------------------------------
    def __call__(self, waveform):
        wav, restore = self._to_2d(waveform)

        if self.noise_files and random.random() < self.p_noise:
            wav = self._add_noise(wav)
        if self.rir_files and random.random() < self.p_rir:
            wav = self._add_reverb(wav)
        if self.codec_available and random.random() < self.p_codec:
            wav = self._apply_codec(wav)

        return restore(wav)

    @staticmethod
    def _to_2d(waveform):
        """Normalize input to a (1, T) float32 tensor; return a restorer to the original type/shape."""
        if isinstance(waveform, np.ndarray):
            orig_1d = waveform.ndim == 1
            t = torch.from_numpy(waveform).float()
            t = t.unsqueeze(0) if orig_1d else t
            def restore(x):
                x = x.squeeze(0) if orig_1d else x
                return x.detach().cpu().numpy().astype(np.float32)
            return t, restore

        orig_1d = waveform.dim() == 1
        t = waveform.float()
        t = t.unsqueeze(0) if orig_1d else t
        def restore(x):
            return x.squeeze(0) if orig_1d else x
        return t, restore

    # ---- experiment tracking --------------------------------------------------
    def as_dict(self):
        """Flat, log-friendly view of the augmentation config (mlflow / wandb)."""
        return {
            "aug_sample_rate": self.sample_rate,
            "aug_p_noise": self.p_noise,
            "aug_p_rir": self.p_rir,
            "aug_p_codec": self.p_codec,
            "aug_snr_db_min": self.snr_db_range[0],
            "aug_snr_db_max": self.snr_db_range[1],
            "aug_musan_dir": str(self.musan_dir),
            "aug_rir_dir": str(self.rir_dir),
            "aug_musan_subsets": ",".join(self.musan_subsets) if self.musan_subsets else "all",
            "aug_n_noise_files": len(self.noise_files),
            "aug_n_rir_files": len(self.rir_files),
            "aug_codec_configs": ",".join(c["name"] for c in self.codec_configs),
            "aug_codec_available": self.codec_available,
        }


if __name__ == "__main__":
    # smoke test: 1 s sine through each branch, no datasets required
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--musan_dir", default=None)
    parser.add_argument("--rir_dir", default=None)
    args = parser.parse_args()

    sr = 16000
    t = torch.arange(sr) / sr
    sine = (0.5 * torch.sin(2 * math.pi * 220 * t)).numpy().astype(np.float32)

    aug = WavLMAugmentationPipeline(sample_rate=sr, musan_dir=args.musan_dir, rir_dir=args.rir_dir)
    print("\nconfig:", aug.as_dict())

    out = aug(sine)
    print(f"\nin  shape/dtype: {sine.shape} {sine.dtype}")
    print(f"out shape/dtype: {out.shape} {out.dtype}")
    assert out.shape == sine.shape and out.dtype == sine.dtype
    print("smoke test OK")
