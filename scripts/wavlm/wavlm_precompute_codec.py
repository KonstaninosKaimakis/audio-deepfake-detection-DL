"""Offline codec pre-augmentation: render codec-degraded copies of the TRAIN split to disk.

Run this once before fine-tuning. WavLMDataset then randomly swaps in these variants (prob
cfg.augment.p_codec)

For each train wav it writes `<codec_cache_dir>/<class>/<stem>__<codec>.wav`
for every codec in DEFAULT_CODEC_CONFIGS (or a `--codecs` subset).

    python -m scripts.wavlm.wavlm_precompute_codec --config training_configs/wavlm_base_plus.yaml
"""
import argparse
import random
from pathlib import Path

import librosa
import torch
import soundfile as sf
from tqdm import tqdm

from scripts.wavlm.utils import load_config
from scripts.wavlm.wavlm_augment import WavLMAugmentationPipeline, DEFAULT_CODEC_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training_configs/wavlm_base_plus.yaml")
    parser.add_argument("--codecs", nargs="*", default=None,
                        help="subset of codec names to render (default: all in DEFAULT_CODEC_CONFIGS)")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="render variants for only this fraction of train files per class (saves disk)")
    parser.add_argument("--seed", type=int, default=42, help="seed for --fraction sampling (reproducible)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sr = cfg.data.sample_rate
    train_dir = Path(cfg.data.train_dir)
    out_dir = Path(cfg.augment.codec_cache_dir)
    labels = {"real": 0, "fake": 1}

    configs = DEFAULT_CODEC_CONFIGS
    if args.codecs:
        wanted = set(args.codecs)
        configs = [c for c in DEFAULT_CODEC_CONFIGS if c["name"] in wanted]
        if not configs:
            raise SystemExit(f"No codec configs match {args.codecs}; "
                             f"available: {[c['name'] for c in DEFAULT_CODEC_CONFIGS]}")

    # pipeline is used only for its codec machinery -> skip MUSAN/RIR indexing
    pipe = WavLMAugmentationPipeline(sample_rate=sr, musan_dir=None, rir_dir=None, verbose=False)
    if not pipe.codec_available:
        raise SystemExit("PyAV (av) is not importable; cannot render codecs.")

    print(f"Rendering {len(configs)} codec variants/file: {[c['name'] for c in configs]}")
    print(f"  src: {train_dir}")
    print(f"  dst: {out_dir}")

    n_done = n_skip = n_fail = 0
    for class_name in labels:
        src_class = train_dir / class_name
        if not src_class.exists():
            print(f"  skip: missing class dir {src_class}")
            continue
        dst_class = out_dir / class_name
        dst_class.mkdir(parents=True, exist_ok=True)

        wavs = sorted(src_class.glob("*.wav"))
        if args.fraction < 1.0:
            total = len(wavs)
            k = max(1, int(total * args.fraction))
            wavs = sorted(random.Random(args.seed).sample(wavs, k))
            print(f"  {class_name}: sampling {k}/{total} files (fraction={args.fraction})")
        for wav_path in tqdm(wavs, desc=class_name, leave=False):
            wav = None  # lazy-load the source only when at least one variant is missing
            for codec_cfg in configs:
                dst = dst_class / f"{wav_path.stem}__{codec_cfg['name']}.wav"
                if dst.exists():
                    n_skip += 1
                    continue
                if wav is None:
                    y, _ = librosa.load(wav_path, sr=sr, mono=True)
                    wav = torch.from_numpy(y).unsqueeze(0)
                if wav.shape[-1] < 2:
                    break  # degenerate clip; nothing to encode
                try:
                    out = pipe.encode_with_codec(wav, codec_cfg, match_length=True)
                    sf.write(str(dst), out.squeeze(0).numpy(), sr)
                    n_done += 1
                except Exception as e:
                    n_fail += 1
                    if n_fail <= 10:
                        print(f"  fail {wav_path.name} [{codec_cfg['name']}]: {e}")

    print(f"\nDone. rendered {n_done}, skipped (existing) {n_skip}, failed {n_fail}")
    print(f"Variants under {out_dir}")


if __name__ == "__main__":
    main()
