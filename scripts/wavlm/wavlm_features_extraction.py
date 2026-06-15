# AUDIO DEEPFAKE DETECTION - WAVLM FEATURE EXTRACTION (Phase 1a)
# Runs a frozen WavLM over raw audio and saves per-layer, time-pooled
# embeddings to disk as .npy files. One (n_layers, hidden) array per clip.
#
# Why pre-extract? WavLM is frozen here, so its output for a clip never changes.
# We run the expensive transformer ONCE and dump embeddings to disk; the cheap
# probe (training_wavlm_probe.py) then trains in seconds and can be re-tuned
# freely without ever touching WavLM again. Same two-stage pattern as the CNN track.
#
# We save ALL hidden layers (embeddings + every transformer block), mean-pooled
# over time. This lets the probe LEARN which layers matter (a weighted layer-sum,
# the standard SUPERB / anti-spoofing approach) with no re-extraction.

import os
from pathlib import Path
import numpy as np
import torch
import librosa
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMModel

# Configuration
config = {
    # Checkpoint. "microsoft/wavlm-base-plus" (768-d, 13 layers) is the default.
    # Swapping to "microsoft/wavlm-large" (1024-d, 25 layers) is a one-line change.
    "CHECKPOINT": "microsoft/wavlm-base-plus",
    "TRAIN_PATH": os.path.join(os.getcwd(), "data" , "FoR_dataset" , "for-norm" , "for-norm" , "training"),
    "VAL_PATH": os.path.join(os.getcwd(), "data" , "FoR_dataset" , "for-norm" , "for-norm" , "testing"),
    "TEST_PATH": os.path.join(os.getcwd(), "data" , "in-the-wild-audio-deepfake" , "release_in_the_wild"),
    #"TEST_PATH": os.path.join(os.getcwd(), "data", "FoR_dataset", "for-norm", "for-norm", "testing"),
    "OUTPUT_DIR": Path(os.getcwd()) / "data" / "wavlm_base_plus_features",
    "LABEL_MAP": {"real": 0, "fake": 1},
    "SAMPLE_RATE": 16000,
    "MAX_DURATION": 8.0,
    "EXTRACT_BATCH_SIZE": 16,
    "USE_BF16": True,  # bf16 autocast on CUDA (5070 Ti supports it)
    "SKIP_EXISTING": True,  # resume-friendly: skip already-saved .npy files
    "LIMIT": None,  # int -> cap files per split for a smoke test
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if (config["USE_BF16"] and DEVICE.type == "cuda") else torch.float32

def load_audio(path):
    y, _ = librosa.load(path, sr=config["SAMPLE_RATE"], duration=config["MAX_DURATION"])
    return y.astype(np.float32)


@torch.no_grad()
def extract_batch(waves, model, feat_extractor):
    """
    waves: list of 1-D float32 arrays.
    Returns (B, n_layers, hidden) -- every hidden state, masked-mean-pooled over time.
    """
    inputs = feat_extractor(
        waves,
        sampling_rate=config["SAMPLE_RATE"],
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_values   = inputs["input_values"].to(DEVICE)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    with torch.autocast(device_type=DEVICE.type, dtype=DTYPE, enabled=(DTYPE != torch.float32)):
        out = model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # hidden_states: tuple of (B, frames, hidden), len = n_transformer_layers + 1
    hidden = torch.stack(out.hidden_states, dim=1).float()   # (B, L, frames, hidden)
    B, L, T, H = hidden.shape

    # Build a frame-level mask from the sample-level attention_mask so padded
    # frames don't pollute the mean. WavLM downsamples ~320x; the model exposes
    # the exact mapping via this helper.
    if attention_mask is not None:
        frame_mask = model._get_feature_vector_attention_mask(T, attention_mask)  # (B, frames)
        frame_mask = frame_mask.to(hidden.dtype)
        m = frame_mask[:, None, :, None]                      # (B, 1, frames, 1)
        pooled = (hidden * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)  # (B, L, hidden)
    else:
        pooled = hidden.mean(dim=2)

    return pooled.cpu().numpy().astype(np.float32)


# ==============================================================================
# EXTRACTION LOOP
# ==============================================================================
def gather_files(split_path: Path) -> list:
    items = []
    for label_name in config["LABEL_MAP"]:
        label_dir = split_path / label_name
        if not label_dir.is_dir():
            continue
        for wav in sorted(label_dir.glob("*.wav")):
            items.append((wav, label_name))
    return items


def process_split(split_name: str, split_path: Path, model, feat_extractor):
    print("=" * 60)
    print(f"PROCESSING SPLIT: {split_name.upper()}")
    print("=" * 60)

    if not split_path.exists():
        print(f"WARNING: path not found, skipping: {split_path}\n")
        return

    files = gather_files(split_path)
    if config["LIMIT"] is not None:
        files = files[: config["LIMIT"]]
    if not files:
        print(f"WARNING: no .wav files under {split_path}. Skipping.\n")
        return

    print(f"Found {len(files)} files. Extracting WavLM features...")

    batch_waves, batch_targets = [], []
    saved = skipped = failed = 0

    def flush():
        nonlocal saved, failed
        if not batch_waves:
            return
        try:
            feats = extract_batch(batch_waves, model, feat_extractor)
        except Exception as e:
            print(f"  Batch failed ({len(batch_waves)} files): {e}")
            failed += len(batch_waves)
            batch_waves.clear()
            batch_targets.clear()
            return
        for feat, out_path in zip(feats, batch_targets):
            np.save(out_path, feat)
            saved += 1
        batch_waves.clear()
        batch_targets.clear()

    for wav_path, label_name in tqdm(files, desc=f"{split_name.capitalize()}"):
        target_dir = config["OUTPUT_DIR"] / split_name / label_name
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{wav_path.stem}.npy"

        if config["SKIP_EXISTING"] and out_path.exists():
            skipped += 1
            continue

        try:
            batch_waves.append(load_audio(str(wav_path)))
            batch_targets.append(out_path)
        except Exception as e:
            print(f"  Load failed {wav_path.name}: {e}")
            failed += 1
            continue

        if len(batch_waves) >= config["EXTRACT_BATCH_SIZE"]:
            flush()

    flush()
    print(f"-> saved {saved}, skipped {skipped}, failed {failed} | {config['OUTPUT_DIR'] / split_name}\n")


def main():
    print(f"Device     : {DEVICE}  (dtype: {DTYPE})")
    print(f"Checkpoint : {config['CHECKPOINT']}")
    print(f"Output dir : {config['OUTPUT_DIR']}\n")

    print("Loading WavLM (frozen)...")
    feat_extractor = AutoFeatureExtractor.from_pretrained(config["CHECKPOINT"])
    model = WavLMModel.from_pretrained(config["CHECKPOINT"]).to(DEVICE).eval()
    n_layers = model.config.num_hidden_layers + 1
    print(f"  {n_layers} hidden layers, hidden_size={model.config.hidden_size}\n")

    for split_name, split_path in [
        ("training", config["TRAIN_PATH"]),
        ("validation", config["VAL_PATH"]),
        ("testing", config["TEST_PATH"]),
    ]:
        process_split(split_name, Path(split_path), model, feat_extractor)

    print("=" * 60)
    print(f"DONE. Feature arrays saved under: {config['OUTPUT_DIR']}")
    print(f"Each file shape: ({n_layers}, {model.config.hidden_size})")
    print("=" * 60)


if __name__ == "__main__":
    main()
