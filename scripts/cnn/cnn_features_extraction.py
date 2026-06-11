"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AUDIO DEEPFAKE DETECTION — OFFLINE PREPROCESSING SCRIPT              ║
║  Extracts, normalizes, and saves spectrograms to disk as .npy files          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pickle
import random
import warnings
from pathlib import Path
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]

# ==============================================================================
# ▼▼▼  CONFIG — Local repo paths for your own environment  ▼▼▼
# ==============================================================================
CONFIG = {
    "TRAIN_PATH": REPO_ROOT / "data" / "for-norm" / "for-norm" / "training",
    "VAL_PATH":   REPO_ROOT / "data" / "for-norm" / "for-norm" / "testing",
    "TEST_PATH":  REPO_ROOT / "data" / "release_in_the_wild_trimmed_normalized",

    "MAX_DURATION": 8.5,       # Hardcoded from your Step 1 results!
    "SAMPLE_RATE":  16000,
    "N_FFT":        1024,       
    "HOP_LENGTH":   256,
    "N_MELS":       128,        
    "SPEC_TYPE":    "mel",      # "mel" | "mfcc" | "spectrogram"
    "SCALER_FIT_SAMPLES": 10000,
}

# ── Derived globals ───────────────────────────────────────────────────────────
_spec_type = CONFIG["SPEC_TYPE"]
_out_dir   = REPO_ROOT / "data" / f"cnn_{_spec_type}_features"
_scaler_dir = REPO_ROOT / "data" / "scalers"
_scaler_dir.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"real": 0, "fake": 1}

# ==============================================================================
# AUDIO PROCESSING CORE (Identical to your original script)
# ==============================================================================
def load_fixed_audio(path: str, max_duration: float) -> np.ndarray:
    sr         = CONFIG["SAMPLE_RATE"]
    target_len = int(max_duration * sr)
    y, _ = librosa.load(path, sr=sr, duration=max_duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y.astype(np.float32)

def audio_to_spectrogram(y: np.ndarray) -> np.ndarray:
    sr         = CONFIG["SAMPLE_RATE"]
    n_fft      = CONFIG["N_FFT"]
    hop_length = CONFIG["HOP_LENGTH"]

    if _spec_type == "mel":
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=CONFIG["N_MELS"])
        return librosa.power_to_db(S, ref=np.max).astype(np.float32)
    elif _spec_type == "mfcc":
        return librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=CONFIG["N_MFCC"]).astype(np.float32)
    elif _spec_type == "spectrogram":
        S = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
        return librosa.power_to_db(np.abs(S) ** 2, ref=np.max).astype(np.float32)
    else:
        raise ValueError(f"Unknown SPEC_TYPE '{_spec_type}'")

def fit_or_load_scaler(max_duration: float) -> StandardScaler:
    scaler_path = _scaler_dir / f"scaler_{_spec_type}.pkl"
    if scaler_path.exists():
        print(f"-> Found existing scaler. Loading from {scaler_path}\n")
        with open(scaler_path, "rb") as f:
            return pickle.load(f)

    print("=" * 60)
    print("STEP 1 — FITTING SCALER")
    print("=" * 60)

    all_wavs = []
    for label_dir in Path(CONFIG["TRAIN_PATH"]).iterdir():
        if label_dir.is_dir() and label_dir.name.lower() in LABEL_MAP:
            all_wavs.extend(label_dir.glob("*.wav"))

    n_samples = min(CONFIG["SCALER_FIT_SAMPLES"], len(all_wavs))
    sampled   = random.sample(all_wavs, n_samples)
    print(f"Fitting scaler on {n_samples} training files...")

    all_values = []
    for wav in tqdm(sampled, desc="Scaler calculation", leave=False):
        try:
            y    = load_fixed_audio(str(wav), max_duration)
            spec = audio_to_spectrogram(y)
            all_values.append(spec.reshape(-1))
        except Exception as e:
            print(f"Warning — skipping {wav.name}: {e}")

    scaler = StandardScaler().fit(np.concatenate(all_values).reshape(-1, 1))
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"-> Scaler successfully saved to {scaler_path}\n")
    return scaler

# ==============================================================================
# OFFLINE EXTRACTION LOOP
# ==============================================================================
def process_and_save_split(split_name: str, source_root: str, scaler: StandardScaler, max_duration: float):
    print("=" * 60)
    print(f"PROCESSING SPLIT: {split_name.upper()}")
    print("=" * 60)
    
    source_path = Path(source_root)
    all_wavs = list(source_path.rglob("*.wav"))
    
    if not all_wavs:
        print(f"⚠️ Warning: No WAV files found in {source_root}. Skipping split.")
        return

    print(f"Found {len(all_wavs)} files. Extracting and saving features...")
    
    success_count = 0
    for wav_path in tqdm(all_wavs, desc=f"{split_name.capitalize()} Progress"):
        # Identify if real or fake based on folder hierarchy
        label_folder = None
        for parent in wav_path.parents:
            if parent.name.lower() in LABEL_MAP:
                label_folder = parent.name.lower()
                break
        
        if not label_folder:
            continue  # Skip files not contained inside a real/fake directory
            
        # Mirror structure: processed_data / split_name / label_folder
        target_dir = _out_dir / split_name / label_folder
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Output filename: change .wav to .npy
        target_file_path = target_dir / f"{wav_path.stem}.npy"
        
        try:
            # 1. Load & Extract
            y    = load_fixed_audio(str(wav_path), max_duration)
            spec = audio_to_spectrogram(y)

            # 2. Normalize
            shape       = spec.shape
            spec_normed = scaler.transform(spec.reshape(-1, 1)).reshape(shape).astype(np.float32)

            # 3. Save as raw binary array
            np.save(str(target_file_path), spec_normed)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {wav_path.name}: {e}")
            
    print(f"-> Successfully saved {success_count} arrays to {target_dir.parent}\n")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    max_duration = CONFIG["MAX_DURATION"]
    print(f"Target duration set to: {max_duration} seconds.")
    
    # Step 1: Fit/Get Scaler
    scaler = fit_or_load_scaler(max_duration)
    
    # Step 2: Loop through each split and process offline
    splits = [
        ("training",   CONFIG["TRAIN_PATH"]),
        ("validation", CONFIG["VAL_PATH"]),
        ("testing",    CONFIG["TEST_PATH"]),
    ]
    
    for split_name, source_root in splits:
        process_and_save_split(split_name, source_root, scaler, max_duration)
        
    print("=" * 60)
    print(f"🎉 DONE! All normalized .npy matrices saved under: {_out_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()