# pipeline/prosody_pipeline.py
# Prosody pipeline: librosa 18-dim feature extraction + z-score normalization
# Features: pitch mean/std (2), speaking rate (1), RMS energy (1),
#           spectral centroid (1), spectral rolloff (1), MFCCs (12). Total = 18.
# Note: plan doc said 22 but the actual feature count is 18.
# Ref: McFee et al., librosa 2015

import json
import logging
import numpy as np

from pipeline.config import SAMPLE_RATE, D_PROSODY, PROSODY_STATS

logger = logging.getLogger(__name__)


# ── Raw feature extraction ────────────────────────────────────────────────────

def extract_prosody_raw(chunk_waveform: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract 22-dimensional prosody feature vector from one chunk.

    Feature layout (indices):
      0  : pitch mean (f0)
      1  : pitch std  (f0)
      2  : speaking rate (fraction of voiced frames)
      3  : RMS energy mean
      4  : spectral centroid mean
      5  : spectral rolloff mean
      6-17: MFCCs 1–12 (mean across time)

    NaN values (e.g., unvoiced frames in pyin) are replaced by 0 before z-scoring.
    Returns raw (un-normalized) (22,) float32 array.
    """
    import librosa

    if len(chunk_waveform) == 0:
        return np.zeros(D_PROSODY, dtype=np.float32)

    # Pitch (pyin) — fmin/fmax chosen for human speech range
    f0, voiced_flag, _ = librosa.pyin(
        chunk_waveform.astype(np.float64),
        fmin=50,
        fmax=400,
        sr=sr,
    )
    pitch_mean  = float(np.nanmean(f0))  if f0 is not None and len(f0) > 0 else 0.0
    pitch_std   = float(np.nanstd(f0))   if f0 is not None and len(f0) > 0 else 0.0
    speak_rate  = float(np.sum(~np.isnan(f0)) / len(f0)) if f0 is not None and len(f0) > 0 else 0.0

    # Energy
    rms = float(librosa.feature.rms(y=chunk_waveform).mean())

    # Spectral shape
    centroid = float(librosa.feature.spectral_centroid(y=chunk_waveform, sr=sr).mean())
    rolloff  = float(librosa.feature.spectral_rolloff(y=chunk_waveform, sr=sr).mean())

    # MFCCs (12 coefficients, mean across time)
    mfccs = librosa.feature.mfcc(y=chunk_waveform, sr=sr, n_mfcc=12).mean(axis=1)  # (12,)

    raw = np.array(
        [pitch_mean, pitch_std, speak_rate, rms, centroid, rolloff, *mfccs],
        dtype=np.float32,
    )
    # Replace NaN/Inf from pyin edge cases
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw  # (22,)


# ── Z-score normalization ─────────────────────────────────────────────────────

def z_score(raw: np.ndarray, stats: dict) -> np.ndarray:
    """
    Apply z-score normalization: (raw - mean) / (std + eps).
    stats must have keys 'mean' and 'std' (lists of length 22).
    If stats is None, returns raw unchanged (for debugging only).
    """
    if stats is None:
        return raw
    mean = np.array(stats["mean"], dtype=np.float32)
    std  = np.array(stats["std"],  dtype=np.float32)
    return (raw - mean) / (std + 1e-8)


# ── Fit stats (training split only) ─────────────────────────────────────────

def fit_prosody_stats(audio_chunks: list, sr: int = SAMPLE_RATE) -> dict:
    """
    Fit z-score statistics on a collection of audio waveform chunks.
    MUST be called on training split only — never on val/test.

    Args:
        audio_chunks : list of (T,) float32 numpy arrays

    Returns:
        stats dict with keys 'mean' and 'std' (lists of length 22).
    """
    all_raw = []
    for waveform in audio_chunks:
        raw = extract_prosody_raw(waveform, sr=sr)
        all_raw.append(raw)

    arr  = np.stack(all_raw)          # (N, 22)
    mean = arr.mean(axis=0).tolist()
    std  = arr.std(axis=0).tolist()

    # Replace zero std with 1 to avoid division by zero
    std = [s if s > 1e-8 else 1.0 for s in std]
    return {"mean": mean, "std": std}


def save_prosody_stats(stats: dict, path: str = PROSODY_STATS):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Prosody stats saved to %s", path)


def load_prosody_stats(path: str = PROSODY_STATS) -> dict:
    with open(path) as f:
        stats = json.load(f)
    logger.info("Prosody stats loaded from %s", path)
    return stats


# ── Full prosody pipeline ─────────────────────────────────────────────────────

def prosody_pipeline(
    chunk_waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    stats: dict = None,
) -> np.ndarray:
    """
    Full prosody pipeline for one chunk.

    Args:
        chunk_waveform : (T,) float32 numpy array at 16 kHz
        sr             : sample rate
        stats          : prosody_stats dict (from load_prosody_stats).
                         Pass None only for debugging — always use stats at inference.

    Returns: (22,) z-scored float32 numpy array
    """
    raw = extract_prosody_raw(chunk_waveform, sr=sr)
    return z_score(raw, stats)
