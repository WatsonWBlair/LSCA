# encoding/pipelines/prosody.py
# Prosody pipeline: 22-dim librosa feature extraction + z-score normalization.
# Features: pitch (2), speaking rate (1), RMS (1), spectral (2), MFCCs (12),
#           jitter (1), shimmer (1), HNR (1), zero-crossing rate (1). Total = 22.
# Ref: McFee et al., librosa 2015

from __future__ import annotations

import json
import logging

import numpy as np

from encoding.config import CAMELSConfig

logger = logging.getLogger(__name__)


def extract_prosody_raw(
    chunk_waveform: np.ndarray,
    sr: int,
    d_prosody: int,
) -> np.ndarray:
    """
    Extract 22-dimensional prosody feature vector from one chunk.

    Feature layout:
       0: pitch mean (f0)          10: MFCC 5
       1: pitch std (f0)           11: MFCC 6
       2: speaking rate            12: MFCC 7
       3: RMS energy mean          13: MFCC 8
       4: spectral centroid        14: MFCC 9
       5: spectral rolloff         15: MFCC 10
       6: MFCC 1                   16: MFCC 11
       7: MFCC 2                   17: MFCC 12
       8: MFCC 3                   18: jitter
       9: MFCC 4                   19: shimmer
                                   20: HNR
                                   21: zero-crossing rate
    """
    import librosa

    if len(chunk_waveform) == 0:
        return np.zeros(d_prosody, dtype=np.float32)

    wav = chunk_waveform.astype(np.float64)

    # Pitch (yin — ~10-15x faster than pyin; no voiced_flag output)
    f0 = librosa.yin(wav, fmin=50, fmax=400, sr=sr)
    voiced = (f0 > 50) & (f0 < 400) & np.isfinite(f0)
    f0 = f0.astype(float)
    f0[~voiced] = np.nan
    pitch_mean = float(np.nanmean(f0)) if f0 is not None and len(f0) > 0 else 0.0
    pitch_std = float(np.nanstd(f0)) if f0 is not None and len(f0) > 0 else 0.0
    speak_rate = float(np.sum(~np.isnan(f0)) / len(f0)) if f0 is not None and len(f0) > 0 else 0.0

    # Energy
    rms = float(librosa.feature.rms(y=chunk_waveform).mean())

    # Spectral shape
    centroid = float(librosa.feature.spectral_centroid(y=chunk_waveform, sr=sr).mean())
    rolloff = float(librosa.feature.spectral_rolloff(y=chunk_waveform, sr=sr).mean())

    # MFCCs (12 coefficients, mean across time)
    mfccs = librosa.feature.mfcc(y=chunk_waveform, sr=sr, n_mfcc=12).mean(axis=1)  # (12,)

    # Jitter (pitch perturbation quotient)
    jitter = 0.0
    if f0 is not None:
        f0_voiced = f0[~np.isnan(f0)]
        if len(f0_voiced) > 1:
            periods = 1.0 / (f0_voiced + 1e-8)
            jitter = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-8))

    # Shimmer (amplitude perturbation)
    shimmer = 0.0
    rms_frames = librosa.feature.rms(y=chunk_waveform).squeeze()
    if len(rms_frames) > 1:
        shimmer = float(np.mean(np.abs(np.diff(rms_frames))) / (np.mean(rms_frames) + 1e-8))

    # Harmonics-to-noise ratio (approximated via autocorrelation)
    hnr = 0.0
    if len(chunk_waveform) > 0:
        autocorr = np.correlate(chunk_waveform, chunk_waveform, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        if autocorr[0] > 0:
            # Find first peak after lag 0 (fundamental period)
            min_lag = max(1, int(sr / 400))  # 400 Hz max
            max_lag = min(len(autocorr), int(sr / 50))  # 50 Hz min
            if max_lag > min_lag:
                segment = autocorr[min_lag:max_lag]
                if len(segment) > 0:
                    peak_val = float(np.max(segment))
                    noise_power = max(autocorr[0] - peak_val, 1e-10)
                    hnr = 10.0 * np.log10(peak_val / noise_power + 1e-10)

    # Zero-crossing rate
    zcr = float(librosa.feature.zero_crossing_rate(y=chunk_waveform).mean())

    raw = np.array(
        [pitch_mean, pitch_std, speak_rate, rms, centroid, rolloff,
         *mfccs, jitter, shimmer, hnr, zcr],
        dtype=np.float32,
    )
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw  # (22,)


def z_score(raw: np.ndarray, stats: dict | None) -> np.ndarray:
    """Apply z-score normalization: (raw - mean) / (std + eps)."""
    if stats is None:
        return raw
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    return (raw - mean) / (std + 1e-8)


def fit_prosody_stats(
    audio_chunks: list[np.ndarray],
    sr: int,
    d_prosody: int,
) -> dict:
    """
    Fit z-score statistics on training split audio chunks.
    MUST be called on training split only.
    """
    all_raw = [extract_prosody_raw(wf, sr=sr, d_prosody=d_prosody) for wf in audio_chunks]
    arr = np.stack(all_raw)
    mean = arr.mean(axis=0).tolist()
    std = arr.std(axis=0).tolist()
    std = [s if s > 1e-8 else 1.0 for s in std]
    return {"mean": mean, "std": std}


def save_prosody_stats(stats: dict, path: str):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Prosody stats saved to %s", path)


def load_prosody_stats(path: str) -> dict:
    with open(path) as f:
        stats = json.load(f)
    logger.info("Prosody stats loaded from %s", path)
    return stats


def prosody_pipeline(
    chunk_waveform: np.ndarray,
    cfg: CAMELSConfig,
    stats: dict | None = None,
) -> np.ndarray:
    """
    Full prosody pipeline for one chunk.
    Returns: (d_prosody,) z-scored float32 numpy array.
    """
    raw = extract_prosody_raw(
        chunk_waveform,
        sr=cfg.streaming.sample_rate,
        d_prosody=cfg.latent.d_prosody,
    )
    return z_score(raw, stats)
