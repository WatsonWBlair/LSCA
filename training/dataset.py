# training/dataset.py
# MultimodalDataset: loads pre-extracted raw features from .npy files.
# Preprocessing script: extract_raw_features() processes all .mp4/.wav pairs
# in the data directory and saves v_raw.npy, a_raw.npy, p_raw.npy, t_raw.npy.

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.encoding.utils.config import D_VIDEO, D_AUDIO, D_PROSODY, D_LATENT

logger = logging.getLogger(__name__)

# Raw feature file names (saved once by preprocess_data.py)
VRAW_FILE  = "v_raw.npy"
ARAW_FILE  = "a_raw.npy"
PRAW_FILE  = "p_raw.npy"
TRAW_FILE  = "t_raw.npy"


# ── MultimodalDataset ─────────────────────────────────────────────────────────

class MultimodalDataset(Dataset):
    """
    Loads pre-extracted raw features for all 4 modalities.

    Expected files in feature_dir:
      v_raw.npy  — (N, 768)    MARLIN + TemporalPool output
      a_raw.npy  — (N, d_audio) wav2vec2 mean pool output
      p_raw.npy  — (N, 18)     z-scored prosody features
      t_raw.npy  — (N, 1024)   SONAR text encoding

    Row N in all files corresponds to the same chunk.

    Usage:
        ds = MultimodalDataset("outputs/features")
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        for v_raw, a_raw, p_raw, t_raw in loader:
            ...
    """

    def __init__(self, feature_dir: str):
        self.feature_dir = feature_dir
        self._load()

    def _load(self):
        v_path = os.path.join(self.feature_dir, VRAW_FILE)
        a_path = os.path.join(self.feature_dir, ARAW_FILE)
        p_path = os.path.join(self.feature_dir, PRAW_FILE)
        t_path = os.path.join(self.feature_dir, TRAW_FILE)

        for path in [v_path, a_path, p_path, t_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Feature file not found: {path}\n"
                    "Run scripts/preprocess_data.py first."
                )

        self.v_raw = torch.from_numpy(np.load(v_path).astype(np.float32))
        self.a_raw = torch.from_numpy(np.load(a_path).astype(np.float32))
        self.p_raw = torch.from_numpy(np.load(p_path).astype(np.float32))
        self.t_raw = torch.from_numpy(np.load(t_path).astype(np.float32))

        n = self.v_raw.shape[0]
        assert self.a_raw.shape[0] == n, "Row count mismatch: a_raw"
        assert self.p_raw.shape[0] == n, "Row count mismatch: p_raw"
        assert self.t_raw.shape[0] == n, "Row count mismatch: t_raw"
        logger.info("MultimodalDataset: %d chunks loaded from %s", n, self.feature_dir)

    def __len__(self) -> int:
        return self.v_raw.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.v_raw[idx], self.a_raw[idx], self.p_raw[idx], self.t_raw[idx]


def make_dataloaders(
    feature_dir:  str,
    batch_size:   int = 64,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    num_workers:  int = 0,
    seed:         int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split MultimodalDataset into train/val/test DataLoaders.
    Returns: (train_loader, val_loader, test_loader)
    """
    dataset = MultimodalDataset(feature_dir)
    n       = len(dataset)
    n_test  = max(1, int(n * test_fraction))
    n_val   = max(1, int(n * val_fraction))
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info(
        "DataLoaders: train=%d, val=%d, test=%d (batch=%d)",
        len(train_ds), len(val_ds), len(test_ds), batch_size,
    )
    return train_loader, val_loader, test_loader


# ── Offline feature extraction (run once before training) ────────────────────

def extract_raw_features(
    data_root:    str,
    output_dir:   str,
    models:       dict,
    adapters:     dict,
    prosody_stats: Optional[dict] = None,
    max_videos:   Optional[int]   = None,
):
    """
    Process all .mp4 files in data_root (recursively), extract raw features
    from all 4 frozen encoders, and save to output_dir as .npy files.

    This must be run ONCE before training. The resulting files are the
    training inputs for the AVAE adapters.

    Feature extraction per video:
      v_raw: MARLIN extract_video(crop_face=True) + TemporalAttentionPool
      a_raw: wav2vec2 mean pool of the paired .wav file
      p_raw: librosa 22-dim + z-score (requires prosody_stats)
      t_raw: SONAR encoder of the transcript from the paired .json file
             (uses pre-transcribed words from the dataset's metadata:transcript)
    """
    import librosa
    from src.encoding.pipelines.video_pipeline   import extract_video_file
    from src.encoding.pipelines.audio_pipeline   import audio_pipeline
    from src.encoding.pipelines.prosody_pipeline import prosody_pipeline
    from src.encoding.pipelines.text_pipeline    import encode_text_sonar
    from src.encoding.utils.config           import SAMPLE_RATE, DEFAULT_LANG, WINDOW_SEC, STRIDE_SEC

    os.makedirs(output_dir, exist_ok=True)

    # Find all .mp4 files
    mp4_files = sorted(Path(data_root).rglob("*.mp4"))
    if max_videos is not None:
        mp4_files = mp4_files[:max_videos]
    logger.info("Found %d .mp4 files in %s", len(mp4_files), data_root)

    all_v, all_a, all_p, all_t = [], [], [], []

    for video_path in mp4_files:
        stem = video_path.stem
        wav_path  = video_path.with_suffix(".wav")
        json_path = video_path.with_suffix(".json")

        # ── Video raw feature ──────────────────────────────────────────────
        v_raw = extract_video_file(
            str(video_path),
            models["marlin"],
            adapters["temporal_pool"],
        )  # (768,)

        # ── Audio raw feature ──────────────────────────────────────────────
        if wav_path.exists():
            waveform, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
            waveform     = waveform.astype(np.float32)
            a_raw        = audio_pipeline(waveform, models["wav2vec2"])   # (d_audio,)
            p_raw        = prosody_pipeline(waveform, stats=prosody_stats) # (18,)
        else:
            logger.warning("No .wav for %s — using zeros for audio/prosody", stem)
            a_raw = torch.zeros(D_AUDIO)
            p_raw = np.zeros(D_PROSODY, dtype=np.float32)

        # ── Text raw feature ───────────────────────────────────────────────
        transcript = ""
        if json_path.exists():
            try:
                with open(json_path) as f:
                    meta = json.load(f)
                # Use pre-transcribed text from dataset metadata
                # Each entry is a segment dict with a "transcript" field and
                # a "words" list; concatenate all segments for the full text.
                words_data = meta.get("metadata:transcript", [])
                if isinstance(words_data, list) and len(words_data) > 0:
                    parts = []
                    for entry in words_data:
                        if isinstance(entry, dict):
                            # Prefer the per-segment "transcript" string if present
                            if "transcript" in entry and isinstance(entry["transcript"], str):
                                parts.append(entry["transcript"].strip())
                            elif "words" in entry:
                                parts.append(
                                    " ".join(w["word"] for w in entry["words"] if "word" in w)
                                )
                    transcript = " ".join(p for p in parts if p)
            except Exception as e:
                logger.warning("Failed to read transcript from %s: %s", json_path, e)

        t_raw = encode_text_sonar(transcript, models["sonar"], language=DEFAULT_LANG)  # (1024,)

        all_v.append(v_raw.detach().cpu().numpy())
        all_a.append(a_raw.detach().cpu().numpy())
        all_p.append(p_raw if isinstance(p_raw, np.ndarray) else p_raw.detach().cpu().numpy())
        all_t.append(t_raw.detach().cpu().numpy())

        logger.debug("Processed %s", stem)

    if not all_v:
        logger.error("No features extracted — check data_root path: %s", data_root)
        return

    np.save(os.path.join(output_dir, VRAW_FILE), np.stack(all_v).astype(np.float32))
    np.save(os.path.join(output_dir, ARAW_FILE), np.stack(all_a).astype(np.float32))
    np.save(os.path.join(output_dir, PRAW_FILE), np.stack(all_p).astype(np.float32))
    np.save(os.path.join(output_dir, TRAW_FILE), np.stack(all_t).astype(np.float32))

    logger.info(
        "Saved %d chunks → %s (v_raw, a_raw, p_raw, t_raw)",
        len(all_v), output_dir,
    )
