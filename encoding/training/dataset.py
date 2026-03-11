# encoding/training/dataset.py
# MultimodalDataset: loads pre-extracted raw features for 3 modalities.
# v8.1: video (768), phoneme (MAX_PHONES, 768) + labels + mask, prosody (22).

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from encoding.config import CAMELSConfig

logger = logging.getLogger(__name__)

VRAW_FILE = "v_raw.npy"
PH_RAW_FILE = "ph_raw.npy"
PH_LABELS_FILE = "ph_labels.npy"
PH_MASK_FILE = "ph_mask.npy"
PRAW_FILE = "p_raw.npy"


class MultimodalDataset(Dataset):
    """
    Loads pre-extracted raw features for 3 modalities.

    Expected files in feature_dir:
      v_raw.npy      — (N, d_video)
      ph_raw.npy     — (N, MAX_PHONES, d_phoneme)
      ph_labels.npy  — (N, MAX_PHONES)
      ph_mask.npy    — (N, MAX_PHONES)
      p_raw.npy      — (N, d_prosody)

    Row N in all files corresponds to the same chunk.
    """

    def __init__(self, feature_dir: str, cfg: CAMELSConfig):
        self.feature_dir = feature_dir
        self.cfg = cfg
        self._load()

    def _load(self):
        paths = {
            "v_raw": os.path.join(self.feature_dir, VRAW_FILE),
            "ph_raw": os.path.join(self.feature_dir, PH_RAW_FILE),
            "ph_labels": os.path.join(self.feature_dir, PH_LABELS_FILE),
            "ph_mask": os.path.join(self.feature_dir, PH_MASK_FILE),
            "p_raw": os.path.join(self.feature_dir, PRAW_FILE),
        }
        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Feature file not found: {path}\n"
                    "Run scripts/preprocess_data.py first."
                )

        self.v_raw = torch.from_numpy(np.load(paths["v_raw"]).astype(np.float32))
        self.ph_raw = torch.from_numpy(np.load(paths["ph_raw"]).astype(np.float32))
        self.ph_labels = torch.from_numpy(np.load(paths["ph_labels"]).astype(np.int64))
        self.ph_mask = torch.from_numpy(np.load(paths["ph_mask"]).astype(np.float32)).bool()
        self.p_raw = torch.from_numpy(np.load(paths["p_raw"]).astype(np.float32))

        n = self.v_raw.shape[0]
        assert self.ph_raw.shape[0] == n, f"Row count mismatch: ph_raw ({self.ph_raw.shape[0]} vs {n})"
        assert self.ph_labels.shape[0] == n, f"Row count mismatch: ph_labels"
        assert self.ph_mask.shape[0] == n, f"Row count mismatch: ph_mask"
        assert self.p_raw.shape[0] == n, f"Row count mismatch: p_raw ({self.p_raw.shape[0]} vs {n})"

        # Validate dimensions against config
        assert self.v_raw.shape[1] == self.cfg.latent.d_video, (
            f"v_raw dim {self.v_raw.shape[1]} != d_video {self.cfg.latent.d_video}"
        )
        assert self.ph_raw.shape[2] == self.cfg.latent.d_phoneme, (
            f"ph_raw dim {self.ph_raw.shape[2]} != d_phoneme {self.cfg.latent.d_phoneme}"
        )
        assert self.p_raw.shape[1] == self.cfg.latent.d_prosody, (
            f"p_raw dim {self.p_raw.shape[1]} != d_prosody {self.cfg.latent.d_prosody}"
        )

        logger.info("MultimodalDataset: %d chunks from %s", n, self.feature_dir)

    def __len__(self) -> int:
        return self.v_raw.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.v_raw[idx],       # (d_video,)
            self.ph_raw[idx],      # (MAX_PHONES, d_phoneme)
            self.ph_labels[idx],   # (MAX_PHONES,)
            self.ph_mask[idx],     # (MAX_PHONES,)
            self.p_raw[idx],       # (d_prosody,)
        )


class PregenDataset(Dataset):
    """
    Loads pregenerated backbone tokens from datasets/pregenerated/{backbone_tag}/.
    Aggregates all session/stem subdirectories into a single flat index.
    p_raw is raw (not z-scored) — AVAEAdapter.input_norm handles normalization.
    """

    def __init__(self, pregen_root: str, cfg: CAMELSConfig):
        self.pregen_root = pregen_root
        self.cfg = cfg
        self._index: list[tuple[str, int]] = []  # (stem_dir, local_row_idx)
        self._arrays: dict[str, dict[str, np.ndarray]] = {}  # stem_dir -> {name: mmap}
        self._build_index()

    def _build_index(self):
        root = Path(self.pregen_root)
        stem_dirs = sorted(p for p in root.rglob("v_raw.npy") if p.is_file())
        for npy_path in stem_dirs:
            stem_dir = str(npy_path.parent)
            v = np.load(str(npy_path), mmap_mode="r")
            n = v.shape[0]
            if n == 0:
                continue
            self._arrays[stem_dir] = {
                "v_raw": v,
                "ph_raw": np.load(os.path.join(stem_dir, "ph_raw.npy"), mmap_mode="r"),
                "ph_labels": np.load(os.path.join(stem_dir, "ph_labels.npy"), mmap_mode="r"),
                "ph_mask": np.load(os.path.join(stem_dir, "ph_mask.npy"), mmap_mode="r"),
                "p_raw": np.load(os.path.join(stem_dir, "p_raw.npy"), mmap_mode="r"),
            }
            for i in range(n):
                self._index.append((stem_dir, i))
        logger.info("PregenDataset: %d chunks from %s", len(self._index), self.pregen_root)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        stem_dir, i = self._index[idx]
        arrs = self._arrays[stem_dir]
        return (
            torch.from_numpy(arrs["v_raw"][i].astype(np.float32)),
            torch.from_numpy(arrs["ph_raw"][i].astype(np.float32)),
            torch.from_numpy(arrs["ph_labels"][i].astype(np.int64)),
            torch.from_numpy(arrs["ph_mask"][i].astype(np.float32)).bool(),
            torch.from_numpy(arrs["p_raw"][i].astype(np.float32)),
        )


def make_dataloaders(
    feature_dir: str,
    cfg: CAMELSConfig,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split into train/val/test DataLoaders."""
    dataset = MultimodalDataset(feature_dir, cfg)
    n = len(dataset)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    bs = cfg.training.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers)

    logger.info("DataLoaders: train=%d, val=%d, test=%d (batch=%d)", len(train_ds), len(val_ds), len(test_ds), bs)
    return train_loader, val_loader, test_loader
