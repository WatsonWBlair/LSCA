# encoding/training/dataset.py
# MultimodalDataset: loads pre-extracted raw features for 3 modalities.
# v8.1: video (768), phoneme (MAX_PHONES, 768) + labels + mask, prosody (22).

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
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
    Loads pre-extracted raw features for 3 modalities via memory-mapped numpy files.

    Expected files in feature_dir (produced by scripts/consolidate_pregenerated.py):
      v_raw.npy      — (N, d_video)
      ph_raw.npy     — (N, MAX_PHONES, d_phoneme)
      ph_labels.npy  — (N, MAX_PHONES)
      ph_mask.npy    — (N, MAX_PHONES)
      p_raw.npy      — (N, d_prosody)

    Row N in all files corresponds to the same chunk.

    Files are opened with mmap_mode='r' — the OS pages in only the slices
    accessed by each __getitem__ call. RAM at construction is near zero;
    only batch-sized windows are resident during training.
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
                    "Run: invoke consolidate-pregenerated"
                )

        # mmap_mode='r': files stay on disk; OS pages in slices on demand.
        self._v_raw     = np.load(paths["v_raw"],     mmap_mode="r")
        self._ph_raw    = np.load(paths["ph_raw"],    mmap_mode="r")
        self._ph_labels = np.load(paths["ph_labels"], mmap_mode="r")
        self._ph_mask   = np.load(paths["ph_mask"],   mmap_mode="r")
        self._p_raw     = np.load(paths["p_raw"],     mmap_mode="r")

        n = self._v_raw.shape[0]
        assert self._ph_raw.shape[0] == n,    f"Row count mismatch: ph_raw ({self._ph_raw.shape[0]} vs {n})"
        assert self._ph_labels.shape[0] == n, "Row count mismatch: ph_labels"
        assert self._ph_mask.shape[0] == n,   "Row count mismatch: ph_mask"
        assert self._p_raw.shape[0] == n,     f"Row count mismatch: p_raw ({self._p_raw.shape[0]} vs {n})"

        # Validate dimensions against config
        assert self._v_raw.shape[1] == self.cfg.latent.d_video, (
            f"v_raw dim {self._v_raw.shape[1]} != d_video {self.cfg.latent.d_video}"
        )
        assert self._ph_raw.shape[2] == self.cfg.latent.d_phoneme, (
            f"ph_raw dim {self._ph_raw.shape[2]} != d_phoneme {self.cfg.latent.d_phoneme}"
        )
        assert self._p_raw.shape[1] == self.cfg.latent.d_prosody, (
            f"p_raw dim {self._p_raw.shape[1]} != d_prosody {self.cfg.latent.d_prosody}"
        )

        logger.info("MultimodalDataset: %d chunks from %s (mmap)", n, self.feature_dir)

    def __len__(self) -> int:
        return self._v_raw.shape[0]

    def __getitem__(self, idx: int):
        # .copy() is required: mmap slices are views without owned memory.
        # DataLoader multiprocessing workers must own the data they transfer.
        # astype(copy=False) avoids a double copy when dtype already matches.
        return (
            torch.from_numpy(self._v_raw[idx].astype(np.float32, copy=False).copy()),
            torch.from_numpy(self._ph_raw[idx].astype(np.float32, copy=False).copy()),
            torch.from_numpy(self._ph_labels[idx].copy()),
            torch.from_numpy(self._ph_mask[idx].astype(np.float32, copy=False).copy()).bool(),
            torch.from_numpy(self._p_raw[idx].astype(np.float32, copy=False).copy()),
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


class DyadicPairDataset(Dataset):
    """
    Loads matched dyadic pairs from pregenerated tokens.

    Uses `partner_stem` recorded in each stem's chunks.jsonl to pair chunk rows
    from two participants in the same conversation. Only emits pairs where both
    stems and partner_stem exist under the same session directory.

    __getitem__ returns (A_features, B_features) where each is a 5-tuple matching
    the PregenDataset item format: (v_raw, ph_raw, ph_labels, ph_mask, p_raw).

    Stems with no partner_stem or whose partner is missing fall back to no pairs.
    """

    def __init__(self, pregen_root: str, cfg: CAMELSConfig):
        self.pregen_root = pregen_root
        self.cfg = cfg
        self._pairs: list[tuple[str, int, str, int]] = []  # (stem_dir_A, row_A, stem_dir_B, row_B)
        self._arrays: dict[str, dict[str, np.ndarray]] = {}
        self._build_pairs()

    def _load_stem(self, stem_dir: str) -> None:
        if stem_dir in self._arrays:
            return
        self._arrays[stem_dir] = {
            "v_raw":     np.load(os.path.join(stem_dir, "v_raw.npy"),     mmap_mode="r"),
            "ph_raw":    np.load(os.path.join(stem_dir, "ph_raw.npy"),    mmap_mode="r"),
            "ph_labels": np.load(os.path.join(stem_dir, "ph_labels.npy"), mmap_mode="r"),
            "ph_mask":   np.load(os.path.join(stem_dir, "ph_mask.npy"),   mmap_mode="r"),
            "p_raw":     np.load(os.path.join(stem_dir, "p_raw.npy"),     mmap_mode="r"),
        }

    def _build_pairs(self) -> None:
        root = Path(self.pregen_root)
        stem_dirs = sorted(p.parent for p in root.rglob("v_raw.npy") if p.is_file())

        # (session_dir_str, stem_name) -> stem_dir_str
        stem_dir_map: dict[tuple[str, str], str] = {
            (str(sd.parent), sd.name): str(sd) for sd in stem_dirs
        }

        # Group stems by (session_dir, partner_stem) to pair both directions once
        paired: set[tuple[str, str]] = set()

        for stem_dir in stem_dirs:
            jsonl_path = stem_dir / "chunks.jsonl"
            if not jsonl_path.exists():
                continue

            partner_stem: str | None = None
            chunk_count_a = 0
            with open(jsonl_path) as f:
                for line in f:
                    rec = json.loads(line)
                    if partner_stem is None:
                        partner_stem = rec.get("partner_stem")
                    chunk_count_a += 1

            if not partner_stem:
                continue

            session_str = str(stem_dir.parent)
            partner_dir_str = stem_dir_map.get((session_str, partner_stem))
            if not partner_dir_str:
                continue

            pair_key = tuple(sorted([str(stem_dir), partner_dir_str]))
            if pair_key in paired:
                continue
            paired.add(pair_key)

            partner_jsonl = Path(partner_dir_str) / "chunks.jsonl"
            if not partner_jsonl.exists():
                continue
            chunk_count_b = sum(1 for _ in open(partner_jsonl))

            n = min(chunk_count_a, chunk_count_b)
            for i in range(n):
                self._pairs.append((str(stem_dir), i, partner_dir_str, i))

            self._load_stem(str(stem_dir))
            self._load_stem(partner_dir_str)

        logger.info("DyadicPairDataset: %d pairs from %s", len(self._pairs), self.pregen_root)

    def __len__(self) -> int:
        return len(self._pairs)

    def _get_features(self, stem_dir: str, i: int):
        arrs = self._arrays[stem_dir]
        return (
            torch.from_numpy(arrs["v_raw"][i].astype(np.float32)),
            torch.from_numpy(arrs["ph_raw"][i].astype(np.float32)),
            torch.from_numpy(arrs["ph_labels"][i].astype(np.int64)),
            torch.from_numpy(arrs["ph_mask"][i].astype(np.float32)).bool(),
            torch.from_numpy(arrs["p_raw"][i].astype(np.float32)),
        )

    def __getitem__(self, idx: int):
        stem_dir_a, row_a, stem_dir_b, row_b = self._pairs[idx]
        return self._get_features(stem_dir_a, row_a), self._get_features(stem_dir_b, row_b)


def make_dataloaders(
    feature_dir: str,
    cfg: CAMELSConfig,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    batch_size: int | None = None,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split into train/val/test DataLoaders.

    batch_size overrides cfg.training.batch_size when provided.
    pin_memory=True (default) enables async PCIe DMA when combined with
    non_blocking=True device transfers in the training loop.
    persistent_workers=True avoids per-epoch fork overhead when num_workers > 0.
    """
    dataset = MultimodalDataset(feature_dir, cfg)
    n = len(dataset)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    bs = batch_size if batch_size is not None else cfg.training.batch_size
    pw = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=num_workers, drop_last=True,
        pin_memory=pin_memory, persistent_workers=pw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=pw,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=pw,
    )

    logger.info("DataLoaders: train=%d, val=%d, test=%d (batch=%d, workers=%d)",
                len(train_ds), len(val_ds), len(test_ds), bs, num_workers)
    return train_loader, val_loader, test_loader
