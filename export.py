# encoding/export.py
# Shape-validated .npy export helpers for CAMELS v8.1.

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from config import CAMELSConfig

logger = logging.getLogger(__name__)


def validate_embedding_shape(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    name: str,
):
    """Assert tensor shape matches expected. Logs error without raising."""
    actual = tuple(tensor.shape)
    if actual != expected_shape:
        logger.error(
            "Shape validation failed for %s: got %s, expected %s",
            name, actual, expected_shape,
        )
        return False
    return True


def validate_row_sync(output_dir: str, chunk_id: int, cfg: CAMELSConfig) -> bool:
    """Verify all export files have chunk_id + 1 rows."""
    expected = chunk_id + 1
    ok = True
    for fname in [cfg.export.zv_file, cfg.export.zp_file, cfg.export.zph_file]:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            logger.error("Row sync: %s missing at chunk %d", fname, chunk_id)
            ok = False
            continue
        arr = np.load(path)
        if arr.shape[0] != expected:
            logger.error("Row sync: %s has %d rows, expected %d", fname, arr.shape[0], expected)
            ok = False
    return ok


def validate_export_dimensions(output_dir: str, cfg: CAMELSConfig) -> bool:
    """Verify all export files have correct latent dimensions."""
    ok = True
    d = cfg.latent.d_latent
    max_ph = cfg.latent.max_phones

    checks = [
        (cfg.export.zv_file, lambda s: s[-1] == d, f"last dim should be {d}"),
        (cfg.export.zp_file, lambda s: s[-1] == d, f"last dim should be {d}"),
        (cfg.export.zph_file, lambda s: len(s) == 3 and s[1] == max_ph and s[2] == d,
         f"shape should be (N, {max_ph}, {d})"),
    ]

    for fname, check_fn, msg in checks:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            continue
        arr = np.load(path)
        if not check_fn(arr.shape):
            logger.error("Dimension validation failed for %s: shape=%s, %s", fname, arr.shape, msg)
            ok = False

    return ok
