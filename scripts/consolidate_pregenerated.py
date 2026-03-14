#!/usr/bin/env python3
# scripts/consolidate_pregenerated.py
# Consolidate per-stem pregenerated .npy files into a single flat set of
# memory-mappable files for efficient training.
#
# Usage:
#   python scripts/consolidate_pregenerated.py \
#       --pregen-dir datasets/pregenerated/marlin-vit-base-ytf__wav2vec2-lv-60-espea \
#       --output-dir datasets/consolidated/marlin-vit-base-ytf__wav2vec2-lv-60-espea
#
# Or via invoke:
#   invoke consolidate-pregenerated
#
# Peak RAM: one stem at a time (~350 MB max). Output files are written as
# np.memmap and can be opened with mmap_mode='r' during training — the OS
# pages in only the slices actually accessed, so training RAM stays near zero.
#
# Backbone mixing: always consolidate within a single backbone tag directory.
# Each backbone combination gets its own consolidated output directory,
# preserving the ability to switch or compare encoders.

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("consolidate")


def parse_args():
    p = argparse.ArgumentParser(
        description="Consolidate per-stem pregenerated .npy files into flat mmap files."
    )
    p.add_argument(
        "--pregen-dir", required=True,
        help="Root of a single backbone tag directory, e.g. "
             "datasets/pregenerated/marlin-vit-base-ytf__wav2vec2-lv-60-espea",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Destination directory for consolidated files.",
    )
    return p.parse_args()


def _find_stem_dirs(pregen_dir: Path) -> list[Path]:
    """Return sorted list of leaf directories that contain v_raw.npy."""
    return sorted(p.parent for p in pregen_dir.rglob("v_raw.npy") if p.is_file())


def main():
    args = parse_args()
    pregen_dir = Path(args.pregen_dir)
    output_dir = Path(args.output_dir)

    if not pregen_dir.exists():
        logger.error("pregen-dir does not exist: %s", pregen_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Pass 1: discover stems and count total chunks                        #
    # ------------------------------------------------------------------ #
    logger.info("Scanning %s ...", pregen_dir)
    stem_dirs = _find_stem_dirs(pregen_dir)
    if not stem_dirs:
        logger.error("No v_raw.npy files found under %s", pregen_dir)
        sys.exit(1)

    logger.info("Found %d stems — counting chunks ...", len(stem_dirs))
    stem_meta = []  # list of {stem_dir, n_chunks}
    for sd in stem_dirs:
        n = np.load(sd / "v_raw.npy", mmap_mode="r").shape[0]
        if n > 0:
            stem_meta.append({"stem_dir": sd, "n_chunks": n})

    n_total = sum(m["n_chunks"] for m in stem_meta)
    logger.info("Total chunks: %d across %d non-empty stems", n_total, len(stem_meta))

    # Infer shapes from the first stem
    first = stem_meta[0]["stem_dir"]
    v_shape  = np.load(first / "v_raw.npy",      mmap_mode="r").shape[1:]   # (d_video,)
    ph_shape = np.load(first / "ph_raw.npy",     mmap_mode="r").shape[1:]   # (max_phones, d_phoneme)
    pl_shape = np.load(first / "ph_labels.npy",  mmap_mode="r").shape[1:]   # (max_phones,)
    pm_shape = np.load(first / "ph_mask.npy",    mmap_mode="r").shape[1:]   # (max_phones,)
    p_shape  = np.load(first / "p_raw.npy",      mmap_mode="r").shape[1:]   # (d_prosody,)

    logger.info(
        "Shapes: v=%s  ph=%s  labels=%s  mask=%s  prosody=%s",
        v_shape, ph_shape, pl_shape, pm_shape, p_shape,
    )

    # ------------------------------------------------------------------ #
    # Pass 2: pre-allocate output memmaps                                  #
    # ------------------------------------------------------------------ #
    def _alloc(name, shape, dtype):
        path = output_dir / name
        arr = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
        return arr

    logger.info("Allocating output files in %s ...", output_dir)
    out_v  = _alloc("v_raw.npy",      (n_total, *v_shape),  np.float32)
    out_ph = _alloc("ph_raw.npy",     (n_total, *ph_shape), np.float32)
    out_pl = _alloc("ph_labels.npy",  (n_total, *pl_shape), np.int64)
    out_pm = _alloc("ph_mask.npy",    (n_total, *pm_shape), np.float32)
    out_p  = _alloc("p_raw.npy",      (n_total, *p_shape),  np.float32)

    # ------------------------------------------------------------------ #
    # Pass 3: fill memmaps one stem at a time                              #
    # ------------------------------------------------------------------ #
    manifest = []
    cursor = 0

    for idx, meta in enumerate(stem_meta):
        sd = meta["stem_dir"]
        n = meta["n_chunks"]
        session_name = sd.parent.name
        stem_name = sd.name

        logger.info(
            "[%d/%d] %s/%s  (%d chunks → rows %d:%d)",
            idx + 1, len(stem_meta), session_name, stem_name, n, cursor, cursor + n,
        )

        out_v [cursor:cursor + n] = np.load(sd / "v_raw.npy",     mmap_mode="r")[:].astype(np.float32)
        out_ph[cursor:cursor + n] = np.load(sd / "ph_raw.npy",    mmap_mode="r")[:].astype(np.float32)
        out_pl[cursor:cursor + n] = np.load(sd / "ph_labels.npy", mmap_mode="r")[:].astype(np.int64)
        out_pm[cursor:cursor + n] = np.load(sd / "ph_mask.npy",   mmap_mode="r")[:].astype(np.float32)
        out_p [cursor:cursor + n] = np.load(sd / "p_raw.npy",     mmap_mode="r")[:].astype(np.float32)

        manifest.append({
            "session":   session_name,
            "stem":      stem_name,
            "start_row": cursor,
            "end_row":   cursor + n,
            "n_chunks":  n,
        })
        cursor += n

    # Flush all memmaps to disk
    del out_v, out_ph, out_pl, out_pm, out_p

    # ------------------------------------------------------------------ #
    # Write manifest                                                        #
    # ------------------------------------------------------------------ #
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    logger.info(
        "Done. %d chunks consolidated to %s", n_total, output_dir,
    )
    logger.info(
        "Files: v_raw.npy  ph_raw.npy  ph_labels.npy  ph_mask.npy  p_raw.npy  manifest.jsonl",
    )
    logger.info(
        "Point --feature-dir at %s when training.", output_dir,
    )


if __name__ == "__main__":
    main()
