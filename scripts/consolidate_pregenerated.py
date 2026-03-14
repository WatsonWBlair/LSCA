#!/usr/bin/env python3
# scripts/consolidate_pregenerated.py
# Consolidate per-stem pregenerated .npy files into a single flat set of
# memory-mappable files for efficient training.
#
# Usage:
#   python scripts/consolidate_pregenerated.py \
#       --pregen-dir datasets/pregenerated/marlin-vit-base-ytf__wav2vec2-lv-60-espea \
#       --output-dir datasets/consolidated/marlin-vit-base-ytf__wav2vec2-lv-60-espea \
#       --workers 8
#
# Or via invoke:
#   invoke consolidate-pregenerated
#
# Peak RAM: proportional to --workers (each worker holds one stem, ~350 MB).
# Output files are written as np.memmap and can be opened with mmap_mode='r'
# during training — the OS pages in only the slices actually accessed, so
# training RAM stays near zero.
#
# Backbone mixing: always consolidate within a single backbone tag directory.
# Each backbone combination gets its own consolidated output directory,
# preserving the ability to switch or compare encoders.

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    p.add_argument(
        "--workers", default=3, type=int,
        help="Parallel threads for filling memmaps (default: 3). "
             "Each worker holds one stem in RAM (~350 MB). "
             "Set to 1 for sequential (lowest RAM).",
    )
    return p.parse_args()


def _find_stem_dirs(pregen_dir: Path) -> list[Path]:
    """Return sorted list of leaf directories that contain v_raw.npy."""
    return sorted(p.parent for p in pregen_dir.rglob("v_raw.npy") if p.is_file())


def _fill_stem(job: dict) -> dict:
    """Worker: load one stem and write its rows into the pre-allocated memmaps.

    Each stem writes to a non-overlapping slice of the output files, so
    concurrent writes from multiple threads are safe without any locking.
    Each thread opens its own np.memmap view (mode='r+') of the shared files.
    """
    sd        = job["stem_dir"]
    start     = job["start_row"]
    end       = job["end_row"]
    n         = end - start
    out_dir   = job["out_dir"]
    shapes    = job["shapes"]
    idx       = job["idx"]
    n_total_stems = job["n_total_stems"]

    session_name = sd.parent.name
    stem_name    = sd.name

    # Open output memmaps in 'r+' mode — appends to existing allocation.
    # Each thread gets its own view object; numpy delegates actual I/O to the OS.
    mm_v  = np.memmap(out_dir / "v_raw.npy",     dtype=np.float32, mode="r+", shape=shapes["v"])
    mm_ph = np.memmap(out_dir / "ph_raw.npy",    dtype=np.float32, mode="r+", shape=shapes["ph"])
    mm_pl = np.memmap(out_dir / "ph_labels.npy", dtype=np.int64,   mode="r+", shape=shapes["pl"])
    mm_pm = np.memmap(out_dir / "ph_mask.npy",   dtype=np.float32, mode="r+", shape=shapes["pm"])
    mm_p  = np.memmap(out_dir / "p_raw.npy",     dtype=np.float32, mode="r+", shape=shapes["p"])

    mm_v [start:end] = np.load(sd / "v_raw.npy",     mmap_mode="r")[:].astype(np.float32)
    mm_ph[start:end] = np.load(sd / "ph_raw.npy",    mmap_mode="r")[:].astype(np.float32)
    mm_pl[start:end] = np.load(sd / "ph_labels.npy", mmap_mode="r")[:].astype(np.int64)
    mm_pm[start:end] = np.load(sd / "ph_mask.npy",   mmap_mode="r")[:].astype(np.float32)
    mm_p [start:end] = np.load(sd / "p_raw.npy",     mmap_mode="r")[:].astype(np.float32)

    # Flush this thread's view; the OS write-back is not dependent on other threads.
    del mm_v, mm_ph, mm_pl, mm_pm, mm_p

    logger.info(
        "[%d/%d] %s/%s  (%d chunks → rows %d:%d)",
        idx, n_total_stems, session_name, stem_name, n, start, end,
    )

    return {
        "session":   session_name,
        "stem":      stem_name,
        "start_row": start,
        "end_row":   end,
        "n_chunks":  n,
    }


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
    stem_meta = []
    for sd in stem_dirs:
        n = np.load(sd / "v_raw.npy", mmap_mode="r").shape[0]
        if n > 0:
            stem_meta.append({"stem_dir": sd, "n_chunks": n})

    n_total = sum(m["n_chunks"] for m in stem_meta)
    logger.info("Total chunks: %d across %d non-empty stems", n_total, len(stem_meta))

    # Infer shapes from the first stem
    first   = stem_meta[0]["stem_dir"]
    v_shape = np.load(first / "v_raw.npy",     mmap_mode="r").shape[1:]
    ph_shape= np.load(first / "ph_raw.npy",    mmap_mode="r").shape[1:]
    pl_shape= np.load(first / "ph_labels.npy", mmap_mode="r").shape[1:]
    pm_shape= np.load(first / "ph_mask.npy",   mmap_mode="r").shape[1:]
    p_shape = np.load(first / "p_raw.npy",     mmap_mode="r").shape[1:]

    logger.info(
        "Shapes: v=%s  ph=%s  labels=%s  mask=%s  prosody=%s",
        v_shape, ph_shape, pl_shape, pm_shape, p_shape,
    )

    full_shapes = {
        "v":  (n_total, *v_shape),
        "ph": (n_total, *ph_shape),
        "pl": (n_total, *pl_shape),
        "pm": (n_total, *pm_shape),
        "p":  (n_total, *p_shape),
    }

    # ------------------------------------------------------------------ #
    # Pass 2: pre-allocate output memmaps                                  #
    # ------------------------------------------------------------------ #
    logger.info("Allocating output files in %s ...", output_dir)
    np.memmap(output_dir / "v_raw.npy",     dtype=np.float32, mode="w+", shape=full_shapes["v"])
    np.memmap(output_dir / "ph_raw.npy",    dtype=np.float32, mode="w+", shape=full_shapes["ph"])
    np.memmap(output_dir / "ph_labels.npy", dtype=np.int64,   mode="w+", shape=full_shapes["pl"])
    np.memmap(output_dir / "ph_mask.npy",   dtype=np.float32, mode="w+", shape=full_shapes["pm"])
    np.memmap(output_dir / "p_raw.npy",     dtype=np.float32, mode="w+", shape=full_shapes["p"])
    logger.info("Allocation complete. Beginning parallel fill with %d workers ...", args.workers)

    # ------------------------------------------------------------------ #
    # Pass 3: fill memmaps in parallel                                     #
    # Each stem writes to a pre-computed non-overlapping row range —       #
    # no locking needed.                                                   #
    # ------------------------------------------------------------------ #
    cursor = 0
    jobs = []
    for idx, meta in enumerate(stem_meta, start=1):
        n = meta["n_chunks"]
        jobs.append({
            "stem_dir":      meta["stem_dir"],
            "start_row":     cursor,
            "end_row":       cursor + n,
            "out_dir":       output_dir,
            "shapes":        full_shapes,
            "idx":           idx,
            "n_total_stems": len(stem_meta),
        })
        cursor += n

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fill_stem, job): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())

    # Re-sort manifest by start_row so it reflects original stem order
    results.sort(key=lambda r: r["start_row"])

    # ------------------------------------------------------------------ #
    # Write manifest                                                        #
    # ------------------------------------------------------------------ #
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    logger.info("Done. %d chunks consolidated to %s", n_total, output_dir)
    logger.info("Files: v_raw.npy  ph_raw.npy  ph_labels.npy  ph_mask.npy  p_raw.npy  manifest.jsonl")
    logger.info("Point --feature-dir at %s when training.", output_dir)


if __name__ == "__main__":
    main()
