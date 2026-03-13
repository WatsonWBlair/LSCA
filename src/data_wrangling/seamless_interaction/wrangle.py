"""Wrangling utilities for Seamless Interaction dataset."""

import csv
import json
import re
import shutil
from pathlib import Path

from src.data_wrangling.seamless_interaction.crop import process_interaction
from src.data_wrangling.seamless_interaction.download import (
    download_pairs_iter,
    download_sessions_iter,
)

INTERACTIONS_CSV = Path(
    "C:/Users/watso/Development/seamless_interaction/assets/interactions.csv"
)

FILENAME_PATTERN = re.compile(r"^V(\d+)_S(\d+)_I(\d+)_P(\w+)$")
SOURCE_DIR = Path('datasets/seamless_interaction')
OUTPUT_DIR = Path('datasets/wrangled')


def load_interactions_lookup() -> dict[str, dict]:
    """Read interactions.csv once and return lookup by 8-digit zero-padded I-number.

    Returns:
        dict keyed by prompt_hash string (e.g. "00000131") → {prompt_a, prompt_b,
        ipc_a, ipc_b, interaction_type}
    """
    lookup: dict[str, dict] = {}
    with open(INTERACTIONS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["prompt_hash"].zfill(8)
            lookup[key] = {
                "prompt_a": row["participant_a_prompt_text"],
                "prompt_b": row["participant_b_prompt_text"],
                "ipc_a": row["ipc_a"],
                "ipc_b": row["ipc_b"],
                "interaction_type": row["interaction_type"],
            }
    return lookup


def _process_npz(npz_path: Path, extra_fields: dict | None = None) -> bool:
    """Crop, copy, and cleanup one staged NPZ file. Returns True on success."""
    file_id = npz_path.stem
    match = FILENAME_PATTERN.match(file_id)
    if not match:
        return False

    _, session, interaction, participant = match.groups()
    short_name = f"I{interaction}_P{participant}"
    session_dir = OUTPUT_DIR / f"S{session}"

    if not (session_dir / f"{short_name}.mp4").exists():
        video_path = npz_path.with_suffix('.mp4')
        if not video_path.exists():
            print(f"Skipping {file_id}: no .mp4 alongside NPZ (V01 format?)")
            for ext in ['.mp4', '.wav', '.json', '.npz']:
                src = npz_path.with_suffix(ext)
                if src.exists():
                    src.unlink()
            return False

        print(f"Processing: {file_id}")
        try:
            process_interaction(npz_path, session_dir, short_name)
        except Exception as e:
            print(f"Warning: could not process {file_id}: {e} — skipping")
            for ext in ['.mp4', '.wav', '.json', '.npz']:
                src = npz_path.with_suffix(ext)
                if src.exists():
                    src.unlink()
            return False

        for ext in ['.wav', '.json', '.npz']:
            src = npz_path.with_suffix(ext)
            if not src.exists():
                continue
            dst = session_dir / f"{short_name}{ext}"
            if ext == '.json' and extra_fields:
                with open(src, encoding='utf-8') as f:
                    data = json.load(f)
                data.update(extra_fields)
                with open(dst, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
            else:
                shutil.copy2(src, dst)

    # Cleanup source files
    for ext in ['.mp4', '.wav', '.json', '.npz']:
        src = npz_path.with_suffix(ext)
        if src.exists():
            src.unlink()
    return True


def wrangle_seamless(count: int, style: str = "improvised", split: str = "dev"):
    """Download, crop, and cleanup Seamless pairs one at a time."""
    print(f"Processing {count} pair(s)...")

    for pair in download_pairs_iter(style, split, num_pairs=count):
        for file_id in pair:
            found = list((SOURCE_DIR / style / split).rglob(f"{file_id}.npz"))
            if not found:
                continue
            _process_npz(found[0])

    # Remove empty directories
    for path in sorted(SOURCE_DIR.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def wrangle_seamless_sessions(
    num_sessions: int = 28,
    style: str = "improvised",
    split: str = "dev",
):
    """Download, crop, and cleanup Seamless sessions in chronological order.

    Processes sessions one interaction at a time (clean-as-you-go). Attaches
    session metadata (position, prompts, IPC codes) to each output JSON.
    """
    interactions_lookup = load_interactions_lookup()
    print(f"Loaded {len(interactions_lookup)} interaction prompt records")

    session_idx_counter: dict[str, int] = {}

    for session_key, interaction_key, file_ids, session_total in download_sessions_iter(
        style, split, num_sessions=num_sessions
    ):
        # Track 0-indexed position of this interaction within its session
        if session_key not in session_idx_counter:
            session_idx_counter[session_key] = 0
        session_interaction_idx = session_idx_counter[session_key]
        session_idx_counter[session_key] += 1

        # Lookup prompt info by zero-padded I-number (strip leading "I")
        i_num_padded = interaction_key[1:].zfill(8)
        prompt_info = interactions_lookup.get(i_num_padded, {})

        extra_fields = {
            "session_interaction_idx": session_interaction_idx,
            "session_total_interactions": session_total,
            "session_id": session_key,
            **prompt_info,
        }

        for file_id in file_ids:
            found = list((SOURCE_DIR / style / split).rglob(f"{file_id}.npz"))
            if not found:
                continue
            _process_npz(found[0], extra_fields=extra_fields)

    # Remove empty directories
    for path in sorted(SOURCE_DIR.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def wrangle_staged_seamless(style: str = "improvised", split: str = "dev"):
    """Process all NPZs already staged in SOURCE_DIR/style/split/ (backfill)."""
    base = SOURCE_DIR / style / split
    if not base.exists():
        print(f"No staged data found at {base}")
        return

    staged_npzs = list(base.rglob("*.npz"))
    print(f"Found {len(staged_npzs)} staged NPZ(s) to process...")

    processed = sum(_process_npz(npz_path) for npz_path in staged_npzs)

    # Remove empty directories
    for path in sorted(SOURCE_DIR.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()

    print(f"Done. Processed {processed}/{len(staged_npzs)} staged file(s).")
