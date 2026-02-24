"""Wrangling utilities for Seamless Interaction dataset."""

import re
import shutil
from pathlib import Path

from src.data_wrangling.seamless_interaction.crop import process_interaction
from src.data_wrangling.seamless_interaction.download import download_pairs_iter

FILENAME_PATTERN = re.compile(r"^V(\d+)_S(\d+)_I(\d+)_P(\w+)$")
SOURCE_DIR = Path('datasets/seamless_interaction')
OUTPUT_DIR = Path('datasets/wrangled')


def wrangle_seamless(count: int, style: str = "improvised", split: str = "dev"):
    """Download, crop, and cleanup Seamless pairs one at a time."""
    print(f"Processing {count} pair(s)...")

    for pair in download_pairs_iter(style, split, num_pairs=count):
        for file_id in pair:
            npz_path = SOURCE_DIR / style / split / f"{file_id}.npz"
            if not npz_path.exists():
                continue

            match = FILENAME_PATTERN.match(file_id)
            if not match:
                continue

            _, session, interaction, participant = match.groups()
            short_name = f"I{interaction}_P{participant}"
            session_dir = OUTPUT_DIR / f"S{session}"

            if not (session_dir / f"{short_name}.mp4").exists():
                print(f"Processing: {file_id}")
                process_interaction(npz_path, session_dir, short_name)
                for ext in ['.wav', '.json', '.npz']:
                    src = npz_path.with_suffix(ext)
                    if src.exists():
                        shutil.copy2(src, session_dir / f"{short_name}{ext}")

            # Cleanup source files
            for ext in ['.mp4', '.wav', '.json', '.npz']:
                src = npz_path.with_suffix(ext)
                if src.exists():
                    src.unlink()

    # Remove empty directories
    for path in sorted(SOURCE_DIR.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
