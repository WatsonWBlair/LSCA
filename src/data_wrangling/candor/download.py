"""CANDOR Corpus download utilities.

Downloads raw_media_part_XXX.zip files from pre-signed S3 URLs.
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def load_urls(urls_file: Path) -> list[str]:
    """Load URLs from a space-separated file.

    Args:
        urls_file: Path to file containing space-separated URLs.

    Returns:
        List of URL strings.
    """
    with open(urls_file) as f:
        content = f.read().strip()
    return content.split()


def get_part_name(url: str) -> str:
    """Extract the part name from a URL.

    Args:
        url: Pre-signed S3 URL.

    Returns:
        Filename like 'raw_media_part_001.zip'.
    """
    # URL format: .../raw_media_part_XXX.zip?X-Amz-...
    path_part = url.split('?')[0]
    return path_part.split('/')[-1]


def download_part(url: str, output_dir: Path) -> Path:
    """Download a single zip file.

    Args:
        url: Pre-signed S3 URL to download.
        output_dir: Directory to save the file.

    Returns:
        Path to the downloaded zip file.
    """
    filename = get_part_name(url)
    output_path = output_dir / filename

    if output_path.exists():
        logger.info(f"Skipping {filename} (already exists)")
        return output_path

    logger.info(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, output_path)
    logger.info(f"Downloaded {filename}")

    return output_path


def extract_part(zip_path: Path, extract_dir: Path) -> None:
    """Extract a zip file.

    Args:
        zip_path: Path to the zip file.
        extract_dir: Directory to extract contents into.
    """
    logger.info(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    logger.info(f"Extracted {zip_path.name}")


def download_candor(
    urls_file: Path,
    output_dir: Path,
    start: int = 1,
    count: int | None = None,
    extract: bool = False,
) -> None:
    """Download CANDOR dataset parts.

    Args:
        urls_file: Path to file containing pre-signed S3 URLs.
        output_dir: Directory to save downloads and extractions.
        start: Part number to start from (1-indexed, default: 1).
        count: Number of parts to download (default: all remaining).
        extract: Whether to extract zip files after download (default: False).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = load_urls(urls_file)
    logger.info(f"Found {len(urls)} URLs in {urls_file.name}")

    # Select range of URLs (1-indexed start)
    start_idx = start - 1
    if count is not None:
        end_idx = start_idx + count
        urls = urls[start_idx:end_idx]
    else:
        urls = urls[start_idx:]

    logger.info(f"Downloading parts {start} to {start + len(urls) - 1}")

    for url in urls:
        zip_path = download_part(url, output_dir)

        if extract:
            # Extract if not already extracted
            part_name = zip_path.stem  # e.g., 'raw_media_part_001'
            extract_marker = output_dir / f"{part_name}_extracted"

            if not extract_marker.exists():
                extract_part(zip_path, output_dir)
                extract_marker.touch()  # Mark as extracted
            else:
                logger.info(f"Skipping extraction of {zip_path.name} (already extracted)")

    logger.info("CANDOR download complete")


def process_part(zip_path: Path, output_dir: Path) -> bool:
    """Extract, process audio, and cleanup for a single downloaded part.

    Args:
        zip_path: Path to the already-downloaded zip file.
        output_dir: Directory for extraction and processing.

    Returns:
        True if processing occurred, False if already complete.
    """
    from src.data_wrangling.candor.extract import (
        cleanup_extras,
        extract_conversation_audio,
    )

    part_name = zip_path.stem  # e.g., 'raw_media_part_001'

    wrangled_marker = output_dir / f"{part_name}_wrangled"
    if wrangled_marker.exists():
        logger.info(f"Skipping {part_name} (already wrangled)")
        return False

    # Extract zip if not already extracted
    extracted_marker = output_dir / f"{part_name}_extracted"
    if not extracted_marker.exists():
        extract_part(zip_path, output_dir)
        extracted_marker.touch()

    # Process all conversations with raw/ directories
    for conv_dir in output_dir.iterdir():
        if not conv_dir.is_dir() or conv_dir.name.startswith('raw_media_part'):
            continue
        raw_dir = conv_dir / "raw"
        if not raw_dir.exists():
            continue  # Already cleaned

        # Extract audio (with correct user_id naming)
        extract_conversation_audio(conv_dir)

        # Remove extra files (combined video/audio, thumbnail, etc.)
        cleanup_extras(conv_dir)

        # Wrangle into datasets/wrangled/
        from src.data_wrangling.candor.wrangle import wrangle_conversation
        part_num = zip_path.stem.split("_")[-1]  # 'raw_media_part_004' -> '004'
        wrangle_conversation(conv_dir, part_num, Path("datasets/wrangled"))

        # Remove raw/ directory
        shutil.rmtree(raw_dir)
        logger.info(f"Removed {raw_dir}")

    wrangled_marker.touch()
    logger.info(f"Completed {part_name}")
    return True


def wrangle_candor(
    output_dir: Path,
    start: int = 1,
    count: int | None = None,
) -> int:
    """Iteratively extract, process, and cleanup downloaded CANDOR parts.

    For each zip found in output_dir: extracts, processes audio, removes raw/.
    Supports resume via marker files.

    Args:
        output_dir: Directory containing downloaded zip files.
        start: Part number to start from (1-indexed, based on sorted zip list).
        count: Number of parts to process (default: all remaining).

    Returns:
        Number of parts processed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(output_dir.glob("*.zip"))
    logger.info(f"Found {len(zip_files)} zip file(s) in {output_dir}")

    # Select range (1-indexed start)
    start_idx = start - 1
    if count is not None:
        zip_files = zip_files[start_idx:start_idx + count]
    else:
        zip_files = zip_files[start_idx:]

    if not zip_files:
        logger.info("No zip files to process.")
        return 0

    logger.info(f"Processing {len(zip_files)} part(s) starting from index {start}")

    processed = 0
    for i, zip_path in enumerate(zip_files, start=start):
        logger.info(f"--- Part {i}: {zip_path.name} ---")
        if process_part(zip_path, output_dir):
            processed += 1

    logger.info(f"Wrangling complete. Processed {processed} part(s).")
    return processed
