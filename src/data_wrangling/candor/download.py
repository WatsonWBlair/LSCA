"""CANDOR Corpus download utilities.

Downloads raw_media_part_XXX.zip files from pre-signed S3 URLs.
"""

from __future__ import annotations

import logging
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
