"""Download and extraction utilities for the Seamless Interaction dataset.

The dataset is hosted on HuggingFace in WebDataset tar format and is ~27TB
total. These stubs support batch-by-batch download, extraction, and streaming
for memory-efficient processing.

HuggingFace repo: https://huggingface.co/datasets/facebook/seamless-interaction
Official tools: https://github.com/facebookresearch/seamless-interaction
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

BASE_HF_REPO: str = "facebook/seamless-interaction"


def download_archive(
    style: str,
    split: str,
    batch: str,
    archive: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Download a single tar archive from HuggingFace.

    HuggingFace URL pattern:
        {BASE_HF_REPO}/resolve/main/{style}/{split}/{batch}/{archive}.tar

    Args:
        style: "naturalistic" or "improvised".
        split: "train", "dev", "test", or "private".
        batch: Batch number string, e.g. "0001".
        archive: Archive number string within the batch, e.g. "0004".
        output_dir: Local directory to save the downloaded tar file.
        overwrite: If True, re-download even if the file already exists.

    Returns:
        Path to the downloaded .tar file.
    """
    raise NotImplementedError


def extract_archive(
    tar_path: Path,
    output_dir: Path,
    *,
    cleanup_tar: bool = False,
) -> Path:
    """Extract a tar archive to the target directory.

    Args:
        tar_path: Path to the .tar file.
        output_dir: Directory where the extracted files will be placed.
        cleanup_tar: If True, delete the tar file after successful extraction.

    Returns:
        Path to the extracted directory.
    """
    raise NotImplementedError


def stream_dataset(
    style: str,
    split: str,
    batch: str,
    archive_indices: list[int],
) -> Iterator[dict[str, Any]]:
    """Yield items from HuggingFace WebDataset streaming API.

    Enables memory-efficient processing without downloading full archives.
    Each yielded item contains bytes for mp4, wav, json, and npz files.

    Example usage::

        for item in stream_dataset("improvised", "dev", "0001", [0, 1, 2]):
            video_bytes = item["mp4"]
            audio_array = item["wav"]["array"]
            metadata = item["json"]
            features = item["npz"]

    Args:
        style: "naturalistic" or "improvised".
        split: "train", "dev", "test", or "private".
        batch: Batch number string, e.g. "0001".
        archive_indices: List of archive indices within the batch to stream.

    Yields:
        Dictionary with file contents keyed by extension.
    """
    raise NotImplementedError


def list_local_archives(
    style: str,
    split: str,
    datasets_root: Path,
) -> list[str]:
    """List archive numbers that have been downloaded and extracted locally.

    Scans the local directory structure for existing extracted directories.

    Args:
        style: "naturalistic" or "improvised".
        split: "train", "dev", "test", or "private".
        datasets_root: Root datasets directory (e.g. Path("datasets")).

    Returns:
        Sorted list of archive number strings, e.g. ["0001", "0004", "0012"].
    """
    raise NotImplementedError
