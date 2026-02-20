"""Top-level orchestrator for Seamless Interaction dataset wrangling.

Ties together discovery, cropping, repackaging, and manifest generation
into a single pipeline entry point. Called from the Invoke task system
(tasks.py).

Pipeline steps:
    1. Discover all interaction pairs in the raw directory tree.
    2. For each interaction, repackage files into the wrangled structure.
    3. Optionally crop videos to webcam-style framing using NPZ keypoints.
    4. Generate and write the master JSON manifest.
"""

from __future__ import annotations

from pathlib import Path

from src.data_wrangling.types import InteractionPair

DEFAULT_RAW_ROOT: Path = Path("datasets/seamless_interaction")
DEFAULT_WRANGLED_ROOT: Path = Path("wrangled/seamless_interaction")


def discover_all_interactions(
    raw_root: Path = DEFAULT_RAW_ROOT,
    *,
    styles: list[str] | None = None,
    splits: list[str] | None = None,
) -> list[InteractionPair]:
    """Discover all interaction pairs across the raw dataset.

    Walks the directory tree: {raw_root}/{style}/{split}/{batch}/
    and groups files into InteractionPair objects.

    Args:
        raw_root: Root of the raw Seamless Interaction dataset.
        styles: Filter to specific styles. Default: all found on disk.
        splits: Filter to specific splits. Default: all found on disk.

    Returns:
        List of all InteractionPair objects discovered.
    """
    raise NotImplementedError


def wrangle_seamless_interaction(
    raw_root: Path = DEFAULT_RAW_ROOT,
    wrangled_root: Path = DEFAULT_WRANGLED_ROOT,
    *,
    styles: list[str] | None = None,
    splits: list[str] | None = None,
    crop_video: bool = True,
    write_manifest: bool = True,
) -> Path:
    """Run the full wrangling pipeline for the Seamless Interaction dataset.

    Args:
        raw_root: Root of the raw dataset.
        wrangled_root: Root directory for wrangled output.
        styles: Filter to specific styles. Default: all.
        splits: Filter to specific splits. Default: all.
        crop_video: Whether to apply video cropping.
        write_manifest: Whether to generate the manifest.json file.

    Returns:
        Path to the wrangled root directory.
    """
    raise NotImplementedError
