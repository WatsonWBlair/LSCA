"""Repackage raw Seamless Interaction files into an interaction-centric layout.

Transforms the raw layout (flat files in batch directories with opaque names)
into a hierarchy organized by interaction, where each speaker gets a
subdirectory with canonical filenames.

Raw layout:
    datasets/seamless_interaction/{style}/{split}/{batch}/{stem}.{ext}

Wrangled layout:
    wrangled/seamless_interaction/{style}/{split}/{interaction_key}/{participant_id}/
        video_cropped.mp4
        audio.wav
        transcript.json
        features.npz
"""

from __future__ import annotations

from pathlib import Path

from src.data_wrangling.types import InteractionPair, SpeakerFileSet

DEFAULT_WRANGLED_ROOT: Path = Path("wrangled/seamless_interaction")


def compute_output_paths(
    interaction: InteractionPair,
    wrangled_root: Path = DEFAULT_WRANGLED_ROOT,
) -> dict[str, Path]:
    """Compute the target output paths for all files in an interaction.

    Returns a mapping from original source path strings to their new
    destination Path objects following the wrangled directory convention.

    Args:
        interaction: The InteractionPair to compute paths for.
        wrangled_root: Root directory for wrangled output.

    Returns:
        Dict mapping original file path strings to new destination Paths.
    """
    raise NotImplementedError


def repackage_speaker(
    speaker: SpeakerFileSet,
    output_dir: Path,
    *,
    crop_video: bool = True,
) -> Path:
    """Copy (and optionally crop) a single speaker's files to the output dir.

    Copies .wav, .json, .npz directly with canonical names. If crop_video
    is True, runs the video through the cropping pipeline; otherwise copies
    the raw .mp4.

    Args:
        speaker: The SpeakerFileSet with source file paths.
        output_dir: Target directory for this speaker's files
            (e.g. wrangled/.../S1212_I00000389/P1437/).
        crop_video: Whether to apply video cropping during repackaging.

    Returns:
        Path to the output directory containing the repackaged files.
    """
    raise NotImplementedError


def repackage_interaction(
    interaction: InteractionPair,
    wrangled_root: Path = DEFAULT_WRANGLED_ROOT,
    *,
    crop_video: bool = True,
) -> Path:
    """Repackage all speakers of an interaction into the wrangled structure.

    Args:
        interaction: The InteractionPair to repackage.
        wrangled_root: Root directory for wrangled output.
        crop_video: Whether to apply video cropping.

    Returns:
        Path to the interaction directory
        (e.g. wrangled/.../S1212_I00000389/).
    """
    raise NotImplementedError


def repackage_batch(
    batch_dir: Path,
    style: str,
    split: str,
    batch: str,
    wrangled_root: Path = DEFAULT_WRANGLED_ROOT,
    *,
    crop_video: bool = True,
) -> list[Path]:
    """Repackage all interactions in a batch directory.

    Args:
        batch_dir: Path to the raw batch directory.
        style: "naturalistic" or "improvised".
        split: "train", "dev", "test", or "private".
        batch: Batch number string.
        wrangled_root: Root directory for wrangled output.
        crop_video: Whether to apply video cropping.

    Returns:
        List of paths to repackaged interaction directories.
    """
    raise NotImplementedError
