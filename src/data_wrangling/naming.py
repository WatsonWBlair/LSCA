"""Parse and generate filenames following the Seamless Interaction naming convention.

Convention: V{vendor}_S{session_id}_I{interaction_id}_P{participant_id}.{ext}
Example:    V03_S1212_I00000389_P1437.mp4
"""

from __future__ import annotations

import re
from pathlib import Path

from src.data_wrangling.types import FileIdentifier, InteractionPair, SpeakerFileSet

FILENAME_PATTERN: re.Pattern[str] = re.compile(
    r"^V(\d+)_S(\d+)_I(\d+)_P(\d+)$"
)

EXPECTED_EXTENSIONS: set[str] = {".mp4", ".wav", ".json", ".npz"}


def parse_filename(stem: str) -> FileIdentifier:
    """Parse a filename stem into its structured components.

    Args:
        stem: Filename without extension, e.g. "V03_S1212_I00000389_P1437".

    Returns:
        FileIdentifier with parsed vendor, session_id, interaction_id,
        participant_id.

    Raises:
        ValueError: If the stem does not match the expected naming convention.
    """
    raise NotImplementedError


def build_filename(identifier: FileIdentifier, extension: str) -> str:
    """Construct a filename from structured components.

    Args:
        identifier: The parsed file identifier.
        extension: File extension including the dot, e.g. ".mp4".

    Returns:
        Complete filename string, e.g. "V03_S1212_I00000389_P1437.mp4".
    """
    raise NotImplementedError


def collect_speaker_file_set(directory: Path, stem: str) -> SpeakerFileSet:
    """Collect the four expected files (.mp4, .wav, .json, .npz) for a speaker.

    Args:
        directory: Path to the batch directory containing the files.
        stem: The common filename stem.

    Returns:
        SpeakerFileSet with paths to all four files.

    Raises:
        FileNotFoundError: If any of the four expected files is missing.
    """
    raise NotImplementedError


def discover_interaction_pairs(
    batch_dir: Path,
    style: str,
    split: str,
    batch: str,
) -> list[InteractionPair]:
    """Scan a batch directory and group files into InteractionPair objects.

    Files sharing the same S{session}_I{interaction} key but different
    P{participant} values are paired together. Interactions with only one
    speaker locally available will have speaker_b set to None.

    Args:
        batch_dir: Path to a batch directory (e.g. datasets/.../dev/0004/).
        style: "naturalistic" or "improvised".
        split: "train", "dev", "test", or "private".
        batch: Batch number string, e.g. "0004".

    Returns:
        List of InteractionPair objects found in the directory.
    """
    raise NotImplementedError
