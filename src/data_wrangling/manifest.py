"""Master JSON manifest generation for programmatic segment stitching.

The manifest is the central index that allows downstream code to reconstruct
conversations by looking up interaction pairs, their file locations, and
temporal segment information (VAD-based stitching order).

Manifest JSON schema:
    {
      "version": "1.0",
      "generated_at": "<ISO timestamp>",
      "dataset": "seamless_interaction",
      "statistics": {
        "total_interactions": N,
        "styles": ["improvised", ...],
        "splits": ["dev", ...]
      },
      "interactions": {
        "<interaction_key>": {
          "style": "...",
          "split": "...",
          "batch": "...",
          "speakers": {
            "<participant_id>": {
              "original_stem": "...",
              "files": {
                "video_cropped": "<relative path>",
                "audio": "<relative path>",
                "transcript": "<relative path>",
                "features": "<relative path>"
              },
              "duration_seconds": float | null,
              "num_transcript_segments": int,
              "num_vad_segments": int
            }
          },
          "stitching_order": [
            {"speaker": "<participant_id>", "start": float, "end": float}
          ]
        }
      }
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data_wrangling.types import (
    InteractionPair,
    ManifestEntry,
    TranscriptSegment,
    VADSegment,
)


def load_transcript_metadata(
    json_path: Path,
) -> tuple[list[TranscriptSegment], list[VADSegment]]:
    """Parse a speaker's JSON metadata file into structured segments.

    The JSON structure (confirmed from actual data) is::

        {
            "id": "V03_S1212_I00000389_P1437",
            "metadata:transcript": [
                {"words": [...], "start": float, "end": float, "transcript": str}
            ],
            "metadata:vad": [
                {"start": float, "end": float}
            ]
        }

    Args:
        json_path: Path to the speaker's .json metadata file.

    Returns:
        Tuple of (transcript_segments, vad_segments).
    """
    raise NotImplementedError


def build_manifest_entry(
    interaction: InteractionPair,
    speaker_key: str,
    wrangled_root: Path,
) -> ManifestEntry:
    """Build a single ManifestEntry for one speaker in an interaction.

    Args:
        interaction: The InteractionPair containing this speaker.
        speaker_key: "a" or "b" to select which speaker.
        wrangled_root: Root directory for wrangled output (used to compute
            relative paths).

    Returns:
        A populated ManifestEntry.
    """
    raise NotImplementedError


def build_manifest(
    interactions: list[InteractionPair],
    wrangled_root: Path,
) -> dict[str, Any]:
    """Build the complete manifest dictionary from all processed interactions.

    The stitching_order for each interaction is constructed by merging VAD
    segments from both speakers, sorted chronologically. This enables
    downstream code to reconstruct the turn-taking flow of the conversation.

    Args:
        interactions: All InteractionPair objects to include.
        wrangled_root: Root directory for wrangled output.

    Returns:
        The manifest as a JSON-serializable dictionary.
    """
    raise NotImplementedError


def write_manifest(
    manifest: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write the manifest dictionary to a JSON file.

    Args:
        manifest: The manifest dictionary (from build_manifest).
        output_path: Path where the JSON file will be written.

    Returns:
        The output_path.
    """
    raise NotImplementedError
