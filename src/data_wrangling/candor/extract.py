"""Audio extraction utilities for CANDOR dataset.

Extracts per-participant audio from raw MKV files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)


def load_filename_mapping(conversation_dir: Path) -> dict[str, str]:
    """Load mapping from MKV filename to user_id from metadata.json.

    Args:
        conversation_dir: Path to conversation directory.

    Returns:
        Dict mapping MKV filename to user_id (Prolific ID).
    """
    metadata_path = conversation_dir / "metadata.json"
    if not metadata_path.exists():
        logger.warning(f"No metadata.json in {conversation_dir.name}")
        return {}

    with open(metadata_path) as f:
        metadata = json.load(f)

    mapping = {}
    for speaker in metadata.get("speakers", []):
        user_id = speaker.get("user_id")
        for file_info in speaker.get("files", []):
            mkv_name = file_info.get("filename")
            if mkv_name and user_id:
                mapping[mkv_name] = user_id
    return mapping


def extract_audio(mkv_path: Path, output_path: Path) -> Path:
    """Extract audio from an MKV file to WAV format.

    Args:
        mkv_path: Path to source MKV file.
        output_path: Path for output WAV file.

    Returns:
        Path to the extracted audio file.
    """
    try:
        (
            ffmpeg
            .input(str(mkv_path))
            .output(str(output_path), acodec='pcm_s16le', ar=16000, ac=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode() if e.stderr else 'unknown'}")

    return output_path


def extract_conversation_audio(conversation_dir: Path) -> list[Path]:
    """Extract audio from all raw MKV files in a conversation directory.

    Outputs WAV files to the processed/ subdirectory, named by user_id
    (Prolific ID) to match the processed MP4 filenames.

    Args:
        conversation_dir: Path to conversation directory (contains raw/ subdir).

    Returns:
        List of paths to extracted audio files.
    """
    raw_dir = conversation_dir / "raw"
    processed_dir = conversation_dir / "processed"

    if not raw_dir.exists():
        logger.warning(f"No raw/ directory in {conversation_dir.name}")
        return []

    mkv_files = list(raw_dir.glob("*.mkv"))
    if not mkv_files:
        logger.warning(f"No MKV files in {raw_dir}")
        return []

    # Load mapping from MKV filename to user_id
    filename_mapping = load_filename_mapping(conversation_dir)

    processed_dir.mkdir(exist_ok=True)
    extracted = []

    for mkv_path in mkv_files:
        # Use user_id for output name (matches processed MP4), fallback to MKV stem
        user_id = filename_mapping.get(mkv_path.name, mkv_path.stem)
        output_path = processed_dir / f"{user_id}.wav"

        if output_path.exists():
            logger.info(f"Skipping {mkv_path.name} (audio already extracted)")
            extracted.append(output_path)
            continue

        logger.info(f"Extracting audio: {mkv_path.name} -> {output_path.name}")
        extract_audio(mkv_path, output_path)
        extracted.append(output_path)

    return extracted


def extract_all_audio(candor_dir: Path) -> int:
    """Extract audio from all conversations in the CANDOR dataset directory.

    Args:
        candor_dir: Root CANDOR dataset directory.

    Returns:
        Number of conversations processed.
    """
    # Find conversation directories (UUID-named directories)
    conversation_dirs = [
        d for d in candor_dir.iterdir()
        if d.is_dir() and not d.name.startswith('raw_media_part')
    ]

    if not conversation_dirs:
        logger.warning("No conversation directories found")
        return 0

    logger.info(f"Found {len(conversation_dirs)} conversations")

    processed = 0
    for conv_dir in sorted(conversation_dirs):
        extracted = extract_conversation_audio(conv_dir)
        if extracted:
            processed += 1

    logger.info(f"Processed {processed} conversations")
    return processed


def cleanup_extras(conversation_dir: Path) -> None:
    """Remove extraneous files to save space.

    Removes combined video/audio and metadata files that aren't needed
    for training. Keeps per-participant MP4s and WAVs.

    Args:
        conversation_dir: Path to conversation directory.
    """
    processed_dir = conversation_dir / "processed"
    conv_id = conversation_dir.name

    # Delete combined video/audio and extras
    extras = [
        processed_dir / f"{conv_id}.mp4",   # Combined video
        processed_dir / f"{conv_id}.mp3",   # Combined audio
        processed_dir / "thumbnail.png",
        processed_dir / "channel_map.json",
    ]

    for path in extras:
        if path.exists():
            path.unlink()
            logger.info(f"Removed {path.name}")
