"""Wrangle CMU-MOSEI segments into datasets/wrangled/ format."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# CMU-MOSEI emotion labels (binary flags from SDK)
EMOTION_LABELS = ("happy", "sad", "angry", "fearful", "disgusted", "surprised")


def discover_segments(raw_root: str | Path) -> list[dict]:
    """Read CMU-MultimodalSDK or HuggingFace manifest and return segment dicts.

    Each dict has keys: segment_id, video_path, label (may be None if labels
    are loaded separately). Skips entries whose video file is missing with a warning.
    """
    raw_root = Path(raw_root)
    segments: list[dict] = []
    skipped = 0

    # HuggingFace layout: raw_root/Videos/{segment_id}.mp4
    # SDK layout: raw_root/Raw/{segment_id}.mp4
    # Try both; whichever contains mp4 files wins.
    candidate_dirs = [
        raw_root / "Videos",
        raw_root / "Raw",
        raw_root,
    ]
    video_dir: Path | None = None
    for d in candidate_dirs:
        if d.is_dir() and any(d.glob("*.mp4")):
            video_dir = d
            break

    if video_dir is None:
        logger.warning("No video directory found under %s — no segments discovered", raw_root)
        return []

    # Check for a labels manifest (CMU-MultimodalSDK saves .pkl; HuggingFace saves .json)
    label_map: dict[str, dict] = {}
    json_manifest = raw_root / "labels.json"
    if json_manifest.exists():
        with open(json_manifest) as f:
            label_map = json.load(f)

    for mp4_path in sorted(video_dir.glob("*.mp4")):
        segment_id = mp4_path.stem
        label = label_map.get(segment_id)
        segments.append({"segment_id": segment_id, "video_path": mp4_path, "label": label})

    # Also walk subdirectories (SDK raw splits organised by train/valid/test)
    for split_dir in sorted(video_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for mp4_path in sorted(split_dir.glob("*.mp4")):
            if not mp4_path.exists():
                logger.warning("Video missing (skipping): %s", mp4_path)
                skipped += 1
                continue
            segment_id = mp4_path.stem
            label = label_map.get(segment_id)
            segments.append({"segment_id": segment_id, "video_path": mp4_path, "label": label})

    if skipped:
        logger.info("Skipped %d missing video file(s)", skipped)
    logger.info("Discovered %d segment(s) under %s", len(segments), raw_root)
    return segments


def extract_labels(sdk_data: dict) -> dict[str, dict]:
    """Convert CMU-MultimodalSDK label dict to per-segment label dicts.

    Expected sdk_data structure (from mmsdk):
        {split: {segment_id: {"label": sentiment_float,
                               "emotions": [happy, sad, angry, fearful, disgusted, surprised]}}}

    Returns:
        {segment_id: {split, sentiment, emotion_flags: dict}}
    """
    result: dict[str, dict] = {}
    for split, split_data in sdk_data.items():
        for seg_id, values in split_data.items():
            sentiment = float(values.get("label", 0.0))
            emotions_raw = values.get("emotions", [0] * 6)
            emotion_flags = {
                name: bool(int(v))
                for name, v in zip(EMOTION_LABELS, emotions_raw)
            }
            result[seg_id] = {
                "split": split,
                "sentiment": sentiment,
                "emotion_flags": emotion_flags,
            }
    return result


def _extract_wav(mp4_path: Path, wav_path: Path) -> bool:
    """Extract 16kHz mono WAV from mp4 using ffmpeg. Returns True on success."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(mp4_path),
        "-ar", "16000", "-ac", "1", "-vn",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning("ffmpeg failed for %s: %s", mp4_path, result.stderr.decode()[:200])
        return False
    return True


def wrangle_cmu_mosei(
    raw_root: str | Path,
    wrangled_root: str | Path,
    max_segments: int | None = None,
) -> None:
    """Wrangle available CMU-MOSEI segments into datasets/wrangled/MS{batch}/.

    For each segment, extracts a 16kHz WAV and writes an (mp4, wav, json) triplet.
    The JSON includes a ``metadata:labels`` key with split, sentiment, and emotion_flags.

    Args:
        raw_root: Path to raw CMU-MOSEI download (contains Videos/ or Raw/).
        wrangled_root: Output root (datasets/wrangled/). Segments go to MS{batch}/.
        max_segments: If set, stop after processing this many segments.
    """
    raw_root = Path(raw_root)
    wrangled_root = Path(wrangled_root)

    segments = discover_segments(raw_root)
    if not segments:
        logger.error("No segments found — nothing to wrangle.")
        return

    if max_segments is not None:
        segments = segments[:max_segments]

    # Determine next free MS batch number
    existing = sorted(wrangled_root.glob("MS*"))
    batch_num = len(existing)
    batch_dir = wrangled_root / f"MS{batch_num:04d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for seg in segments:
        segment_id: str = seg["segment_id"]
        video_path: Path = Path(seg["video_path"])
        label: dict | None = seg.get("label")

        if not video_path.exists():
            logger.warning("Video unavailable (skipping): %s", video_path)
            skipped += 1
            continue

        out_mp4 = batch_dir / f"{segment_id}.mp4"
        out_wav = batch_dir / f"{segment_id}.wav"
        out_json = batch_dir / f"{segment_id}.json"

        # Copy video
        shutil.copy2(str(video_path), str(out_mp4))

        # Extract audio
        if not _extract_wav(out_mp4, out_wav):
            out_mp4.unlink(missing_ok=True)
            skipped += 1
            continue

        # Write metadata JSON
        metadata: dict = {
            "id": segment_id,
            "source": "cmu_mosei",
            "metadata:vad": [],
            "metadata:transcript": [],
        }
        if label is not None:
            metadata["metadata:labels"] = label

        with open(out_json, "w") as f:
            json.dump(metadata, f, indent=2)

        processed += 1

    logger.info(
        "CMU-MOSEI wrangling complete: %d processed, %d skipped → %s",
        processed, skipped, batch_dir,
    )
