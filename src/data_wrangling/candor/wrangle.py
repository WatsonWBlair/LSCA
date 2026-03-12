"""Wrangle CANDOR conversations into datasets/wrangled/ format."""

from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

GAP_THRESHOLD = 1.5  # seconds between words to split utterances


def _timedelta_to_seconds(td_str: str) -> float:
    """Convert 'HH:MM:SS' string to float seconds."""
    parts = td_str.strip().split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def extract_vad(conv_dir: Path, user_id: str) -> list[dict]:
    """Read audio_video_features.csv and merge is_speaking runs into {start, end} segments."""
    csv_path = conv_dir / "audio_video_features.csv"
    segments = []
    current_start = None
    current_end = None

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["user_id"] != user_id:
                continue
            t = _timedelta_to_seconds(row["timedelta"])
            speaking = row["is_speaking"].strip() == "True"
            if speaking:
                if current_start is None:
                    current_start = t
                current_end = t + 1.0
            else:
                if current_start is not None:
                    segments.append({"start": current_start, "end": current_end})
                    current_start = None
                    current_end = None

    if current_start is not None:
        segments.append({"start": current_start, "end": current_end})

    return segments


def extract_transcript(conv_dir: Path, user_id: str, metadata: dict) -> list[dict]:
    """Read transcribe_output.json and group words into utterances (1.5 s gap)."""
    channel_label = None
    for speaker in metadata.get("speakers", []):
        if speaker.get("user_id") == user_id:
            ch = speaker.get("channel", "L")
            channel_label = "ch_0" if ch == "L" else "ch_1"
            break

    if channel_label is None:
        logger.warning("No channel found for user %s", user_id)
        return []

    transcript_path = conv_dir / "transcription" / "transcribe_output.json"
    if not transcript_path.exists():
        logger.warning("No transcribe_output.json in %s", conv_dir.name)
        return []

    with open(transcript_path) as f:
        data = json.load(f)

    channels = data.get("results", {}).get("channel_labels", {}).get("channels", [])
    items = []
    for ch in channels:
        if ch.get("channel_label") == channel_label:
            items = ch.get("items", [])
            break

    words = []
    for item in items:
        if item.get("type") != "pronunciation":
            continue
        alts = item.get("alternatives", [{}])
        words.append({
            "word": alts[0].get("content", ""),
            "start": float(item["start_time"]),
            "end": float(item["end_time"]),
            "score": float(alts[0].get("confidence", 0.0)),
        })

    if not words:
        return []

    utterances = []
    group = [words[0]]
    for w in words[1:]:
        if w["start"] - group[-1]["end"] > GAP_THRESHOLD:
            utterances.append(group)
            group = [w]
        else:
            group.append(w)
    utterances.append(group)

    result = []
    for group in utterances:
        result.append({
            "words": group,
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "transcript": " ".join(w["word"] for w in group),
        })

    return result


def extract_survey(conv_dir: Path, user_id: str) -> dict:
    """Read survey.csv and return the row matching user_id as a dict."""
    survey_path = conv_dir / "survey.csv"
    if not survey_path.exists():
        return {}

    with open(survey_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("user_id") == user_id:
                return dict(row)

    return {}


def extract_avf(conv_dir: Path, user_id: str) -> list[dict]:
    """Read audio_video_features.csv and return per-second rows for this user."""
    csv_path = conv_dir / "audio_video_features.csv"
    rows = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["user_id"] != user_id:
                continue
            entry = {"time_sec": _timedelta_to_seconds(row["timedelta"])}
            for k, v in row.items():
                if k not in ("timedelta", "user_id"):
                    entry[k] = v
            rows.append(entry)

    return rows


def wrangle_conversation(conv_dir: Path, part_num: str, wrangled_root: Path) -> int:
    """Wrangle a single conversation into wrangled_root/C{part_num}/. Returns user count."""
    metadata_path = conv_dir / "metadata.json"
    if not metadata_path.exists():
        logger.warning("No metadata.json in %s — skipping", conv_dir.name)
        return 0

    with open(metadata_path) as f:
        metadata = json.load(f)

    conv_uuid = conv_dir.name
    out_dir = Path(wrangled_root) / f"C{part_num}"
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = conv_dir / "processed"
    raw_dir = conv_dir / "raw"
    if raw_dir.exists():
        try:
            from src.data_wrangling.candor.extract import extract_conversation_audio
            extract_conversation_audio(conv_dir)
        except ImportError:
            logger.warning("ffmpeg not available — skipping wav extraction for %s", conv_dir.name)

    written = 0
    for speaker in metadata.get("speakers", []):
        user_id = speaker.get("user_id")
        if not user_id:
            continue

        mp4_src = processed_dir / f"{user_id}.mp4"
        wav_src = processed_dir / f"{user_id}.wav"

        if not mp4_src.exists():
            logger.warning("No mp4 for user %s in %s — skipping", user_id, conv_dir.name)
            continue

        stem = f"{conv_uuid}_{user_id}"
        mp4_dst = out_dir / f"{stem}.mp4"
        wav_dst = out_dir / f"{stem}.wav"
        json_dst = out_dir / f"{stem}.json"

        if json_dst.exists():
            logger.info("Skipping %s (already wrangled)", stem)
            written += 1
            continue

        shutil.copy2(mp4_src, mp4_dst)

        if wav_src.exists():
            shutil.copy2(wav_src, wav_dst)
        else:
            logger.warning("No wav for user %s in %s", user_id, conv_dir.name)

        record = {
            "id": f"CANDOR_{conv_uuid}_{user_id}",
            "metadata:vad": extract_vad(conv_dir, user_id),
            "metadata:transcript": extract_transcript(conv_dir, user_id, metadata),
            "metadata:survey": extract_survey(conv_dir, user_id),
            "metadata:audio_video_features": extract_avf(conv_dir, user_id),
        }

        with open(json_dst, "w") as f:
            json.dump(record, f, indent=2)

        logger.info("Wrangled %s", stem)
        written += 1

    return written


def wrangle_all_candor(
    candor_dir: Path,
    wrangled_root: Path,
    part_nums: list[str] | None = None,
) -> int:
    """Wrangle all already-processed CANDOR parts into wrangled_root."""
    candor_dir = Path(candor_dir)
    wrangled_root = Path(wrangled_root)
    total = 0

    if part_nums is not None:
        part_dirs = [candor_dir / p for p in part_nums]
    else:
        part_dirs = sorted(d for d in candor_dir.iterdir() if d.is_dir())

    for part_dir in part_dirs:
        if not part_dir.is_dir():
            continue
        part_num = part_dir.name  # e.g., '004'

        for conv_dir in sorted(part_dir.iterdir()):
            if not conv_dir.is_dir():
                continue
            processed_dir = conv_dir / "processed"
            if not processed_dir.exists():
                continue
            # Only process dirs with per-user mp4s (skip combined conv mp4)
            user_mp4s = [
                p for p in processed_dir.glob("*.mp4")
                if p.stem != conv_dir.name
            ]
            if not user_mp4s:
                continue

            count = wrangle_conversation(conv_dir, part_num, wrangled_root)
            total += count

    logger.info("Total users wrangled: %d", total)
    return total
