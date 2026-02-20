"""Shared dataclasses and type definitions for Seamless Interaction data wrangling.

Naming convention reference:
    V{vendor}_S{session}_I{interaction}_P{participant}.{ext}
    Example: V03_S1212_I00000389_P1437.mp4

Dataset source: https://huggingface.co/datasets/facebook/seamless-interaction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FileIdentifier:
    """Parsed components of a Seamless Interaction filename stem.

    Attributes:
        vendor: Collection site/vendor ID (e.g. "03").
        session_id: Unique session identifier (e.g. "1212").
        interaction_id: Interaction within a session (e.g. "00000389").
        participant_id: Individual participant (e.g. "1437").
    """

    vendor: str
    session_id: str
    interaction_id: str
    participant_id: str

    @property
    def stem(self) -> str:
        """Reconstruct the filename stem (without extension)."""
        return (
            f"V{self.vendor}"
            f"_S{self.session_id}"
            f"_I{self.interaction_id}"
            f"_P{self.participant_id}"
        )

    @property
    def interaction_key(self) -> str:
        """Key for grouping speakers in the same interaction."""
        return f"S{self.session_id}_I{self.interaction_id}"


@dataclass
class SpeakerFileSet:
    """Paths to the four files for a single speaker in an interaction.

    Attributes:
        identifier: Parsed filename components.
        video: Path to the .mp4 file.
        audio: Path to the .wav file.
        metadata_json: Path to the .json file (transcript + VAD).
        features_npz: Path to the .npz file (keypoints, movement, body model).
    """

    identifier: FileIdentifier
    video: Path
    audio: Path
    metadata_json: Path
    features_npz: Path


@dataclass
class InteractionPair:
    """A paired (or unpaired) interaction with one or two speakers.

    Attributes:
        interaction_key: Shared key (e.g. "S1212_I00000389").
        style: "naturalistic" or "improvised".
        split: "train", "dev", "test", or "private".
        batch: Batch identifier string (e.g. "0004").
        speaker_a: First speaker's file set.
        speaker_b: Second speaker's file set, or None if not yet available.
    """

    interaction_key: str
    style: str
    split: str
    batch: str
    speaker_a: SpeakerFileSet
    speaker_b: SpeakerFileSet | None = None


@dataclass
class CropRegion:
    """Pixel coordinates defining a crop region within a video frame.

    Attributes:
        x: Left edge of the crop box in pixels.
        y: Top edge of the crop box in pixels.
        width: Width of the crop box in pixels.
        height: Height of the crop box in pixels.
        output_width: Target width after resize (default 1280).
        output_height: Target height after resize (default 720).
    """

    x: int
    y: int
    width: int
    height: int
    output_width: int = 1280
    output_height: int = 720


@dataclass
class TranscriptSegment:
    """A single utterance segment from the JSON transcript metadata.

    Attributes:
        transcript: The transcribed text.
        start: Start time in seconds.
        end: End time in seconds.
        words: Raw word-level entries with timing and confidence scores.
    """

    transcript: str
    start: float
    end: float
    words: list[dict] = field(default_factory=list)


@dataclass
class VADSegment:
    """A single Voice Activity Detection segment.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
    """

    start: float
    end: float


@dataclass
class ManifestEntry:
    """One entry in the master manifest for a single speaker's wrangled data.

    Attributes:
        interaction_key: Shared interaction key.
        participant_id: This speaker's participant ID.
        style: "naturalistic" or "improvised".
        split: Data split name.
        batch: Batch identifier.
        original_stem: Original filename stem (e.g. "V03_S1212_I00000389_P1437").
        cropped_video_path: Relative path to the cropped video from wrangled root.
        audio_path: Relative path to the audio file.
        transcript_path: Relative path to the transcript JSON.
        features_path: Relative path to the features NPZ.
        partner_participant_id: The other speaker's participant ID, or None.
        duration_seconds: Total duration in seconds, or None if not yet computed.
        num_transcript_segments: Count of transcript utterance segments.
        num_vad_segments: Count of VAD segments.
    """

    interaction_key: str
    participant_id: str
    style: str
    split: str
    batch: str
    original_stem: str
    cropped_video_path: str
    audio_path: str
    transcript_path: str
    features_path: str
    partner_participant_id: str | None = None
    duration_seconds: float | None = None
    num_transcript_segments: int = 0
    num_vad_segments: int = 0
