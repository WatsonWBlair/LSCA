"""Tests for data wrangling shared dataclasses."""

import pytest

from src.data_wrangling.types import (
    CropRegion,
    FileIdentifier,
    InteractionPair,
    SpeakerFileSet,
    TranscriptSegment,
    VADSegment,
)
from pathlib import Path


class TestFileIdentifier:
    def test_stem_reconstruction(self):
        ident = FileIdentifier(
            vendor="03",
            session_id="1212",
            interaction_id="00000389",
            participant_id="1437",
        )
        assert ident.stem == "V03_S1212_I00000389_P1437"

    def test_interaction_key(self):
        ident = FileIdentifier(
            vendor="03",
            session_id="1212",
            interaction_id="00000389",
            participant_id="1437",
        )
        assert ident.interaction_key == "S1212_I00000389"

    def test_frozen(self):
        ident = FileIdentifier(
            vendor="03",
            session_id="1212",
            interaction_id="00000389",
            participant_id="1437",
        )
        with pytest.raises(AttributeError):
            ident.vendor = "04"

    def test_different_participants_same_interaction_key(self):
        a = FileIdentifier("03", "1212", "00000389", "1437")
        b = FileIdentifier("03", "1212", "00000389", "4045")
        assert a.interaction_key == b.interaction_key
        assert a.stem != b.stem


class TestSpeakerFileSet:
    def test_construction(self):
        ident = FileIdentifier("03", "1212", "00000389", "1437")
        sfs = SpeakerFileSet(
            identifier=ident,
            video=Path("/data/test.mp4"),
            audio=Path("/data/test.wav"),
            metadata_json=Path("/data/test.json"),
            features_npz=Path("/data/test.npz"),
        )
        assert sfs.identifier.participant_id == "1437"
        assert sfs.video.suffix == ".mp4"


class TestInteractionPair:
    def test_unpaired_speaker(self):
        ident = FileIdentifier("03", "1212", "00000389", "1437")
        sfs = SpeakerFileSet(
            identifier=ident,
            video=Path("/data/test.mp4"),
            audio=Path("/data/test.wav"),
            metadata_json=Path("/data/test.json"),
            features_npz=Path("/data/test.npz"),
        )
        pair = InteractionPair(
            interaction_key="S1212_I00000389",
            style="improvised",
            split="dev",
            batch="0004",
            speaker_a=sfs,
        )
        assert pair.speaker_b is None


class TestCropRegion:
    def test_defaults(self):
        region = CropRegion(x=0, y=0, width=640, height=480)
        assert region.output_width == 1280
        assert region.output_height == 720


class TestTranscriptSegment:
    def test_construction(self):
        seg = TranscriptSegment(
            transcript="Hello world.",
            start=1.0,
            end=2.5,
        )
        assert seg.words == []


class TestVADSegment:
    def test_construction(self):
        seg = VADSegment(start=8.466, end=9.71)
        assert seg.end > seg.start
