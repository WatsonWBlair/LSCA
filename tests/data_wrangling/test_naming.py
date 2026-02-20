"""Tests for filename parsing and naming convention utilities."""

import pytest

from src.data_wrangling.naming import (
    FILENAME_PATTERN,
    build_filename,
    collect_speaker_file_set,
    discover_interaction_pairs,
    parse_filename,
)
from src.data_wrangling.types import FileIdentifier
from pathlib import Path


class TestFilenamePattern:
    def test_valid_stem_matches(self):
        match = FILENAME_PATTERN.match("V03_S1212_I00000389_P1437")
        assert match is not None

    def test_extracts_groups(self):
        match = FILENAME_PATTERN.match("V03_S1212_I00000389_P1437")
        assert match.groups() == ("03", "1212", "00000389", "1437")

    def test_different_lengths(self):
        match = FILENAME_PATTERN.match("V01_S1_I1_P1")
        assert match is not None

    def test_rejects_with_extension(self):
        assert FILENAME_PATTERN.match("V03_S1212_I00000389_P1437.mp4") is None

    def test_rejects_malformed(self):
        assert FILENAME_PATTERN.match("bad_filename") is None

    def test_rejects_missing_prefix(self):
        assert FILENAME_PATTERN.match("S1212_I00000389_P1437") is None


class TestParseFilename:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            parse_filename("V03_S1212_I00000389_P1437")


class TestBuildFilename:
    def test_raises_not_implemented(self):
        ident = FileIdentifier("03", "1212", "00000389", "1437")
        with pytest.raises(NotImplementedError):
            build_filename(ident, ".mp4")


class TestCollectSpeakerFileSet:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            collect_speaker_file_set(Path("/tmp"), "V03_S1212_I00000389_P1437")


class TestDiscoverInteractionPairs:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            discover_interaction_pairs(Path("/tmp"), "improvised", "dev", "0004")
