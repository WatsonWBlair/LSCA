"""Tests for data wrangling type definitions."""

from src.data_wrangling.types import CropRegion


class TestCropRegion:
    def test_construction(self):
        region = CropRegion(x=100, y=200, width=640, height=480)
        assert region.x == 100
        assert region.y == 200
        assert region.width == 640
        assert region.height == 480

    def test_defaults(self):
        region = CropRegion(x=0, y=0, width=640, height=480)
        assert region.output_width == 1280
        assert region.output_height == 720

    def test_custom_output_size(self):
        region = CropRegion(
            x=0, y=0, width=640, height=480,
            output_width=1920, output_height=1080
        )
        assert region.output_width == 1920
        assert region.output_height == 1080
