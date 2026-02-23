"""Type definitions for video cropping operations."""

from __future__ import annotations

from dataclasses import dataclass


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
