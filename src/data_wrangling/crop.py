"""Video cropping utilities for reframing footage to webcam-style framing.

The goal is to crop each single-speaker video to mid-chest-and-up framing
centered on the speaker's face, resembling a typical video call camera angle.

The NPZ files already contain bounding boxes and keypoints at 30Hz
(keys: 'boxes_and_keypoints:box', 'boxes_and_keypoints:is_valid_box',
'boxes_and_keypoints:keypoints'). These should be used rather than running
face detection from scratch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.data_wrangling.types import CropRegion


def load_keypoints(
    npz_path: Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Load bounding boxes, validity flags, and keypoints from an NPZ file.

    Expected NPZ keys:
        - 'boxes_and_keypoints:box': bounding boxes per frame
        - 'boxes_and_keypoints:is_valid_box': validity flag per frame
        - 'boxes_and_keypoints:keypoints': facial/body keypoints per frame

    Args:
        npz_path: Path to the speaker's .npz feature file.

    Returns:
        Tuple of (boxes, is_valid, keypoints) numpy arrays.
    """
    raise NotImplementedError


def compute_crop_region(
    boxes: npt.NDArray[np.float64],
    validity: npt.NDArray[np.bool_],
    frame_width: int,
    frame_height: int,
    *,
    aspect_ratio: tuple[int, int] = (16, 9),
    chest_margin_factor: float = 2.5,
) -> CropRegion:
    """Compute a stable crop region from a sequence of bounding boxes.

    Uses valid bounding boxes across sampled frames to determine a
    stabilized crop that includes mid-chest and up while maintaining
    the target aspect ratio.

    Args:
        boxes: Bounding box array from NPZ, shape (N, 4) or similar.
        validity: Boolean validity flags, shape (N,).
        frame_width: Width of the original video frame in pixels.
        frame_height: Height of the original video frame in pixels.
        aspect_ratio: Desired output aspect ratio as (width, height).
        chest_margin_factor: Multiplier applied to face/box height to
            determine how much area below the face to include.

    Returns:
        CropRegion defining the pixel coordinates and output dimensions.
    """
    raise NotImplementedError


def crop_video(
    input_path: Path,
    output_path: Path,
    crop_region: CropRegion,
) -> Path:
    """Apply a crop region to a video and write the cropped output.

    Uses OpenCV (cv2.VideoCapture / cv2.VideoWriter) to read the source
    video, apply the crop, resize to the target output resolution, and
    write the result.

    Args:
        input_path: Path to the source .mp4 file.
        output_path: Path where the cropped .mp4 will be written.
        crop_region: The CropRegion defining the crop and output dimensions.

    Returns:
        Path to the written output file (same as output_path).
    """
    raise NotImplementedError
