"""Video cropping utilities for reframing footage to webcam-style framing.

The goal is to crop each single-speaker video to mid-chest-and-up framing
centered on the speaker's face, resembling a typical video call camera angle.

The NPZ files already contain bounding boxes and keypoints at 30Hz
(keys: 'boxes_and_keypoints:box', 'boxes_and_keypoints:is_valid_box',
'boxes_and_keypoints:keypoints'). These should be used rather than running
face detection from scratch.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import ffmpeg
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

from src.data_wrangling.types import CropRegion


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """Get video storage dimensions using ffprobe.

    Returns raw storage dimensions. FFmpeg will handle SAR when outputting.

    Args:
        video_path: Path to the video file.

    Returns:
        Tuple of (width, height) in storage pixels.
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    stream = data['streams'][0]
    return stream['width'], stream['height']


def load_keypoints(
    npz_path: Path,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Load keypoints and validity flags from an NPZ file.

    Expected NPZ keys:
        - 'boxes_and_keypoints:is_valid_box': validity flag per frame, shape (N,)
        - 'boxes_and_keypoints:keypoints': 133 keypoints per frame, shape (N, 133, 3)
          - Keypoints 0-16: body (COCO format)
          - Keypoints 23-90: face landmarks (68 points)

    Args:
        npz_path: Path to the speaker's .npz feature file.

    Returns:
        Tuple of (keypoints, is_valid) numpy arrays.
    """
    with np.load(npz_path) as data:
        is_valid = data['boxes_and_keypoints:is_valid_box']
        keypoints = data['boxes_and_keypoints:keypoints']
    return keypoints, is_valid


# Face keypoint indices in COCO WholeBody format
FACE_KEYPOINT_START = 23
FACE_KEYPOINT_END = 91  # exclusive


def compute_face_bbox(
    keypoints: npt.NDArray[np.float32],
    validity: npt.NDArray[np.bool_],
    confidence_threshold: float = 0.3,
) -> tuple[float, float, float, float]:
    """Compute a stable face bounding box from face keypoints.

    Uses face keypoints (indices 23-90) to compute a bounding box around
    the face region. Takes the median across valid frames for stability.

    Args:
        keypoints: Keypoints array, shape (N, 133, 3) as (x, y, confidence).
        validity: Boolean validity flags, shape (N,).
        confidence_threshold: Minimum confidence to include a keypoint.

    Returns:
        Tuple of (x, y, width, height) for the face bounding box.
    """
    valid_keypoints = keypoints[validity]
    if len(valid_keypoints) == 0:
        raise ValueError("No valid keypoint frames found")

    # Extract face keypoints (indices 23-90)
    face_kps = valid_keypoints[:, FACE_KEYPOINT_START:FACE_KEYPOINT_END, :]

    # Filter by confidence and compute per-frame face bounds
    face_boxes = []
    for frame_kps in face_kps:
        confident = frame_kps[:, 2] >= confidence_threshold
        if confident.sum() < 5:  # Need at least 5 points for reliable box
            continue
        x_coords = frame_kps[confident, 0]
        y_coords = frame_kps[confident, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        face_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])

    if len(face_boxes) == 0:
        raise ValueError("No frames with sufficient confident face keypoints")

    # Take median for stability
    median_box = np.median(face_boxes, axis=0)
    return tuple(median_box)


def compute_crop_region(
    keypoints: npt.NDArray[np.float32],
    validity: npt.NDArray[np.bool_],
    frame_width: int,
    frame_height: int,
    *,
    margin_factor: float = 4,
) -> CropRegion:
    """Compute a square crop region centered on the face.

    Creates a square crop in storage space. FFmpeg will apply SAR
    when outputting to give correct display proportions.

    Args:
        keypoints: Keypoints array, shape (N, 133, 3).
        validity: Boolean validity flags, shape (N,).
        frame_width: Width of the video frame in storage pixels.
        frame_height: Height of the video frame in storage pixels.
        margin_factor: Multiplier for expanding face bbox to crop size.

    Returns:
        CropRegion defining a square crop in storage pixels.
    """
    # Get face bounding box from keypoints
    face_x, face_y, face_w, face_h = compute_face_bbox(keypoints, validity)

    # Face center
    face_center_x = face_x + face_w / 2
    face_center_y = face_y + face_h / 2

    # Square crop size based on face height with margin
    crop_size = face_h * margin_factor

    # Center square on face
    crop_x = face_center_x - crop_size / 2
    crop_y = face_center_y - crop_size / 2

    # Clamp to frame boundaries
    crop_x = max(0, min(crop_x, frame_width - crop_size))
    crop_y = max(0, min(crop_y, frame_height - crop_size))

    # Ensure crop doesn't exceed frame
    if crop_size > frame_width:
        crop_size = frame_width
    if crop_size > frame_height:
        crop_size = frame_height

    return CropRegion(
        x=int(crop_x),
        y=int(crop_y),
        width=int(crop_size),
        height=int(crop_size),
    )


def crop_preview(
    input_path: Path,
    output_path: Path,
    crop_region: CropRegion,
    frame_index: int = 0,
) -> Path:
    """Extract a single cropped frame as a preview image using FFmpeg.

    Uses FFmpeg for frame extraction to properly handle sample aspect ratio.

    Args:
        input_path: Path to the source .mp4 file.
        output_path: Path where the preview image will be written (.jpg or .png).
        crop_region: The CropRegion defining the crop.
        frame_index: Which frame to extract (default: first frame).

    Returns:
        Path to the written preview image.
    """
    # Use FFmpeg to extract and crop in one step (handles SAR correctly)
    try:
        (
            ffmpeg
            .input(str(input_path))
            .filter('select', f'eq(n,{frame_index})')
            .filter('setsar', '1')
            .filter('crop',
                    w=crop_region.width,
                    h=crop_region.height,
                    x=crop_region.x,
                    y=crop_region.y)
            .filter('scale', crop_region.width, crop_region.height)
            .output(str(output_path), vframes=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode() if e.stderr else 'unknown'}")

    return output_path


def crop_video(
    input_path: Path,
    output_path: Path,
    crop_region: CropRegion,
) -> Path:
    """Crop an entire video to the specified region.

    Args:
        input_path: Path to the source .mp4 file.
        output_path: Path where the cropped video will be written.
        crop_region: The CropRegion defining the crop.

    Returns:
        Path to the cropped video.
    """
    try:
        (
            ffmpeg
            .input(str(input_path))
            .filter('setsar', '1')
            .filter('crop',
                    w=crop_region.width,
                    h=crop_region.height,
                    x=crop_region.x,
                    y=crop_region.y)
            .output(str(output_path), vcodec='libx264', crf=23)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode() if e.stderr else 'unknown'}")

    return output_path


def process_interaction(
    npz_path: Path,
    output_dir: Path,
    short_name: str,
) -> Path:
    """Crop a video based on face keypoints.

    Args:
        npz_path: Path to the .npz keypoints file.
        output_dir: Directory where output video will be written.
        short_name: Base name for output file (e.g., "I00001227_P1618").

    Returns:
        Path to the cropped video file.
    """
    video_path = npz_path.with_suffix('.mp4')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load keypoints and compute crop region
    keypoints, validity = load_keypoints(npz_path)
    frame_width, frame_height = get_video_dimensions(video_path)
    crop_region = compute_crop_region(keypoints, validity, frame_width, frame_height)
    logger.info(f"Crop: x={crop_region.x}, y={crop_region.y}, w={crop_region.width}, h={crop_region.height}")

    # Crop video
    video_out = output_dir / f"{short_name}.mp4"
    crop_video(video_path, video_out, crop_region)

    return video_out
