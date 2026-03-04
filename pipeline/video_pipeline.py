# pipeline/video_pipeline.py
# Video pipeline: sliding window sampling → MARLIN encoding → TemporalAttentionPool
# Input:  list of (3, 224, 224) RGB ImageNet-normalized tensors (from FrameBuffer)
# Output: (768,) pooled MARLIN embedding for one chunk

import logging
import torch
import numpy as np
from typing import List

import torchvision.transforms.functional as TF

from pipeline.config import (
    MARLIN_FRAMES, WINDOW_SEC, STRIDE_SEC,
    IMAGENET_MEAN, IMAGENET_STD, MARLIN_SIZE,
)

logger = logging.getLogger(__name__)


# ── Uniform frame sampling helper ────────────────────────────────────────────

def uniform_sample(frames: list, n: int) -> list:
    """
    Sample exactly n frames from a list by uniform spacing.
    If len(frames) < n, repeats the last frame to pad.
    """
    if len(frames) == 0:
        return [torch.zeros(3, MARLIN_SIZE, MARLIN_SIZE)] * n
    if len(frames) == n:
        return frames
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


# ── Sliding window sampling ───────────────────────────────────────────────────

def sliding_window_sample(
    chunk_frames: list,
    fps: float,
    window_sec: float = WINDOW_SEC,
    stride_sec: float = STRIDE_SEC,
    n: int = MARLIN_FRAMES,
) -> list:
    """
    Partition chunk_frames into sliding windows of window_sec, stepping by stride_sec.
    Each window is uniformly sampled down to exactly n frames.

    Returns: list of windows, each window is a list of n (3, 224, 224) tensors.
    If the chunk is shorter than one window, returns a single window.
    """
    total   = len(chunk_frames)
    win_len = max(1, int(window_sec * fps))
    stride  = max(1, int(stride_sec * fps))

    if total <= win_len:
        return [uniform_sample(chunk_frames, n)]

    windows = []
    start   = 0
    while start + win_len <= total:
        windows.append(uniform_sample(chunk_frames[start:start + win_len], n))
        start += stride

    # Ensure at least one window even if loop didn't fire
    if not windows:
        windows.append(uniform_sample(chunk_frames, n))

    return windows


# ── MARLIN encoding ───────────────────────────────────────────────────────────

def encode_windows_marlin(windows: list, marlin_model) -> torch.Tensor:
    """
    Encode a list of windows through MARLIN (frozen).
    Each window is a list of 16 (3, 224, 224) tensors.

    Returns: (W, 768) — one embedding per window.
    """
    embeddings = []
    with torch.no_grad():
        for win in windows:
            # Stack frames: list of (3,224,224) → (16,3,224,224) → permute → (1,3,16,224,224)
            x = torch.stack(win)                     # (16, 3, 224, 224)
            x = x.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, 16, 224, 224)

            device = next(marlin_model.parameters()).device
            x = x.to(device)

            feat = marlin_model.extract_features(x, keep_seq=False)  # (1, 768)
            embeddings.append(feat.squeeze(0).cpu())

    return torch.stack(embeddings)  # (W, 768)


# ── Full video pipeline ───────────────────────────────────────────────────────

def video_pipeline(
    chunk_frames: list,
    fps: float,
    marlin_model,
    temporal_pool,
) -> torch.Tensor:
    """
    Full video pipeline for one chunk.

    Args:
        chunk_frames  : list of (3, 224, 224) RGB ImageNet-normalized tensors
        fps           : frames per second of the source (used for window sizing)
        marlin_model  : frozen MARLIN model (from model_loader)
        temporal_pool : trainable TemporalAttentionPool

    Returns: (768,) video embedding for this chunk
    """
    if len(chunk_frames) == 0:
        logger.warning("video_pipeline: empty chunk_frames — returning zeros")
        return torch.zeros(768)

    windows = sliding_window_sample(chunk_frames, fps)   # list of windows
    H       = encode_windows_marlin(windows, marlin_model)  # (W, 768)
    return temporal_pool(H)                              # (768,)


# ── Offline / file-based extraction (for training data preprocessing) ─────────

def extract_video_file(
    video_path: str,
    marlin_model,
    temporal_pool,
    fps: float = 30.0,
) -> torch.Tensor:
    """
    Extract a single (768,) embedding from a pre-saved video file.

    Reads frames with OpenCV, resizes to 224×224, applies ImageNet normalization,
    then uses the same sliding-window + MARLIN extract_features() path as the
    live-streaming pipeline.  This avoids MARLIN's file-level face-detector
    (FaceXZoo) which requires a separate model download.

    Returns: (768,) tensor, or zeros if extraction fails.
    """
    import cv2

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("extract_video_file: cannot open %s", video_path)
            return torch.zeros(768)

        video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        frames: List[torch.Tensor] = []

        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb   = cv2.resize(rgb, (MARLIN_SIZE, MARLIN_SIZE), interpolation=cv2.INTER_LINEAR)
            t     = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0   # (3,224,224)
            t     = TF.normalize(t, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
            frames.append(t)

        cap.release()

        if not frames:
            logger.warning("extract_video_file: no frames decoded from %s", video_path)
            return torch.zeros(768)

        # Re-use the same sliding-window + MARLIN path as the live pipeline
        windows = sliding_window_sample(frames, video_fps)    # list of windows
        H       = encode_windows_marlin(windows, marlin_model) # (W, 768)
        return temporal_pool(H)                                # (768,)

    except Exception as e:
        logger.error("extract_video_file failed for %s: %s", video_path, e)
        return torch.zeros(768)


# ── Preprocessing sanity checks ──────────────────────────────────────────────

def check_frame_tensor(tensor: torch.Tensor, frame_idx: int = 0):
    """
    Assert that a frame tensor is correctly formatted for MARLIN:
    - shape: (3, 224, 224)
    - dtype: float32
    - mean near 0 (ImageNet normalization applied)
    Logs a warning if anything is off; does NOT raise.
    """
    if tensor.shape != torch.Size([3, MARLIN_SIZE, MARLIN_SIZE]):
        logger.warning("Frame %d has wrong shape: %s (expected [3,224,224])", frame_idx, tensor.shape)
    if tensor.dtype != torch.float32:
        logger.warning("Frame %d dtype %s — expected float32", frame_idx, tensor.dtype)
    mean_val = tensor.mean().item()
    if abs(mean_val) > 1.0:
        logger.warning(
            "Frame %d mean=%.3f — ImageNet normalization may not have been applied "
            "(expected mean near 0.0)", frame_idx, mean_val
        )
