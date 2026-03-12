# encoding/pipelines/video.py
# Video pipeline: sliding window sampling -> MARLIN encoding -> TemporalAttentionPool
# Input:  list of (3, 224, 224) RGB ImageNet-normalized tensors (from FrameBuffer)
# Output: (d_video,) pooled MARLIN embedding for one chunk
# Ref: MARLIN, Cai et al. CVPR 2023

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
import torchvision.transforms.functional as TF

from config import CAMELSConfig

logger = logging.getLogger(__name__)


def uniform_sample(frames: list[torch.Tensor], n: int, frame_size: int) -> list[torch.Tensor]:
    """Sample exactly n frames by uniform spacing. Pads with zeros if empty."""
    if len(frames) == 0:
        return [torch.zeros(3, frame_size, frame_size)] * n
    if len(frames) == n:
        return frames
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


def sliding_window_sample(
    chunk_frames: list[torch.Tensor],
    fps: float,
    cfg: CAMELSConfig,
) -> list[list[torch.Tensor]]:
    """
    Partition chunk_frames into sliding windows, each uniformly sampled
    to exactly marlin_frames frames.
    """
    n = cfg.streaming.marlin_frames
    frame_size = cfg.streaming.marlin_size
    win_len = max(1, int(cfg.streaming.window_sec * fps))
    stride = max(1, int(cfg.streaming.stride_sec * fps))
    total = len(chunk_frames)

    if total <= win_len:
        return [uniform_sample(chunk_frames, n, frame_size)]

    windows = []
    start = 0
    while start + win_len <= total:
        windows.append(uniform_sample(chunk_frames[start:start + win_len], n, frame_size))
        start += stride

    if not windows:
        windows.append(uniform_sample(chunk_frames, n, frame_size))

    return windows


def encode_windows_marlin(
    windows: list[list[torch.Tensor]],
    marlin_model,
) -> torch.Tensor:
    """
    Encode windows through MARLIN (frozen).
    Each window: list of marlin_frames (3, 224, 224) tensors.
    Returns: (W, d_video) — one embedding per window.
    """
    embeddings = []
    with torch.no_grad():
        for win in windows:
            x = torch.stack(win)                      # (16, 3, 224, 224)
            x = x.permute(1, 0, 2, 3).unsqueeze(0)   # (1, 3, 16, 224, 224)
            device = next(marlin_model.parameters()).device
            x = x.to(device)
            feat = marlin_model.extract_features(x, keep_seq=False)  # (1, d_video)
            embeddings.append(feat.squeeze(0).cpu())

    return torch.stack(embeddings)  # (W, d_video)


def video_pipeline(
    chunk_frames: list[torch.Tensor],
    fps: float,
    marlin_model,
    temporal_pool,
    cfg: CAMELSConfig,
) -> torch.Tensor:
    """
    Full video pipeline for one chunk.
    Returns: (d_video,) video embedding.
    """
    if len(chunk_frames) == 0:
        logger.warning("video_pipeline: empty chunk_frames — returning zeros")
        return torch.zeros(cfg.latent.d_video)

    windows = sliding_window_sample(chunk_frames, fps, cfg)
    H = encode_windows_marlin(windows, marlin_model)  # (W, d_video)
    return temporal_pool(H)                            # (d_video,)


def extract_video_file(
    video_path: str,
    marlin_model,
    temporal_pool,
    cfg: CAMELSConfig,
    fps: float = 30.0,
) -> torch.Tensor:
    """
    Extract a single (d_video,) embedding from a video file.
    Reads frames with OpenCV, resizes, applies ImageNet normalization.
    """
    import cv2

    marlin_size = cfg.streaming.marlin_size
    im_mean = list(cfg.streaming.imagenet_mean)
    im_std = list(cfg.streaming.imagenet_std)

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("extract_video_file: cannot open %s", video_path)
            return torch.zeros(cfg.latent.d_video)

        video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        frames: List[torch.Tensor] = []

        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (marlin_size, marlin_size), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t = TF.normalize(t, mean=im_mean, std=im_std)
            frames.append(t)

        cap.release()

        if not frames:
            logger.warning("extract_video_file: no frames from %s", video_path)
            return torch.zeros(cfg.latent.d_video)

        windows = sliding_window_sample(frames, video_fps, cfg)
        H = encode_windows_marlin(windows, marlin_model)
        return temporal_pool(H)

    except Exception as e:
        logger.error("extract_video_file failed for %s: %s", video_path, e)
        return torch.zeros(cfg.latent.d_video)


def check_frame_tensor(tensor: torch.Tensor, cfg: CAMELSConfig, frame_idx: int = 0):
    """Assert frame tensor is correctly formatted for MARLIN."""
    s = cfg.streaming.marlin_size
    if tensor.shape != torch.Size([3, s, s]):
        logger.warning("Frame %d wrong shape: %s (expected [3,%d,%d])", frame_idx, tensor.shape, s, s)
    if tensor.dtype != torch.float32:
        logger.warning("Frame %d dtype %s — expected float32", frame_idx, tensor.dtype)
    mean_val = tensor.mean().item()
    if abs(mean_val) > 1.0:
        logger.warning("Frame %d mean=%.3f — ImageNet normalization may be missing", frame_idx, mean_val)
