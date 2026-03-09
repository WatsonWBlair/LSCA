# encoding/streaming/buffers.py
# Thread-safe ring buffers for live audio and video capture.
# AudioBuffer  : deque-backed 16 kHz mono float32 ring buffer
# FrameBuffer  : deque-backed RGB ImageNet-normalized frame + timestamp ring buffer

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision import transforms

from encoding.config import CAMELSConfig

logger = logging.getLogger(__name__)


class AudioBuffer:
    """
    Thread-safe ring buffer for live audio at sample_rate Hz.
    sounddevice pushes via callback; scheduler reads via flush_window().
    """

    def __init__(self, cfg: CAMELSConfig):
        capacity = cfg.streaming.sample_rate * cfg.streaming.audio_buffer_sec
        self._buf = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._sample_rate = cfg.streaming.sample_rate

    def callback(self, indata, frames, time_info, status):
        """sounddevice InputStream callback — appends mono float32 samples."""
        with self._lock:
            self._buf.extend(indata[:, 0].tolist())

    def extend(self, samples: np.ndarray):
        """Manually extend buffer (tests or file replay)."""
        with self._lock:
            self._buf.extend(samples.tolist())

    def flush_window(self, window_sec: float) -> np.ndarray:
        """Return the last window_sec of audio as (T,) float32."""
        n_samples = int(window_sec * self._sample_rate)
        with self._lock:
            buf_list = list(self._buf)
        if len(buf_list) == 0:
            return np.zeros(n_samples, dtype=np.float32)
        arr = np.array(buf_list, dtype=np.float32)
        return arr[-n_samples:] if len(arr) >= n_samples else arr

    def rms(self, window_sec: float) -> float:
        """Compute RMS (std) of the last window — silence detection."""
        audio = self.flush_window(window_sec)
        return float(np.std(audio)) if len(audio) > 0 else 0.0

    def __len__(self):
        with self._lock:
            return len(self._buf)


class FrameBuffer:
    """
    Thread-safe ring buffer for RGB ImageNet-normalized video frames.
    Background thread: camera -> MediaPipe face detect -> crop -> normalize -> buffer.
    Zero tensors stored on face detection miss (preserves temporal alignment).
    """

    def __init__(self, cfg: CAMELSConfig):
        fps = cfg.streaming.target_fps
        capacity = int(cfg.streaming.frame_buffer_sec * fps)
        self._frames: deque = deque(maxlen=capacity)
        self._timestamps: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._fps = fps
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._detector = None
        self._cfg = cfg
        self._normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                list(cfg.streaming.imagenet_mean),
                list(cfg.streaming.imagenet_std),
            ),
        ])

    def _create_detector(self):
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        return mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=self._cfg.streaming.face_detect_conf,
        )

    def _detect_and_crop(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Run MediaPipe face detection. Returns (224, 224, 3) RGB uint8 or None."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(frame_rgb)

        if not results or not results.detections:
            return None

        detection = max(results.detections, key=lambda d: d.score[0])
        h, w = frame_bgr.shape[:2]
        bbox = detection.location_data.relative_bounding_box

        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        x2 = min(w, x1 + bw)
        y2 = min(h, y1 + bh)

        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        size = self._cfg.streaming.marlin_size
        return cv2.resize(crop, (size, size))

    def _push_frame(self, frame_bgr: np.ndarray, ts: float):
        crop = self._detect_and_crop(frame_bgr)
        size = self._cfg.streaming.marlin_size

        if crop is not None:
            tensor = self._normalize(crop)
        else:
            tensor = torch.zeros(3, size, size)

        with self._lock:
            self._frames.append(tensor)
            self._timestamps.append(ts)

    def _capture_loop(self, camera_index: int):
        self._detector = self._create_detector()
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("FrameBuffer: cannot open camera %d", camera_index)
            return

        logger.info("FrameBuffer: camera %d opened", camera_index)
        t0 = time.time()
        frame_n = 0

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            ts = t0 + frame_n / self._fps
            self._push_frame(frame, ts)
            frame_n += 1

        cap.release()

    def start_capture(self, camera_index: int = 0):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, args=(camera_index,), daemon=True, name="FrameCapture",
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def push_frame_from_file(self, frame_bgr: np.ndarray, ts: float):
        if self._detector is None:
            self._detector = self._create_detector()
        self._push_frame(frame_bgr, ts)

    def flush_window(self, window_sec: float) -> tuple[list[torch.Tensor], list[float]]:
        cutoff = int(window_sec * self._fps)
        with self._lock:
            frames = list(self._frames)[-cutoff:]
            timestamps = list(self._timestamps)[-cutoff:]
        return frames, timestamps

    @property
    def fps(self) -> float:
        return self._fps

    def __len__(self):
        with self._lock:
            return len(self._frames)
