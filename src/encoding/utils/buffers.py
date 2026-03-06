# pipeline/buffers.py
# Thread-safe ring buffers for live audio and video capture.
# AudioBuffer  : deque-backed 16 kHz mono float32 ring buffer
# FrameBuffer  : deque-backed RGB ImageNet-normalized frame + timestamp ring buffer
# Both buffers are protected by threading.Lock.

import logging
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from src.encoding.utils.config import (
    SAMPLE_RATE, AUDIO_BUFFER_SEC, TARGET_FPS, FRAME_BUFFER_SEC,
    MARLIN_SIZE, IMAGENET_MEAN, IMAGENET_STD, FACE_DETECT_CONF,
    WINDOW_SEC,
)

logger = logging.getLogger(__name__)

# ImageNet normalization transform (applied once to each face crop)
_normalize = transforms.Compose([
    transforms.ToTensor(),               # HWC uint8 → CHW float32 in [0,1]
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── AudioBuffer ───────────────────────────────────────────────────────────────

class AudioBuffer:
    """
    Thread-safe ring buffer for live audio at SAMPLE_RATE Hz.

    sounddevice pushes chunks via the callback; the scheduler reads
    the last WINDOW_SEC of audio via flush_window().

    Usage:
        buf = AudioBuffer()
        stream = sd.InputStream(..., callback=buf.callback)
        stream.start()
        audio_chunk = buf.flush_window()   # at each scheduler tick
    """

    def __init__(self, maxlen_sec: int = AUDIO_BUFFER_SEC):
        capacity = SAMPLE_RATE * maxlen_sec
        self._buf  = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def callback(self, indata, frames, time_info, status):
        """sounddevice InputStream callback — appends mono float32 samples."""
        with self._lock:
            self._buf.extend(indata[:, 0].tolist())

    def extend(self, samples: np.ndarray):
        """Manually extend buffer (used in tests or file-based replay)."""
        with self._lock:
            self._buf.extend(samples.tolist())

    def flush_window(self, window_sec: float = WINDOW_SEC) -> np.ndarray:
        """
        Return the last window_sec of audio as a (T,) float32 numpy array.
        T = window_sec * SAMPLE_RATE.
        Returns empty array if buffer has fewer samples than requested.
        """
        n_samples = int(window_sec * SAMPLE_RATE)
        with self._lock:
            buf_list = list(self._buf)
        if len(buf_list) == 0:
            return np.zeros(n_samples, dtype=np.float32)
        arr = np.array(buf_list, dtype=np.float32)
        return arr[-n_samples:] if len(arr) >= n_samples else arr

    def rms(self, window_sec: float = WINDOW_SEC) -> float:
        """Compute RMS (std) of the last window — used for silence detection."""
        audio = self.flush_window(window_sec)
        return float(np.std(audio)) if len(audio) > 0 else 0.0

    def __len__(self):
        with self._lock:
            return len(self._buf)


# ── FrameBuffer ───────────────────────────────────────────────────────────────

class FrameBuffer:
    """
    Thread-safe ring buffer for RGB ImageNet-normalized video frames.

    A dedicated thread (face_detect_loop) reads raw BGR frames from the camera,
    runs MediaPipe face detection, crops/normalizes each frame, and pushes
    (tensor, timestamp) pairs into this buffer.

    The scheduler reads the last WINDOW_SEC of frames via flush_window().
    Frames with no face detected are stored as zero tensors to preserve
    temporal alignment — row index invariant must never be violated.

    Usage:
        fb = FrameBuffer()
        fb.start_capture(camera_index=0)   # starts face-detect thread
        frames = fb.flush_window()         # at each scheduler tick
        fb.stop()
    """

    def __init__(
        self,
        maxlen_sec: int = FRAME_BUFFER_SEC,
        fps: float = TARGET_FPS,
    ):
        capacity = int(maxlen_sec * fps)
        self._frames     : deque = deque(maxlen=capacity)   # (3, 224, 224) tensors
        self._timestamps : deque = deque(maxlen=capacity)   # float seconds
        self._lock       = threading.Lock()
        self._fps        = fps
        self._stop_event = threading.Event()
        self._thread     : Optional[threading.Thread] = None
        self._detector   = None  # MediaPipe FaceDetection

    def _create_detector(self):
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        return mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=FACE_DETECT_CONF,
        )

    def _detect_and_crop(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Run MediaPipe face detection on a BGR frame.
        Returns RGB crop of shape (224, 224, 3) uint8, or None if no face found.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results   = self._detector.process(frame_rgb)

        if not results or not results.detections:
            return None

        # Pick highest-confidence detection
        detection = max(results.detections, key=lambda d: d.score[0])
        h, w      = frame_bgr.shape[:2]
        bbox      = detection.location_data.relative_bounding_box

        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        bw = int(bbox.width  * w)
        bh = int(bbox.height * h)
        x2 = min(w, x1 + bw)
        y2 = min(h, y1 + bh)

        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Resize to MARLIN_SIZE × MARLIN_SIZE
        return cv2.resize(crop, (MARLIN_SIZE, MARLIN_SIZE))  # (224, 224, 3) uint8 RGB

    def _push_frame(self, frame_bgr: np.ndarray, ts: float):
        """
        Detect face, normalize, and push (tensor, timestamp) into ring buffer.
        On miss: push zero tensor to preserve temporal alignment.
        """
        crop = self._detect_and_crop(frame_bgr)

        if crop is not None:
            # _normalize expects PIL or HWC uint8 numpy
            tensor = _normalize(crop)          # (3, 224, 224) float32
        else:
            tensor = torch.zeros(3, MARLIN_SIZE, MARLIN_SIZE)

        with self._lock:
            self._frames.append(tensor)
            self._timestamps.append(ts)

    def _capture_loop(self, camera_index: int):
        """Background thread: opens camera and pushes frames continuously."""
        self._detector = self._create_detector()
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("FrameBuffer: cannot open camera %d", camera_index)
            return

        logger.info("FrameBuffer: camera %d opened, FPS=%s", camera_index,
                    cap.get(cv2.CAP_PROP_FPS))
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
        logger.info("FrameBuffer: camera released")

    def start_capture(self, camera_index: int = 0):
        """Start background face-detect + capture thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            args=(camera_index,),
            daemon=True,
            name="FrameCapture",
        )
        self._thread.start()
        logger.info("FrameBuffer: capture thread started (camera %d)", camera_index)

    def stop(self):
        """Stop the capture thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def push_frame_from_file(self, frame_bgr: np.ndarray, ts: float):
        """
        Push a frame from a file-based source (offline / replay mode).
        Use this instead of start_capture() when processing recorded video.
        """
        if self._detector is None:
            self._detector = self._create_detector()
        self._push_frame(frame_bgr, ts)

    def flush_window(
        self,
        window_sec: float = WINDOW_SEC,
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Return frames and timestamps from the last window_sec.
        Returns: (frames, timestamps)
          frames     : list of (3, 224, 224) tensors
          timestamps : list of float seconds
        """
        cutoff_sec = int(window_sec * self._fps)
        with self._lock:
            frames     = list(self._frames)[-cutoff_sec:]
            timestamps = list(self._timestamps)[-cutoff_sec:]
        return frames, timestamps

    @property
    def fps(self) -> float:
        return self._fps

    def __len__(self):
        with self._lock:
            return len(self._frames)
