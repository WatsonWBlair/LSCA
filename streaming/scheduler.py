# encoding/streaming/scheduler.py
# FixedStrideScheduler: fires every stride_sec seconds.
# At each tick: checks RMS silence -> appends zero rows or fires pipelines.

from __future__ import annotations

import logging
import time
import threading
from typing import Callable, Optional

from config import CAMELSConfig

logger = logging.getLogger(__name__)


class FixedStrideScheduler:
    """
    Fires every stride_sec seconds (wall-clock).
    Never skips a chunk_id — silent chunks still consume a row index.
    Row N in every .npy == chunk N in chunks.json.
    """

    def __init__(
        self,
        audio_buffer,
        on_chunk: Callable,
        on_silent: Callable,
        cfg: CAMELSConfig,
    ):
        self._audio_buffer = audio_buffer
        self._on_chunk = on_chunk
        self._on_silent = on_silent
        self._stride_sec = cfg.streaming.stride_sec
        self._window_sec = cfg.streaming.window_sec
        self._rms_threshold = cfg.streaming.rms_silence

        self._chunk_id = 0
        self._session_start: Optional[float] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _scheduler_loop(self):
        self._session_start = time.time()
        next_fire = self._session_start + self._stride_sec

        while not self._stop_event.is_set():
            now = time.time()
            if now < next_fire:
                time.sleep(min(0.005, next_fire - now))
                continue

            end_sec = now - self._session_start
            start_sec = max(0.0, end_sec - self._window_sec)
            rms = self._audio_buffer.rms(window_sec=self._window_sec)

            if rms < self._rms_threshold:
                try:
                    self._on_silent(self._chunk_id, start_sec, end_sec)
                except Exception as e:
                    logger.error("on_silent error at chunk %d: %s", self._chunk_id, e)
            else:
                try:
                    self._on_chunk(self._chunk_id, start_sec, end_sec)
                except Exception as e:
                    logger.error("on_chunk error at chunk %d: %s", self._chunk_id, e)

            self._chunk_id += 1
            next_fire += self._stride_sec

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="FixedStrideScheduler",
        )
        self._thread.start()
        logger.info("Scheduler started: stride=%.1fs, window=%.1fs", self._stride_sec, self._window_sec)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._stride_sec + 1.0)
        logger.info("Scheduler stopped after %d chunks", self._chunk_id)

    @property
    def chunk_id(self) -> int:
        return self._chunk_id
