# pipeline/scheduler.py
# FixedStrideScheduler: fires every STRIDE_SEC seconds.
# At each tick: checks RMS silence → if silent, appends zero rows.
# Otherwise calls run_all_pipelines().
# Replaces VAD entirely — all chunk boundaries are time-based.

import logging
import time
import threading
from typing import Callable, Optional

from src.encoding.utils.config import (
    STRIDE_SEC, WINDOW_SEC, RMS_SILENCE,
)

logger = logging.getLogger(__name__)


class FixedStrideScheduler:
    """
    Fires every STRIDE_SEC seconds (wall-clock).

    At each tick:
      1. Compute RMS of the last window from the audio buffer.
      2. If silent (rms < RMS_SILENCE):
           - Call on_silent(chunk_id, start_sec, end_sec)
           - Increment chunk_id
      3. If not silent:
           - Call on_chunk(chunk_id, start_sec, end_sec)
           - Increment chunk_id
      4. Advance next_fire_time by STRIDE_SEC.

    Never skips a chunk_id — even silent chunks consume a row index.
    Row N in every .npy == chunk N in chunks.json.

    Usage:
        scheduler = FixedStrideScheduler(audio_buffer, on_chunk=..., on_silent=...)
        scheduler.start()
        ...
        scheduler.stop()
    """

    def __init__(
        self,
        audio_buffer,                         # AudioBuffer instance
        on_chunk:  Callable,                  # on_chunk(chunk_id, start_sec, end_sec)
        on_silent: Callable,                  # on_silent(chunk_id, start_sec, end_sec)
        stride_sec: float = STRIDE_SEC,
        window_sec: float = WINDOW_SEC,
        rms_threshold: float = RMS_SILENCE,
    ):
        self._audio_buffer  = audio_buffer
        self._on_chunk      = on_chunk
        self._on_silent     = on_silent
        self._stride_sec    = stride_sec
        self._window_sec    = window_sec
        self._rms_threshold = rms_threshold

        self._chunk_id      = 0
        self._session_start : Optional[float] = None
        self._stop_event    = threading.Event()
        self._thread        : Optional[threading.Thread] = None

    def _scheduler_loop(self):
        self._session_start = time.time()
        next_fire           = self._session_start + self._stride_sec

        while not self._stop_event.is_set():
            now = time.time()
            if now < next_fire:
                time.sleep(min(0.005, next_fire - now))
                continue

            # Wall-clock window: [now - window_sec, now]
            end_sec   = now - self._session_start
            start_sec = max(0.0, end_sec - self._window_sec)

            rms = self._audio_buffer.rms(window_sec=self._window_sec)

            if rms < self._rms_threshold:
                logger.debug(
                    "Scheduler tick %d: SILENT (rms=%.4f < %.4f)",
                    self._chunk_id, rms, self._rms_threshold
                )
                try:
                    self._on_silent(self._chunk_id, start_sec, end_sec)
                except Exception as e:
                    logger.error("on_silent error at chunk %d: %s", self._chunk_id, e)
            else:
                logger.debug(
                    "Scheduler tick %d: processing (rms=%.4f, t=[%.2f, %.2f])",
                    self._chunk_id, rms, start_sec, end_sec
                )
                try:
                    self._on_chunk(self._chunk_id, start_sec, end_sec)
                except Exception as e:
                    logger.error("on_chunk error at chunk %d: %s", self._chunk_id, e)

            self._chunk_id += 1
            next_fire      += self._stride_sec

    def start(self):
        """Start the scheduler in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="FixedStrideScheduler",
        )
        self._thread.start()
        logger.info(
            "FixedStrideScheduler started: stride=%.1fs, window=%.1fs",
            self._stride_sec, self._window_sec,
        )

    def stop(self):
        """Signal scheduler to stop and join thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._stride_sec + 1.0)
        logger.info(
            "FixedStrideScheduler stopped after %d chunks", self._chunk_id
        )

    @property
    def chunk_id(self) -> int:
        return self._chunk_id
