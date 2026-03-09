# encoding/pipelines/transcript.py
# Transcript utility: Emformer RNN-T streaming ASR + transcript.json saving.
# v8.1: Text is NOT a latent modality. Emformer runs continuously as a parallel
# utility, saving token deltas to transcript.json for the HyperGNN teammate.
# Ref: Zhang et al. Interspeech 2021

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque

import torch
import numpy as np

from encoding.config import CAMELSConfig

logger = logging.getLogger(__name__)


class EmformerASR:
    """
    Background-thread streaming ASR using Emformer RNN-T.
    asr_state and asr_hypothesis are NEVER reset between chunks
    (only after reset_silence_sec of silence).

    Writes token deltas to transcript.json at each scheduler tick.
    """

    def __init__(self, models: dict, output_dir: str, cfg: CAMELSConfig):
        self.model = models["emformer_model"]
        self.decoder = models["emformer_decoder"]
        self.token_processor = models["token_processor"]
        self.feat_extractor = models["feat_extractor"]
        self.cfg = cfg
        self.output_dir = output_dir

        self._running = False
        self._thread: threading.Thread | None = None
        self._audio_buf: deque | None = None

        # ASR state (persistent across chunks)
        self.asr_state = None
        self.asr_hypothesis = None
        self._prev_token_count = 0
        self._last_speech_time = time.time()

        # Per-chunk transcript deltas
        self._transcript_entries: list[dict] = []
        self._lock = threading.Lock()

    def start(self, audio_deque: deque):
        """Start background ASR thread."""
        self._audio_buf = audio_deque
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("EmformerASR started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("EmformerASR stopped")

    def _run_loop(self):
        seg_total = self.cfg.streaming.seg_total
        sr = self.cfg.streaming.sample_rate

        while self._running:
            if self._audio_buf is None or len(self._audio_buf) < seg_total:
                time.sleep(0.01)
                continue

            # Take seg_total samples
            segment = np.array(
                [self._audio_buf[i] for i in range(seg_total)],
                dtype=np.float32,
            )

            # Check for silence -> maybe reset
            if np.std(segment) < self.cfg.streaming.rms_silence:
                if time.time() - self._last_speech_time > self.cfg.streaming.reset_silence_sec:
                    self._maybe_reset()
                time.sleep(0.005)
                continue

            self._last_speech_time = time.time()

            try:
                waveform = torch.from_numpy(segment).unsqueeze(0)  # (1, seg_total)
                features, feat_len = self.feat_extractor(waveform, torch.tensor([seg_total]))
                hyps, self.asr_state = self.model.infer(
                    features, feat_len, self.cfg.streaming.asr_beam_width,
                    state=self.asr_state,
                )
                self.asr_hypothesis = hyps
            except Exception as e:
                logger.warning("EmformerASR inference error: %s", e)

            time.sleep(0.005)

    def _maybe_reset(self):
        """Reset ASR state after prolonged silence."""
        if self.asr_state is not None:
            logger.debug("Resetting Emformer state after %.0fs silence",
                         self.cfg.streaming.reset_silence_sec)
        self.asr_state = None
        self.asr_hypothesis = None
        self._prev_token_count = 0

    def get_delta(self, chunk_id: int) -> str:
        """
        Get new tokens since last call (delta snapshot).
        Returns transcript text for this chunk.
        """
        if self.asr_hypothesis is None:
            return ""

        try:
            tokens = self.token_processor(self.asr_hypothesis[0][0], lstrip=False)
            current_count = len(tokens) if isinstance(tokens, str) else 0
            if current_count <= self._prev_token_count:
                return ""
            delta = tokens[self._prev_token_count:]
            self._prev_token_count = current_count
            return delta
        except Exception as e:
            logger.warning("get_delta error: %s", e)
            return ""

    def save_transcript_delta(self, chunk_id: int, transcript_path: str):
        """Save delta text to transcript.json."""
        delta_text = self.get_delta(chunk_id)
        entry = {"chunk_id": chunk_id, "text": delta_text, "language": self.cfg.streaming.default_lang}

        with self._lock:
            self._transcript_entries.append(entry)
            try:
                with open(transcript_path, "w") as f:
                    json.dump(self._transcript_entries, f, indent=2)
            except Exception as e:
                logger.warning("Failed to write transcript.json: %s", e)
