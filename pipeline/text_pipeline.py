# pipeline/text_pipeline.py
# Text pipeline: Emformer RNN-T continuous ASR + SONAR encoder → (1024,)
# The EmformerASR class runs a background thread that continuously processes
# audio in 160 ms segments. The text_pipeline() function reads token deltas
# at each scheduler tick (no state reset between chunks).
# Ref: Zhang et al. 2021 / torchaudio; Duquenne et al. 2023 / SONAR

import json
import logging
import threading
import time
from typing import Tuple

import numpy as np
import torch

from pipeline.config import (
    SAMPLE_RATE, HOP, SEG_HOPS, RC_HOPS, SEG_TOTAL,
    ASR_BEAM_WIDTH, RESET_SILENCE_SEC, DEFAULT_LANG,
    D_LATENT, TRANSCRIPT_FILE,
)

logger = logging.getLogger(__name__)


# ── SONAR text encoding ───────────────────────────────────────────────────────

def encode_text_sonar(
    transcript: str,
    sonar_encoder,
    language: str = DEFAULT_LANG,
) -> torch.Tensor:
    """
    Encode a transcript string into a 1024-D SONAR embedding.
    sonar_encoder : TextToEmbeddingModelPipeline (from model_loader)
    Returns: (1024,) float32 tensor on CPU.

    IMPORTANT: The SONAR checkpoint used here MUST match what the teammate's
    SONAR decoder expects. Checkpoint: text_sonar_basic_encoder.
    """
    if not transcript or not transcript.strip():
        return torch.zeros(D_LATENT)

    with torch.no_grad():
        emb = sonar_encoder.predict([transcript.strip()], source_lang=language)
    return emb.squeeze(0).cpu()   # (1024,)


# ── EmformerASR — stateful streaming ASR class ───────────────────────────────

class EmformerASR:
    """
    Wraps Emformer RNN-T in a background thread for continuous streaming ASR.

    Design:
    - The background thread pulls audio from the audio_buffer deque,
      processes 160 ms segments via Emformer, and keeps asr_state +
      asr_hypothesis alive for the entire session.
    - At each scheduler tick, text_pipeline() calls get_delta(chunk_id) to
      read new tokens since the last tick (token delta). State is NEVER reset
      between chunks (only after a >RESET_SILENCE_SEC silence gap).
    - transcript_log is written to TRANSCRIPT_FILE for handoff to teammate.

    Usage:
        asr = EmformerASR(models)
        asr.start(audio_buffer)     # start background thread
        ...
        delta, lang = asr.get_delta(chunk_id)  # at each scheduler tick
        ...
        asr.stop()
    """

    def __init__(self, models: dict, output_dir: str = "."):
        self.emformer_model   = models["emformer_model"]
        self.emformer_decoder = models["emformer_decoder"]
        self.token_processor  = models["token_processor"]
        self.feat_extractor   = models["feat_extractor"]
        self.output_dir       = output_dir

        # Streaming state — persists for entire session
        self._lock            = threading.Lock()
        self._asr_state       = None
        self._asr_hypothesis  = None
        self._last_token_count = 0
        self._last_token_time  = time.time()   # for long-session reset
        self._transcript_log  : dict = {}       # {chunk_id: {text, language}}

        self._stop_event      = threading.Event()
        self._thread          = None
        self._audio_buffer    = None

    # ── Background ASR loop ───────────────────────────────────────────────

    def _asr_loop(self):
        idx = 0
        while not self._stop_event.is_set():
            buf = np.array(list(self._audio_buffer))
            if len(buf) < idx + SEG_TOTAL:
                time.sleep(0.01)
                continue

            segment    = buf[idx: idx + SEG_TOTAL]
            seg_tensor = torch.tensor(segment.astype(np.float32)).unsqueeze(0)  # (1, 3200)
            seg_len    = torch.tensor([SEG_TOTAL])

            try:
                with torch.no_grad():
                    feats, feat_len = self.feat_extractor(seg_tensor, seg_len)
                    hyps, new_state = self.emformer_decoder.infer(
                        feats, feat_len,
                        beam_width=ASR_BEAM_WIDTH,
                        state=self._asr_state,
                        hypothesis=self._asr_hypothesis,
                    )
                with self._lock:
                    self._asr_state      = new_state
                    self._asr_hypothesis = hyps[0]
                    self._last_token_time = time.time()
            except Exception as e:
                logger.warning("EmformerASR loop error: %s", e)

            idx += SEG_HOPS * HOP   # advance by one segment (not right-context)

    # ── Long-session safety reset ─────────────────────────────────────────

    def _maybe_reset(self):
        """Reset ASR state after a long silence gap to prevent unbounded memory."""
        with self._lock:
            elapsed = time.time() - self._last_token_time
            if elapsed > RESET_SILENCE_SEC:
                logger.info(
                    "EmformerASR: %.1f s silence — resetting state (long-session safety)",
                    elapsed,
                )
                self._asr_state       = None
                self._asr_hypothesis  = None
                self._last_token_count = 0

    # ── Public API ────────────────────────────────────────────────────────

    def start(self, audio_buffer):
        """Start the background ASR thread."""
        self._audio_buffer = audio_buffer
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._asr_loop, daemon=True, name="EmformerASR"
        )
        self._thread.start()
        logger.info("EmformerASR background thread started")

    def stop(self):
        """Signal the background thread to stop and join."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("EmformerASR background thread stopped")

    def get_delta(self, chunk_id: int, language: str = DEFAULT_LANG) -> Tuple[str, str]:
        """
        Read new tokens since the last scheduler tick (delta snapshot).
        Does NOT reset asr_state or asr_hypothesis.

        Returns:
            (text, language)
              text     : decoded string of new tokens (empty string if silence)
              language : language code passed in (echoed back for transcript.json)
        """
        self._maybe_reset()

        with self._lock:
            if self._asr_hypothesis is None:
                text = ""
            else:
                tokens     = self._asr_hypothesis.tokens
                new_tokens = tokens[self._last_token_count:]
                self._last_token_count = len(tokens)
                try:
                    text = self.token_processor(new_tokens) if new_tokens else ""
                except Exception as e:
                    logger.warning("token_processor error: %s", e)
                    text = ""

        entry = {"text": text, "language": language}
        self._transcript_log[str(chunk_id)] = entry
        self._save_transcript()
        return text, language

    def get_transcript_log(self) -> dict:
        return dict(self._transcript_log)

    def _save_transcript(self):
        import os, json
        path = os.path.join(self.output_dir, TRANSCRIPT_FILE)
        try:
            with open(path, "w") as f:
                json.dump(self._transcript_log, f, indent=2)
        except Exception as e:
            logger.warning("Could not write transcript.json: %s", e)


# ── Full text pipeline (called at each scheduler tick) ────────────────────────

def text_pipeline(
    chunk_id: int,
    asr: EmformerASR,
    sonar_encoder,
    language: str = DEFAULT_LANG,
) -> Tuple[torch.Tensor, str, str]:
    """
    Get the transcript delta for this chunk and encode it with SONAR.

    Args:
        chunk_id      : int — current chunk index
        asr           : EmformerASR instance (running background thread)
        sonar_encoder : SONAR TextToEmbeddingModelPipeline (frozen)
        language      : BCP-47 + script code (e.g., 'eng_Latn')

    Returns:
        (t_raw, transcript, language)
          t_raw      : (1024,) SONAR embedding (zeros if silent/no tokens)
          transcript : decoded string (empty if silent)
          language   : language code
    """
    transcript, lang = asr.get_delta(chunk_id, language=language)

    if not transcript:
        return torch.zeros(D_LATENT), "", lang

    t_raw = encode_text_sonar(transcript, sonar_encoder, language=lang)
    return t_raw, transcript, lang
