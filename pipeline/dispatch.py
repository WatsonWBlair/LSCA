# pipeline/dispatch.py
# Core dispatcher: run_all_pipelines(), I/O helpers (append_row, check_row_sync,
# append_zero_row), and chunk registry management.
# Called by FixedStrideScheduler at each non-silent tick.

import json
import logging
import os
import concurrent.futures
from typing import Optional

import numpy as np
import torch

from src.encoding.utils.config import (
    WINDOW_SEC, STRIDE_SEC, RMS_SILENCE,
    ZV_FILE, ZA_FILE, ZP_FILE, ZT_FILE,
    CHUNK_FILE, D_LATENT, SAMPLE_RATE,
)
from src.encoding.pipelines.video_pipeline   import video_pipeline
from src.encoding.pipelines.audio_pipeline   import audio_pipeline
from src.encoding.pipelines.prosody_pipeline import prosody_pipeline
from src.encoding.pipelines.text_pipeline    import text_pipeline

logger = logging.getLogger(__name__)


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _npy_path(output_dir: str, fname: str) -> str:
    return os.path.join(output_dir, fname)


def append_row(output_dir: str, fname: str, vector: torch.Tensor, chunk_id: int):
    """
    Append one (D,) row to an .npy file (creates file on first call).
    Row index after append == chunk_id + 1.
    """
    path  = _npy_path(output_dir, fname)
    row   = vector.detach().cpu().numpy().reshape(1, -1)   # (1, D)

    if os.path.exists(path):
        existing = np.load(path)
        updated  = np.concatenate([existing, row], axis=0)
    else:
        updated = row

    np.save(path, updated)


def append_zero_row(output_dir: str, chunk_id: int):
    """
    Append a 1024-D zero vector to all four .npy files.
    Called for silent chunks and failed chunks to preserve row index invariant.
    """
    zero = torch.zeros(D_LATENT)
    for fname in [ZV_FILE, ZA_FILE, ZP_FILE, ZT_FILE]:
        append_row(output_dir, fname, zero, chunk_id)


def check_row_sync(output_dir: str, chunk_id: int):
    """
    Assert that all four .npy files have exactly chunk_id + 1 rows
    and that the latent dim is D_LATENT. Logs errors without raising
    so that a sync violation never crashes the live pipeline.
    """
    for fname in [ZV_FILE, ZA_FILE, ZP_FILE, ZT_FILE]:
        path = _npy_path(output_dir, fname)
        if not os.path.exists(path):
            logger.error("Row sync: %s missing at chunk %d", fname, chunk_id)
            continue
        arr = np.load(path)
        if arr.shape[0] != chunk_id + 1:
            logger.error(
                "Row sync violation: %s has %d rows, expected %d (chunk %d)",
                fname, arr.shape[0], chunk_id + 1, chunk_id,
            )
        if arr.shape[1] != D_LATENT:
            logger.error(
                "Wrong latent dim in %s: got %d, expected %d",
                fname, arr.shape[1], D_LATENT,
            )


def save_chunk_registry(output_dir: str, registry: list):
    path = os.path.join(output_dir, CHUNK_FILE)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


# ── Main dispatcher ───────────────────────────────────────────────────────────

def run_all_pipelines(
    chunk_id:     int,
    start_sec:    float,
    end_sec:      float,
    frame_buffer,          # FrameBuffer instance
    audio_buffer,          # AudioBuffer instance
    asr,                   # EmformerASR instance
    models:       dict,    # from model_loader.load_all_models()
    adapters:     dict,    # from adapters.build_adapters()
    prosody_stats: Optional[dict],
    output_dir:   str,
    chunk_registry: list,
    language:     str = "eng_Latn",
) -> bool:
    """
    Called by FixedStrideScheduler after silence check passes.

    Steps:
      1. Register chunk in chunk_registry and save to chunks.json.
      2. Flush ring buffers for the current window.
      3. If frames or audio are empty → append zero rows and return False.
      4. Fire all 4 pipelines in parallel via ThreadPoolExecutor.
      5. Encode each modality via adapter.embed() (mu, no sampling).
      6. Append one row per modality to each .npy file.
      7. Sanity-check row sync.

    Returns True on success, False if zero rows were appended.
    """
    # 1. Register chunk
    entry = {"id": chunk_id, "start_sec": round(start_sec, 3), "end_sec": round(end_sec, 3)}
    chunk_registry.append(entry)
    save_chunk_registry(output_dir, chunk_registry)

    # 2. Flush buffers
    frames, _timestamps = frame_buffer.flush_window(window_sec=WINDOW_SEC)
    audio               = audio_buffer.flush_window(window_sec=WINDOW_SEC)

    # 3. Empty check
    if len(frames) == 0 or len(audio) == 0:
        logger.warning(
            "Chunk %d: empty buffer (frames=%d, audio=%d) — appending zero rows",
            chunk_id, len(frames), len(audio),
        )
        append_zero_row(output_dir, chunk_id)
        check_row_sync(output_dir, chunk_id)
        return False

    fps = frame_buffer.fps

    # 4. Fire all 4 pipelines in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        fv = ex.submit(
            video_pipeline,
            frames, fps,
            models["marlin"],
            adapters["temporal_pool"],
        )
        fa = ex.submit(
            audio_pipeline,
            audio,
            models["wav2vec2"],
        )
        fp = ex.submit(
            prosody_pipeline,
            audio,
            stats=prosody_stats,
        )
        ft = ex.submit(
            text_pipeline,
            chunk_id,
            asr,
            models["sonar"],
            language,
        )

        v_raw = fv.result()                         # (768,)
        a_raw = fa.result()                         # (d_audio,)
        p_raw = fp.result()                         # (22,) numpy
        t_raw, transcript, lang = ft.result()       # (1024,), str, str

    # Convert prosody to tensor
    p_raw_t = torch.tensor(p_raw, dtype=torch.float32)

    # 5. Encode via adapter.embed() — mu only, no sampling
    with torch.no_grad():
        z_v = adapters["video_adapter"].embed(v_raw.unsqueeze(0)).squeeze(0)    # (1024,)
        z_a = adapters["audio_adapter"].embed(a_raw.unsqueeze(0)).squeeze(0)    # (1024,)
        z_p = adapters["prosody_adapter"].embed(p_raw_t.unsqueeze(0)).squeeze(0) # (1024,)
        z_t = adapters["text_adapter"].embed(t_raw.unsqueeze(0)).squeeze(0)     # (1024,)

    # 6. Append rows
    append_row(output_dir, ZV_FILE, z_v, chunk_id)
    append_row(output_dir, ZA_FILE, z_a, chunk_id)
    append_row(output_dir, ZP_FILE, z_p, chunk_id)
    append_row(output_dir, ZT_FILE, z_t, chunk_id)

    # 7. Row sync check
    check_row_sync(output_dir, chunk_id)

    logger.debug(
        "Chunk %d [%.2f–%.2f s]: dispatched | transcript='%s'",
        chunk_id, start_sec, end_sec, transcript[:50] if transcript else "",
    )
    return True


def handle_silent_chunk(
    chunk_id:   int,
    start_sec:  float,
    end_sec:    float,
    output_dir: str,
    chunk_registry: list,
):
    """
    Called by FixedStrideScheduler when RMS < threshold.
    Registers the chunk and appends zero rows to maintain row index invariant.
    """
    entry = {
        "id":        chunk_id,
        "start_sec": round(start_sec, 3),
        "end_sec":   round(end_sec, 3),
        "silent":    True,
    }
    chunk_registry.append(entry)
    save_chunk_registry(output_dir, chunk_registry)
    append_zero_row(output_dir, chunk_id)
    check_row_sync(output_dir, chunk_id)
    logger.debug("Chunk %d [%.2f–%.2f s]: SILENT", chunk_id, start_sec, end_sec)
