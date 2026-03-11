# encoding/streaming/dispatch.py
# Core dispatcher: run_all_pipelines(), handle_silent_chunk(), I/O helpers.
# Fires 3 embedding pipelines (video, phoneme, prosody) + transcript save in parallel.

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
from typing import Optional

import numpy as np
import torch

from encoding.config import CAMELSConfig, Modality
from encoding.pipelines.video import video_pipeline
from encoding.pipelines.phoneme import phoneme_pipeline, pad_phonemes
from encoding.pipelines.prosody import prosody_pipeline

logger = logging.getLogger(__name__)


def append_row(output_dir: str, fname: str, vector: torch.Tensor, chunk_id: int):
    """Append one row to an .npy file (creates on first call)."""
    path = os.path.join(output_dir, fname)
    row = vector.detach().cpu().numpy()
    if row.ndim == 1:
        row = row.reshape(1, -1)
    elif row.ndim == 2:
        row = row.reshape(1, *row.shape)

    if os.path.exists(path):
        existing = np.load(path)
        updated = np.concatenate([existing, row], axis=0)
    else:
        updated = row

    np.save(path, updated)


def append_zero_row(output_dir: str, chunk_id: int, cfg: CAMELSConfig):
    """Append zero vectors to all 3 .npy files for silent/failed chunks."""
    d = cfg.latent.d_latent
    max_ph = cfg.latent.max_phones

    append_row(output_dir, cfg.export.zv_file, torch.zeros(d), chunk_id)
    append_row(output_dir, cfg.export.zph_file, torch.zeros(max_ph, d), chunk_id)
    append_row(output_dir, cfg.export.zp_file, torch.zeros(d), chunk_id)


def check_row_sync(output_dir: str, chunk_id: int, cfg: CAMELSConfig):
    """Assert all 3 .npy files have chunk_id + 1 rows."""
    expected = chunk_id + 1
    for fname in [cfg.export.zv_file, cfg.export.zp_file, cfg.export.zph_file]:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            logger.error("Row sync: %s missing at chunk %d", fname, chunk_id)
            continue
        arr = np.load(path)
        if arr.shape[0] != expected:
            logger.error(
                "Row sync violation: %s has %d rows, expected %d",
                fname, arr.shape[0], expected,
            )


def save_chunk_registry(output_dir: str, registry: list, cfg: CAMELSConfig):
    path = os.path.join(output_dir, cfg.export.chunk_file)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def save_phoneme_metadata(output_dir: str, entries: list, cfg: CAMELSConfig):
    path = os.path.join(output_dir, cfg.export.phonemes_file)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


def run_all_pipelines(
    chunk_id: int,
    start_sec: float,
    end_sec: float,
    frame_buffer,
    audio_buffer,
    asr,
    models: dict,
    adapters: dict,
    prosody_stats: Optional[dict],
    output_dir: str,
    chunk_registry: list,
    phoneme_registry: list,
    cfg: CAMELSConfig,
) -> bool:
    """
    Called by FixedStrideScheduler on non-silent chunks.
    Fires 3 pipelines in parallel + transcript save.
    Returns True on success, False if zero rows appended.
    """
    # Register chunk
    entry = {
        "id": chunk_id,
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "modalities": [m.name.lower() for m in Modality],
    }
    chunk_registry.append(entry)
    save_chunk_registry(output_dir, chunk_registry, cfg)

    # Flush buffers
    frames, _ts = frame_buffer.flush_window(window_sec=cfg.streaming.window_sec)
    audio = audio_buffer.flush_window(window_sec=cfg.streaming.window_sec)

    if len(frames) == 0 or len(audio) == 0:
        logger.warning("Chunk %d: empty buffer — appending zero rows", chunk_id)
        append_zero_row(output_dir, chunk_id, cfg)
        check_row_sync(output_dir, chunk_id, cfg)
        return False

    fps = frame_buffer.fps

    # Fire all 3 pipelines + transcript save in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        fv = ex.submit(video_pipeline, frames, fps, models["marlin"], adapters["temporal_pool"], cfg)
        fph = ex.submit(
            phoneme_pipeline, audio, models["wav2vec2_ctc"], models["wav2vec2_processor"], cfg,
            chunk_offset_sec=start_sec,
        )
        fp = ex.submit(prosody_pipeline, audio, cfg, stats=prosody_stats)
        ft = ex.submit(
            asr.save_transcript_delta, chunk_id,
            os.path.join(output_dir, cfg.export.transcript_file),
        )

        v_raw = fv.result()                                    # (d_video,)
        ph_embs, ph_labels, ph_mask, segments = fph.result()   # per-phoneme
        p_raw = fp.result()                                     # (d_prosody,) numpy
        ft.result()  # transcript saved

    # Pad phonemes
    ph_embs_pad, ph_labels_pad, ph_mask_pad = pad_phonemes(
        ph_embs, ph_labels, ph_mask, cfg.latent.max_phones, cfg.latent.d_phoneme,
    )

    # Convert prosody to tensor
    p_raw_t = torch.tensor(p_raw, dtype=torch.float32)

    # Encode via adapters (inference: mu only)
    with torch.no_grad():
        z_v = adapters["video_adapter"].embed(v_raw.unsqueeze(0)).squeeze(0)
        z_ph = adapters["phoneme_adapter"](ph_embs_pad.unsqueeze(0)).squeeze(0)  # (MAX_PHONES, d_latent)
        z_p = adapters["prosody_adapter"].embed(p_raw_t.unsqueeze(0)).squeeze(0)

    # Append rows
    append_row(output_dir, cfg.export.zv_file, z_v, chunk_id)
    append_row(output_dir, cfg.export.zph_file, z_ph, chunk_id)
    append_row(output_dir, cfg.export.zp_file, z_p, chunk_id)

    # Phoneme metadata — full segment info with absolute timestamps
    phoneme_registry.append({
        "chunk_id": chunk_id,
        "count": len(segments),
        "segments": [
            {
                "label": s.get("label", ""),
                "label_id": s.get("label_id", -1),
                "start_sec": s.get("start_sec", 0.0),
                "end_sec": s.get("end_sec", 0.0),
            }
            for s in segments
        ],
    })
    save_phoneme_metadata(output_dir, phoneme_registry, cfg)

    check_row_sync(output_dir, chunk_id, cfg)
    logger.debug("Chunk %d [%.2f–%.2f s]: dispatched", chunk_id, start_sec, end_sec)
    return True


def handle_silent_chunk(
    chunk_id: int,
    start_sec: float,
    end_sec: float,
    output_dir: str,
    chunk_registry: list,
    cfg: CAMELSConfig,
):
    """Append zero rows for silent chunks to maintain row index invariant."""
    entry = {
        "id": chunk_id,
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "silent": True,
        "modalities": [m.name.lower() for m in Modality],
    }
    chunk_registry.append(entry)
    save_chunk_registry(output_dir, chunk_registry, cfg)
    append_zero_row(output_dir, chunk_id, cfg)
    check_row_sync(output_dir, chunk_id, cfg)
    logger.debug("Chunk %d [%.2f–%.2f s]: SILENT", chunk_id, start_sec, end_sec)
