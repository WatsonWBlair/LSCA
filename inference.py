# encoding/inference.py
# Inference utilities for 3-modality CAMELS v8.1 pipeline.

from __future__ import annotations

import concurrent.futures

import numpy as np
import torch

from config import CAMELSConfig, Modality
from pipelines.video import video_pipeline
from pipelines.phoneme import phoneme_pipeline, pad_phonemes
from pipelines.prosody import prosody_pipeline


def infer_chunk(
    chunk_frames: list[torch.Tensor],
    fps: float,
    audio_chunk: np.ndarray,
    models: dict,
    adapters: dict,
    cfg: CAMELSConfig,
    prosody_stats: dict | None = None,
    chunk_offset_sec: float = 0.0,
) -> dict:
    """
    Single-chunk inference: run all 3 pipelines in parallel, encode via adapters.

    Returns dict:
      z_v:          (d_latent,)
      z_ph:         (MAX_PHONES, d_latent)
      z_ph_pooled:  (d_latent,)
      z_p:          (d_latent,)
      ph_mask:      (MAX_PHONES,)
      segments:     list of phoneme dicts with start_sec/end_sec absolute timestamps
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        fv = ex.submit(video_pipeline, chunk_frames, fps, models["marlin"], adapters["temporal_pool"], cfg)
        fph = ex.submit(
            phoneme_pipeline, audio_chunk, models["wav2vec2_ctc"], models["wav2vec2_processor"], cfg,
            chunk_offset_sec=chunk_offset_sec,
        )
        fp = ex.submit(prosody_pipeline, audio_chunk, cfg, stats=prosody_stats)

        v_raw = fv.result()
        ph_embs, ph_labels, ph_mask, segments = fph.result()
        p_raw = fp.result()

    ph_embs_pad, ph_labels_pad, ph_mask_pad = pad_phonemes(
        ph_embs, ph_labels, ph_mask, cfg.latent.max_phones, cfg.latent.d_phoneme,
    )
    p_raw_t = torch.tensor(p_raw, dtype=torch.float32)

    with torch.no_grad():
        z_v = adapters["video_adapter"].embed(v_raw.unsqueeze(0)).squeeze(0)
        z_ph = adapters["phoneme_adapter"](ph_embs_pad.unsqueeze(0)).squeeze(0)
        z_ph_pooled = adapters["phoneme_attn_pool"](
            z_ph.unsqueeze(0), ph_mask_pad.unsqueeze(0),
        ).squeeze(0)
        z_p = adapters["prosody_adapter"].embed(p_raw_t.unsqueeze(0)).squeeze(0)

    return {
        "z_v": z_v,
        "z_ph": z_ph,
        "z_ph_pooled": z_ph_pooled,
        "z_p": z_p,
        "ph_mask": ph_mask_pad,
        "segments": segments,
        "modality_ids": {
            "z_v": Modality.VIDEO,
            "z_ph": Modality.PHONEME,
            "z_ph_pooled": Modality.PHONEME,
            "z_p": Modality.PROSODY,
        },
    }


def infer_batch(
    v_raw: torch.Tensor,
    ph_raw: torch.Tensor,
    ph_mask: torch.Tensor,
    p_raw: torch.Tensor,
    adapters: dict,
) -> dict[str, torch.Tensor]:
    """
    Batch inference on pre-extracted raw features.

    Args:
      v_raw:   (B, d_video)
      ph_raw:  (B, MAX_PHONES, d_phoneme)
      ph_mask: (B, MAX_PHONES)
      p_raw:   (B, d_prosody)

    Returns: dict of z_v (B, d_latent), z_ph_pooled (B, d_latent), z_p (B, d_latent)
    """
    with torch.no_grad():
        z_v = adapters["video_adapter"].embed(v_raw)
        z_ph_seq = adapters["phoneme_adapter"](ph_raw)
        z_ph_pooled = adapters["phoneme_attn_pool"](z_ph_seq, ph_mask)
        z_p = adapters["prosody_adapter"].embed(p_raw)

    return {
        "z_v": z_v,
        "z_ph_pooled": z_ph_pooled,
        "z_p": z_p,
        "modality_ids": {
            "z_v": Modality.VIDEO,
            "z_ph_pooled": Modality.PHONEME,
            "z_p": Modality.PROSODY,
        },
    }
