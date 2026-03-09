# encoding/pipelines/phoneme.py
# Phoneme pipeline: wav2vec2-CTC segmentation + per-phoneme acoustic embeddings.
# One forward pass produces both CTC logits (segmentation) and hidden states (embedding).
# Ref: Xu et al. ICASSP 2021; Shahin et al. Speech Communication 2025

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from encoding.config import CAMELSConfig

logger = logging.getLogger(__name__)


def _ctc_decode_with_boundaries(
    logits: torch.Tensor,
    processor,
) -> list[dict]:
    """
    Greedy CTC decode producing phoneme labels with frame boundaries.

    Returns list of dicts:
      [{'label': str, 'label_id': int, 'start_frame': int, 'end_frame': int}, ...]
    """
    # Greedy decode: argmax at each frame
    pred_ids = torch.argmax(logits, dim=-1).squeeze(0)  # (T_frames,)

    segments: list[dict] = []
    prev_id = -1
    start_frame = 0

    blank_id = 0  # CTC blank is typically index 0

    for i, pid in enumerate(pred_ids.tolist()):
        if pid != prev_id:
            if prev_id != blank_id and prev_id >= 0:
                segments.append({
                    "label_id": prev_id,
                    "start_frame": start_frame,
                    "end_frame": i,
                })
            start_frame = i
            prev_id = pid

    # Final segment
    if prev_id != blank_id and prev_id >= 0:
        segments.append({
            "label_id": prev_id,
            "start_frame": start_frame,
            "end_frame": len(pred_ids),
        })

    # Decode label strings
    for seg in segments:
        seg["label"] = processor.decode([seg["label_id"]]).strip()

    return segments


def phoneme_pipeline(
    audio_chunk: np.ndarray,
    wav2vec2_ctc,
    wav2vec2_processor,
    cfg: CAMELSConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """
    Full phoneme pipeline for one chunk.

    Args:
        audio_chunk: (T,) float32 at sample_rate Hz
        wav2vec2_ctc: frozen Wav2Vec2ForCTC model
        wav2vec2_processor: Wav2Vec2Processor
        cfg: CAMELSConfig

    Returns:
        phone_embs:   (num_phones, d_phoneme) — per-phoneme acoustic embeddings
        phone_labels: (num_phones,)           — phoneme class IDs
        phone_mask:   (num_phones,)           — all ones (before padding)
        segments:     list of dicts with label, label_id, start_frame, end_frame
    """
    d_phoneme = cfg.latent.d_phoneme
    device = next(wav2vec2_ctc.parameters()).device

    if len(audio_chunk) == 0:
        logger.warning("phoneme_pipeline: empty audio — returning zeros")
        return (
            torch.zeros(0, d_phoneme),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.bool),
            [],
        )

    # Prepare input
    inputs = wav2vec2_processor(
        audio_chunk,
        sampling_rate=cfg.streaming.sample_rate,
        return_tensors="pt",
        padding=False,
    )
    input_values = inputs.input_values.to(device)

    # Single forward pass: get both hidden states and logits
    with torch.no_grad():
        outputs = wav2vec2_ctc(input_values, output_hidden_states=True)
        logits = outputs.logits                      # (1, T_frames, vocab_size)
        hidden = outputs.hidden_states[-1]           # (1, T_frames, d_phoneme)

    # CTC decode with frame boundaries
    segments = _ctc_decode_with_boundaries(logits, wav2vec2_processor)

    if not segments:
        logger.debug("phoneme_pipeline: no phonemes detected")
        return (
            torch.zeros(0, d_phoneme),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.bool),
            [],
        )

    hidden_sq = hidden.squeeze(0).cpu()  # (T_frames, d_phoneme)

    # Mean-pool hidden states per phoneme segment
    phone_embs_list = []
    phone_labels_list = []

    for seg in segments:
        s, e = seg["start_frame"], seg["end_frame"]
        if s >= hidden_sq.shape[0]:
            break
        e = min(e, hidden_sq.shape[0])
        if e <= s:
            continue
        emb = hidden_sq[s:e].mean(dim=0)  # (d_phoneme,)
        phone_embs_list.append(emb)
        phone_labels_list.append(seg["label_id"])

    if not phone_embs_list:
        return (
            torch.zeros(0, d_phoneme),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.bool),
            segments,
        )

    phone_embs = torch.stack(phone_embs_list)                     # (num_phones, d_phoneme)
    phone_labels = torch.tensor(phone_labels_list, dtype=torch.long)  # (num_phones,)
    phone_mask = torch.ones(len(phone_embs_list), dtype=torch.bool)   # (num_phones,)

    return phone_embs, phone_labels, phone_mask, segments


def pad_phonemes(
    phone_embs: torch.Tensor,
    phone_labels: torch.Tensor,
    phone_mask: torch.Tensor,
    max_phones: int,
    d: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad/truncate phoneme tensors to (max_phones, d).

    Returns:
        padded_embs:   (max_phones, d)
        padded_labels: (max_phones,)
        padded_mask:   (max_phones,) — 1 = real, 0 = padding
    """
    n = phone_embs.shape[0]

    if n >= max_phones:
        # Truncate
        return (
            phone_embs[:max_phones],
            phone_labels[:max_phones],
            phone_mask[:max_phones],
        )

    # Pad
    padded_embs = torch.zeros(max_phones, d)
    padded_labels = torch.zeros(max_phones, dtype=torch.long)
    padded_mask = torch.zeros(max_phones, dtype=torch.bool)

    padded_embs[:n] = phone_embs
    padded_labels[:n] = phone_labels
    padded_mask[:n] = phone_mask

    return padded_embs, padded_labels, padded_mask
