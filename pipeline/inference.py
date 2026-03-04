# pipeline/inference.py
# Inference entry point: infer_chunk() processes one chunk and returns all 4 embeddings.
# At inference: embed() only — mu, no sampling, no decoder, no VelocityNet.
# All 4 pipelines run in parallel; approx latency 150–300 ms on mid-range GPU.

import concurrent.futures
import logging
from typing import Optional, Tuple

import numpy as np
import torch

from src.encoding.utils.config import WINDOW_SEC, D_LATENT
from src.encoding.pipelines.video_pipeline   import video_pipeline
from src.encoding.pipelines.audio_pipeline   import audio_pipeline
from src.encoding.pipelines.prosody_pipeline import prosody_pipeline
from src.encoding.pipelines.text_pipeline    import text_pipeline

logger = logging.getLogger(__name__)


def infer_chunk(
    chunk_id:      int,
    frame_buffer,              # FrameBuffer instance
    audio_buffer,              # AudioBuffer instance
    asr,                       # EmformerASR instance
    models:        dict,       # from model_loader.load_all_models()
    adapters:      dict,       # from adapters.build_adapters() (loaded from checkpoint)
    prosody_stats: Optional[dict],
    language:      str = "eng_Latn",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
    """
    Run full inference for one scheduler tick.

    Returns:
        z_v         : (1024,) video embedding
        z_a         : (1024,) audio embedding
        z_p         : (1024,) prosody embedding
        z_t         : (1024,) text embedding
        transcript  : decoded string from this chunk's token delta
        language    : language code (e.g., 'eng_Latn')

    All four z_* vectors are SEPARATE — do NOT fuse here.
    Teammate's HyperGNN handles fusion.
    """
    frames, _ = frame_buffer.flush_window(window_sec=WINDOW_SEC)
    audio     = audio_buffer.flush_window(window_sec=WINDOW_SEC)
    fps       = frame_buffer.fps

    # Run all 4 pipelines in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        fv = ex.submit(video_pipeline,   frames, fps, models["marlin"], adapters["temporal_pool"])
        fa = ex.submit(audio_pipeline,   audio, models["wav2vec2"])
        fp = ex.submit(prosody_pipeline, audio, stats=prosody_stats)
        ft = ex.submit(text_pipeline,    chunk_id, asr, models["sonar"], language)

        v_raw               = fv.result()              # (768,)
        a_raw               = fa.result()              # (d_audio,)
        p_raw               = fp.result()              # (22,) numpy
        t_raw, transcript, lang = ft.result()          # (1024,), str, str

    p_raw_t = torch.tensor(p_raw, dtype=torch.float32)

    with torch.no_grad():
        z_v = adapters["video_adapter"].embed(v_raw.unsqueeze(0)).squeeze(0)     # (1024,)
        z_a = adapters["audio_adapter"].embed(a_raw.unsqueeze(0)).squeeze(0)     # (1024,)
        z_p = adapters["prosody_adapter"].embed(p_raw_t.unsqueeze(0)).squeeze(0) # (1024,)
        z_t = adapters["text_adapter"].embed(t_raw.unsqueeze(0)).squeeze(0)      # (1024,)

    return z_v, z_a, z_p, z_t, transcript, lang


def infer_batch(
    v_raws:        torch.Tensor,   # (B, 768)
    a_raws:        torch.Tensor,   # (B, d_audio)
    p_raws:        torch.Tensor,   # (B, 22)
    t_raws:        torch.Tensor,   # (B, 1024)
    adapters:      dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch inference for pre-extracted raw features (used during evaluation).
    Returns: (z_v, z_a, z_p, z_t) each (B, 1024).
    """
    with torch.no_grad():
        z_v = adapters["video_adapter"].embed(v_raws)    # (B, 1024)
        z_a = adapters["audio_adapter"].embed(a_raws)    # (B, 1024)
        z_p = adapters["prosody_adapter"].embed(p_raws)  # (B, 1024)
        z_t = adapters["text_adapter"].embed(t_raws)     # (B, 1024)
    return z_v, z_a, z_p, z_t
