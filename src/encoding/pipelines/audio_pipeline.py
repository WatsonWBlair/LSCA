# pipeline/audio_pipeline.py
# Audio pipeline: wav2vec2 frozen encoder → mean pool → (d_audio,)
# Input:  (T,) float32 numpy array at 16 kHz (from AudioBuffer)
# Output: (768,) or (1024,) depending on wav2vec2 variant

import logging
import numpy as np
import torch

from src.encoding.utils.config import SAMPLE_RATE

logger = logging.getLogger(__name__)


def audio_pipeline(
    chunk_waveform: np.ndarray,
    wav2vec2_model,
) -> torch.Tensor:
    """
    Extract audio embedding for one chunk.

    Args:
        chunk_waveform : (T,) float32 numpy array, 16 kHz mono
        wav2vec2_model : frozen Wav2Vec2Model (from model_loader)

    Returns: (d_audio,) tensor — mean of last hidden state across time
    """
    if chunk_waveform.dtype != np.float32:
        chunk_waveform = chunk_waveform.astype(np.float32)

    if len(chunk_waveform) == 0:
        logger.warning("audio_pipeline: empty waveform — returning zeros")
        # Return zero vector matching d_audio; get d_audio from model config
        d_audio = wav2vec2_model.config.hidden_size
        return torch.zeros(d_audio)

    assert chunk_waveform.dtype == np.float32, "wav2vec2 requires float32 input"

    t = torch.tensor(chunk_waveform).unsqueeze(0)   # (1, T)

    device = next(wav2vec2_model.parameters()).device
    t = t.to(device)

    with torch.no_grad():
        outputs = wav2vec2_model(t)
        hidden  = outputs.last_hidden_state          # (1, T', d_audio)

    return hidden.mean(dim=1).squeeze(0).cpu()       # (d_audio,)
