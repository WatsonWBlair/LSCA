# models/model_loader.py
# Loads all frozen pretrained models used by the CAMELS pipeline.
# Call load_all_models() once at startup and pass the returned dict everywhere.

import logging
import torch
import torchaudio

from src.encoding.utils.config import (
    MARLIN_MODEL_NAME, WAV2VEC2_MODEL, SONAR_ENCODER, SONAR_TOKENIZER,
)

logger = logging.getLogger(__name__)


def load_marlin(device: str = "cpu"):
    """
    Load MARLIN ViT-Base pretrained on YouTubeFaces.
    Downloads and caches on first run to .marlin/
    Encoder only — extract_features() is the only method used at runtime.
    Output: (B, 768) when keep_seq=False.
    """
    from marlin_pytorch import Marlin
    logger.info("Loading MARLIN %s ...", MARLIN_MODEL_NAME)
    model = Marlin.from_online(MARLIN_MODEL_NAME)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info("MARLIN loaded — d_model=768, frozen")
    return model


def load_wav2vec2(device: str = "cpu"):
    """
    Load wav2vec2-base encoder (frozen).
    Input: (1, T) float32 waveform at 16 kHz.
    Output: mean pool last_hidden_state → (768,).
    """
    from transformers import Wav2Vec2Model
    logger.info("Loading wav2vec2 %s ...", WAV2VEC2_MODEL)
    model = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info("wav2vec2 loaded — d_model=768, frozen")
    return model


def load_sonar_encoder(device: str = "cpu"):
    """
    Load SONAR multilingual text encoder (frozen).
    Encodes a transcript string → (1024,) vector in SONAR's shared space.
    The teammate's SONAR decoder must use the SAME checkpoint.
    Checkpoint: text_sonar_basic_encoder
    """
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    logger.info("Loading SONAR encoder %s ...", SONAR_ENCODER)
    encoder = TextToEmbeddingModelPipeline(
        encoder=SONAR_ENCODER,
        tokenizer=SONAR_TOKENIZER,
        device=torch.device(device),
    )
    logger.info("SONAR encoder loaded — d=1024, frozen")
    return encoder


def load_emformer(device: str = "cpu"):
    """
    Load Emformer RNN-T streaming ASR (English, LibriSpeech-trained, frozen).
    Returns (model, decoder, token_processor, feat_extractor) tuple.

    LIMITATION: English-only. For multilingual sessions, swap with Meta MMS:
      torchaudio.pipelines.MMS_300M  (1000+ languages)

    The feat_extractor + model + decoder run in a background thread.
    """
    logger.info("Loading Emformer RNN-T (EMFORMER_RNNT_BASE_LIBRISPEECH) ...")
    bundle          = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
    # torchaudio >= 2.0 moved get_model() → _get_model()
    get_model_fn    = getattr(bundle, "get_model", None) or bundle._get_model
    emformer_model  = get_model_fn().to(device).eval()
    emformer_decoder = bundle.get_decoder()
    token_processor  = bundle.get_token_processor()
    feat_extractor   = bundle.get_streaming_feature_extractor()

    for p in emformer_model.parameters():
        p.requires_grad_(False)

    logger.info("Emformer RNN-T loaded — English-only, frozen")
    return emformer_model, emformer_decoder, token_processor, feat_extractor


def load_all_models(device: str = "cpu") -> dict:
    """
    Load all frozen models in one call. Returns a dict with keys:
      marlin, wav2vec2, sonar,
      emformer_model, emformer_decoder, token_processor, feat_extractor

    Usage:
        models = load_all_models(device='cuda')
    """
    assert device in ("cpu", "cuda", "mps"), f"Unknown device: {device}"
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but not available — falling back to CPU")
        device = "cpu"

    em_model, em_decoder, tok_proc, feat_ext = load_emformer(device)

    return {
        "marlin":          load_marlin(device),
        "wav2vec2":        load_wav2vec2(device),
        "sonar":           load_sonar_encoder(device),
        "emformer_model":  em_model,
        "emformer_decoder": em_decoder,
        "token_processor": tok_proc,
        "feat_extractor":  feat_ext,
        "device":          device,
    }
