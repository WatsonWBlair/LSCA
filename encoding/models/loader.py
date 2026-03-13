# encoding/models/loader.py
# Loads all frozen pretrained models used by the CAMELS pipeline.
# v8.1: MARLIN, wav2vec2-CTC (phoneme), Emformer RNN-T. No SONAR.

from __future__ import annotations

import logging

import torch
import torchaudio

from encoding.config import CAMELSConfig

logger = logging.getLogger(__name__)


def load_marlin(model_name: str, device: str = "cpu"):
    """
    Load MARLIN ViT-Base pretrained on YouTubeFaces (frozen).
    Output: (B, 768) when keep_seq=False.
    """
    from marlin_pytorch import Marlin

    logger.info("Loading MARLIN %s ...", model_name)
    model = Marlin.from_online(model_name)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info("MARLIN loaded — frozen")
    return model


def load_wav2vec2_ctc(model_name: str, device: str = "cpu") -> tuple:
    """
    Load wav2vec2 CTC phoneme recognition model (frozen).
    Uses facebook/wav2vec2-lv-60-espeak-cv-ft for IPA phoneme segmentation.

    Returns: (model, processor, num_phoneme_classes)
    The model produces both hidden_states (for embedding) and logits (for CTC decode).
    num_phoneme_classes is auto-detected from the model's output vocabulary.
    """
    from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor

    logger.info("Loading wav2vec2-CTC %s ...", model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    # Use FeatureExtractor only — avoids requiring espeak-ng system binary.
    # The phoneme string labels from processor.decode() are discarded downstream.
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    num_classes = model.config.vocab_size
    logger.info(
        "wav2vec2-CTC loaded — d_hidden=%d, vocab_size=%d, frozen",
        model.config.hidden_size, num_classes,
    )
    return model, processor, num_classes


def load_emformer(device: str = "cpu") -> tuple:
    """
    Load Emformer RNN-T streaming ASR (English, LibriSpeech, frozen).
    Returns (model, decoder, token_processor, feat_extractor).
    """
    logger.info("Loading Emformer RNN-T ...")
    bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
    get_model_fn = getattr(bundle, "get_model", None) or bundle._get_model
    emformer_model = get_model_fn().to(device).eval()
    emformer_decoder = bundle.get_decoder()
    token_processor = bundle.get_token_processor()
    feat_extractor = bundle.get_streaming_feature_extractor()

    for p in emformer_model.parameters():
        p.requires_grad_(False)

    logger.info("Emformer RNN-T loaded — English-only, frozen")
    return emformer_model, emformer_decoder, token_processor, feat_extractor


def load_all_models(
    cfg: CAMELSConfig,
    device: str = "cpu",
    load_emformer: bool = True,
    half: bool = False,
) -> dict:
    """
    Load all frozen models. Returns dict with keys:
      marlin, wav2vec2_ctc, wav2vec2_processor, num_phoneme_classes,
      emformer_model, emformer_decoder, token_processor, feat_extractor, device

    Args:
        load_emformer: if False, skip Emformer load and return None for its keys.
        half: if True, cast MARLIN and wav2vec2 to fp16 after loading.
    """
    _load_emformer = load_emformer  # avoid shadowing the module-level function

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available — falling back to CPU")
        device = "cpu"

    # Load MARLIN first so cv2 is imported before torchaudio/transformers
    # alter the Windows DLL search state (avoids "procedure not found" on cv2).
    marlin = load_marlin(cfg.streaming.marlin_model_name, device)
    wav2vec2_ctc, wav2vec2_proc, num_classes = load_wav2vec2_ctc(
        cfg.streaming.wav2vec2_ctc_model, device,
    )

    if half:
        marlin = marlin.half()
        wav2vec2_ctc = wav2vec2_ctc.half()
        logger.info("Models cast to fp16")

    if _load_emformer:
        em_model, em_decoder, tok_proc, feat_ext = load_emformer(device)
    else:
        logger.info("Skipping Emformer load (load_emformer=False)")
        em_model = em_decoder = tok_proc = feat_ext = None

    return {
        "marlin": marlin,
        "wav2vec2_ctc": wav2vec2_ctc,
        "wav2vec2_processor": wav2vec2_proc,
        "num_phoneme_classes": num_classes,
        "emformer_model": em_model,
        "emformer_decoder": em_decoder,
        "token_processor": tok_proc,
        "feat_extractor": feat_ext,
        "device": device,
    }
