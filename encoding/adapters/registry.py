# encoding/adapters/registry.py
# Pluggable adapter construction from CAMELSConfig.
# Adapters are built only for enabled modalities, supporting ablation studies.

from __future__ import annotations

import logging

import torch

from encoding.config import CAMELSConfig
from encoding.adapters.base import AVAEAdapter, TemporalAttentionPool
from encoding.adapters.phoneme import PhonemeAdapter, PhonemeAttnPool, PhonemeProbeHead
from encoding.adapters.velocity import VelocityNet

logger = logging.getLogger(__name__)


def build_adapters(cfg: CAMELSConfig) -> dict:
    """
    Instantiate all adapters from config.
    Only builds components for enabled modalities, supporting modality-drop experiments.
    """
    adapters: dict[str, torch.nn.Module] = {}
    lat = cfg.latent
    adp = cfg.adapter

    if cfg.modality.video_enabled:
        adapters["temporal_pool"] = TemporalAttentionPool(d=lat.d_video)
        adapters["video_adapter"] = AVAEAdapter(
            d_in=lat.d_video, d_latent=lat.d_latent, hidden=adp.hidden_high,
        )

    if cfg.modality.phoneme_enabled:
        adapters["phoneme_adapter"] = PhonemeAdapter(
            d_in=lat.d_phoneme, d_latent=lat.d_latent,
        )
        adapters["phoneme_attn_pool"] = PhonemeAttnPool(d=lat.d_latent)
        if lat.num_phoneme_classes > 0:
            adapters["phoneme_probe"] = PhonemeProbeHead(
                d=lat.d_latent,
                n_classes=lat.num_phoneme_classes,
                hidden=adp.hidden_probe,
            )

    if cfg.modality.prosody_enabled:
        adapters["prosody_adapter"] = AVAEAdapter(
            d_in=lat.d_prosody, d_latent=lat.d_latent, hidden=adp.hidden_prosody,
        )

    # VelocityNets for bidirectional FM (video <-> phoneme)
    if cfg.modality.video_enabled and cfg.modality.phoneme_enabled:
        adapters["velocity_vph"] = VelocityNet(d=lat.d_latent)
        adapters["velocity_phv"] = VelocityNet(d=lat.d_latent)

    logger.info(
        "Built adapters: %s (d_latent=%d)", list(adapters.keys()), lat.d_latent,
    )
    return adapters


def save_adapters(adapters: dict, path: str):
    """Save all adapter state dicts to a .pt checkpoint."""
    torch.save({k: v.state_dict() for k, v in adapters.items()}, path)


def load_adapters(
    path: str,
    cfg: CAMELSConfig,
    device: str = "cpu",
) -> dict:
    """Load adapter state dicts from a .pt checkpoint."""
    adapters = build_adapters(cfg)
    state = torch.load(path, map_location=device, weights_only=True)
    for k, module in adapters.items():
        if k in state:
            module.load_state_dict(state[k])
        else:
            logger.warning("Checkpoint missing key '%s' — using random init", k)
        module.to(device)
    return adapters


def trainable_params(adapters: dict, exclude: set[str] | None = None) -> list:
    """Return trainable parameters, optionally excluding named modules."""
    exclude = exclude or set()
    params = []
    for name, module in adapters.items():
        if name not in exclude:
            params.extend(module.parameters())
    return params
