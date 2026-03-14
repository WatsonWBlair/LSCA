# encoding/training/momentum.py
# Momentum Contrast (MoCo) components for CAMELS v8.1.
#   MoCoQueue: circular buffer of key embeddings for negative sampling
#   MomentumEncoderManager: EMA copies of adapters + per-modality queues

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from encoding.config import CAMELSConfig


class MoCoQueue:
    """Circular buffer of L2-normalized key embeddings for MoCo negatives."""

    def __init__(self, d_latent: int, queue_size: int, device: torch.device | str):
        self.K = queue_size
        self._queue = F.normalize(
            torch.randn(d_latent, queue_size, device=device), dim=0
        )
        self._ptr = 0

    @property
    def queue(self) -> torch.Tensor:
        return self._queue

    @torch.no_grad()
    def enqueue_dequeue(self, keys: torch.Tensor) -> None:
        """Overwrite oldest slots with new keys. keys: (B, d_latent), L2-normalized."""
        B = keys.shape[0]
        assert B <= self.K, f"Batch size {B} exceeds queue size {self.K}"
        ptr = self._ptr
        end = ptr + B
        if end <= self.K:
            self._queue[:, ptr:end] = keys.T
        else:
            first = self.K - ptr
            self._queue[:, ptr:] = keys[:first].T
            self._queue[:, : B - first] = keys[first:].T
        self._ptr = end % self.K


class MomentumEncoderManager:
    """EMA copies of query adapters + per-modality MoCo queues.

    Excluded adapters (not momentum-tracked): velocity_*, phoneme_probe, temporal_pool.
    """

    _EXCLUDE = {"velocity_vph", "velocity_phv", "phoneme_probe", "phoneme_decoder", "temporal_pool"}

    def __init__(self, adapters: dict, cfg: CAMELSConfig, device: torch.device | str):
        self.cfg = cfg
        self.device = device

        # Deep-copy enabled adapters; freeze key encoder params
        self.key_adapters: dict = {}
        for k, v in adapters.items():
            if k not in self._EXCLUDE:
                m = copy.deepcopy(v).to(device)
                for p in m.parameters():
                    p.requires_grad_(False)
                self.key_adapters[k] = m

        # One queue per enabled modality
        d = cfg.latent.d_latent
        K = cfg.moco.queue_size
        self.queues: dict[str, MoCoQueue] = {
            mod: MoCoQueue(d, K, device) for mod in cfg.enabled_modalities()
        }

    @torch.no_grad()
    def ema_update(self, adapters: dict, m: float) -> None:
        """EMA: p_k = m * p_k + (1 - m) * p_q for each key adapter."""
        for k, key_mod in self.key_adapters.items():
            if k not in adapters:
                continue
            for p_k, p_q in zip(key_mod.parameters(), adapters[k].parameters()):
                p_k.mul_(m).add_(p_q.detach().to(self.device), alpha=1.0 - m)

    @torch.no_grad()
    def encode_keys(
        self,
        batch,
        cfg: CAMELSConfig,
        device: str | torch.device,
        stage: str,
    ) -> dict[str, torch.Tensor]:
        """Run frozen key adapters to produce key embeddings."""
        v_raw, ph_raw, _ph_labels, ph_mask, p_raw = batch
        v_raw = v_raw.to(device)
        ph_raw = ph_raw.to(device)
        ph_mask = ph_mask.to(device)
        p_raw = p_raw.to(device)

        z_k_dict: dict[str, torch.Tensor] = {}

        if cfg.modality.video_enabled and "video_adapter" in self.key_adapters:
            z_k_dict["video"] = self.key_adapters["video_adapter"].embed(v_raw)

        if cfg.modality.phoneme_enabled and "phoneme_adapter" in self.key_adapters:
            z_ph_seq = self.key_adapters["phoneme_adapter"](ph_raw)
            if "phoneme_attn_pool" in self.key_adapters:
                z_k_dict["phoneme"] = self.key_adapters["phoneme_attn_pool"](
                    z_ph_seq, ph_mask
                )

        if cfg.modality.prosody_enabled and "prosody_adapter" in self.key_adapters:
            z_k_dict["prosody"] = self.key_adapters["prosody_adapter"].embed(p_raw)

        return z_k_dict

    @torch.no_grad()
    def enqueue(self, z_k_dict: dict[str, torch.Tensor]) -> None:
        """Enqueue normalized key embeddings into per-modality queues."""
        for mod, z_k in z_k_dict.items():
            if mod in self.queues:
                self.queues[mod].enqueue_dequeue(F.normalize(z_k, dim=-1))
