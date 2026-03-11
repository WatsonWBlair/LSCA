# encoding/adapters/phoneme.py
# Phoneme-specific adapter components:
#   PhonemeAdapter   — thin linear projection (NOT AVAE)
#   PhonemeAttnPool  — collapse per-phoneme sequence to single vector
#   PhonemeProbeHead — training-only classification probe (discarded at inference)

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhonemeAdapter(nn.Module):
    """
    Thin linear projection applied per-phoneme position.
    wav2vec2 hidden states are already rich 768-D representations —
    a full AVAE is unnecessary. Phoneme verification uses the probe head instead.
    """

    def __init__(self, d_in: int, d_latent: int):
        super().__init__()
        self.d_in = d_in
        self.d_latent = d_latent
        self.proj = nn.Linear(d_in, d_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, MAX_PHONES, d_in) or (MAX_PHONES, d_in)
        Returns: same shape with last dim projected to d_latent.
        """
        return self.proj(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward — consistent interface with AVAEAdapter."""
        return self.forward(x)


class PhonemeAttnPool(nn.Module):
    """
    Collapse (B, MAX_PHONES, d) to (B, d) via masked softmax attention.
    Used to produce a single-vector phoneme representation for InfoNCE pairing.
    """

    def __init__(self, d: int):
        super().__init__()
        self.scorer = nn.Linear(d, 1)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        H:    (B, MAX_PHONES, d)
        mask: (B, MAX_PHONES) — 1 = real phoneme, 0 = padding
        Returns: (B, d)
        """
        scores = self.scorer(H).squeeze(-1)             # (B, MAX_PHONES)
        # Mask out padding positions with -inf before softmax
        scores = scores.masked_fill(~mask.bool(), float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, MAX_PHONES, 1)
        return (weights * H).sum(dim=1)                     # (B, d)


class PhonemeProbeHead(nn.Module):
    """
    Training-only classification probe for phoneme identity verification.
    Predicts IPA phoneme label per position from z_ph.
    Cross-entropy on unmasked positions confirms z_ph preserves
    phoneme-discriminative information.
    Discarded at inference — NOT loaded into the live pipeline.

    Ref: v8.1 plan Section 9.1.7
    """

    def __init__(self, d: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, z_ph_seq: torch.Tensor) -> torch.Tensor:
        """
        z_ph_seq: (B, MAX_PHONES, d)
        Returns:  (B, MAX_PHONES, n_classes)
        """
        return self.mlp(z_ph_seq)
