# pipeline/adapters.py
# Trainable components: MLP helper, AVAEAdapter (all 4 modalities), VelocityNet.
# Ref: AvaeFlow, Li et al. 2025

import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.config import (
    D_VIDEO, D_AUDIO, D_PROSODY, D_LATENT,
    HIDDEN_HIGH, HIDDEN_PROS,
)


# ── MLP helper ──────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Multi-layer perceptron.
    dims: list of layer sizes, e.g. [768, 256, 256]
    norm: 'layernorm' inserts LayerNorm + GELU between every pair of layers.
          None inserts only GELU (no norm).
    The last linear layer has NO activation or norm — it's a projection head.
    """
    def __init__(self, dims: list, norm: str = None):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # Add activation + optional norm after every layer except the last
            if i < len(dims) - 2:
                if norm == "layernorm":
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── AVAE Adapter ────────────────────────────────────────────────────────────

class AVAEAdapter(nn.Module):
    """
    Attentive Variational AutoEncoder adapter.
    Projects one modality's raw feature into the shared 1024-D latent space.

    Architecture (per modality):
      encoder    : MLP(d_in → hidden → hidden)  with LayerNorm
      mu_head    : Linear(hidden → d_latent)
      logvar_head: Linear(hidden → d_latent)
      decoder    : MLP(d_latent → hidden → d_in)  with LayerNorm  [training only]
      reencoder  : MLP(d_in → hidden → d_latent)                  [training only]

    At inference: embed() returns mu directly (no sampling, no decoder).
    At training:  forward() returns (mu, logvar, z, x_hat, z_prime).

    Decoder output domain (what x_hat reconstructs to):
      VideoAdapter   → MARLIN embedding space   (768,)
      AudioAdapter   → wav2vec2 embedding space (d_audio,)
      ProsodyAdapter → normalized prosody       (22,)
      TextAdapter    → SONAR embedding space    (1024,)
    """

    def __init__(self, d_in: int, d_latent: int = D_LATENT, hidden: int = HIDDEN_HIGH):
        super().__init__()
        self.encoder      = MLP([d_in,     hidden, hidden], norm="layernorm")
        self.mu_head      = nn.Linear(hidden, d_latent)
        self.logvar_head  = nn.Linear(hidden, d_latent)
        self.decoder      = MLP([d_latent, hidden, d_in],  norm="layernorm")
        self.reencoder    = MLP([d_in,     hidden, d_latent])

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(mu)

    def forward(self, x: torch.Tensor):
        """
        Full forward pass for training.
        Returns: (mu, logvar, z, x_hat, z_prime)
          mu      : mean of approximate posterior  (B, d_latent)
          logvar  : log-variance                   (B, d_latent)
          z       : reparameterized sample          (B, d_latent)
          x_hat   : reconstructed input             (B, d_in)
          z_prime : re-encoded z — consistency check (B, d_latent)
        """
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_hat      = self.decoder(z)
        z_prime    = self.reencoder(x_hat)
        return mu, logvar, z, x_hat, z_prime

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Inference only — returns mu, no sampling, no decoder."""
        mu, _ = self.encode(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector back to input domain (sanity check / validation)."""
        return self.decoder(z)


# ── VelocityNet ─────────────────────────────────────────────────────────────

class VelocityNet(nn.Module):
    """
    Flow-matching velocity network.
    Predicts the transport velocity from z_src to z_tgt at interpolation time t.
    Input: (B, d+1)  — concatenated [z_t, t]
    Output: (B, d)   — velocity vector
    Ref: AvaeFlow + FM-Refiner, 2025/26
    """

    def __init__(self, d: int = D_LATENT):
        super().__init__()
        self.net = MLP([d + 1, 1024, 1024, d], norm="layernorm")

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # z_t: (B, d), t: (B, 1)
        return self.net(torch.cat([z_t, t], dim=-1))


# ── TemporalAttentionPool ────────────────────────────────────────────────────

class TemporalAttentionPool(nn.Module):
    """
    Collapses (W, 768) MARLIN window embeddings → (768,) per chunk via
    a learned softmax attention over time.
    ~769 trainable parameters.
    """

    def __init__(self, d: int = D_VIDEO):
        super().__init__()
        self.scorer = nn.Linear(d, 1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (W, d)
        weights = F.softmax(self.scorer(H), dim=0)   # (W, 1)
        return (weights * H).sum(dim=0)               # (d,)


# ── Instantiation helpers ────────────────────────────────────────────────────

def build_adapters(d_audio: int = D_AUDIO) -> dict:
    """
    Instantiate all four AVAE adapters, both VelocityNets, and TemporalAttentionPool.
    Returns a dict ready to be passed to optimizer and training loops.

    d_audio: hidden dim of wav2vec2 (768 for base, 1024 for large).
    """
    return {
        "temporal_pool":   TemporalAttentionPool(d=D_VIDEO),
        "video_adapter":   AVAEAdapter(d_in=D_VIDEO,   d_latent=D_LATENT, hidden=HIDDEN_HIGH),
        "audio_adapter":   AVAEAdapter(d_in=d_audio,   d_latent=D_LATENT, hidden=HIDDEN_HIGH),
        "prosody_adapter": AVAEAdapter(d_in=D_PROSODY, d_latent=D_LATENT, hidden=HIDDEN_PROS),
        "text_adapter":    AVAEAdapter(d_in=D_LATENT,  d_latent=D_LATENT, hidden=HIDDEN_HIGH),
        "velocity_va":     VelocityNet(d=D_LATENT),  # video → audio
        "velocity_av":     VelocityNet(d=D_LATENT),  # audio → video
    }


def trainable_params(adapters: dict) -> list:
    """Return all trainable parameters across all adapter modules."""
    params = []
    for module in adapters.values():
        params += list(module.parameters())
    return params


def save_adapters(adapters: dict, path: str):
    """Save adapter state dicts to a .pt checkpoint."""
    torch.save({k: v.state_dict() for k, v in adapters.items()}, path)


def load_adapters(path: str, d_audio: int = D_AUDIO, device: str = "cpu") -> dict:
    """Load adapter state dicts from a .pt checkpoint."""
    adapters = build_adapters(d_audio=d_audio)
    state    = torch.load(path, map_location=device)
    for k, module in adapters.items():
        module.load_state_dict(state[k])
        module.to(device)
    return adapters
