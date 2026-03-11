# encoding/adapters/base.py
# Core adapter components: MLP, AVAEAdapter, TemporalAttentionPool.
# Ref: AvaeFlow, Li et al. 2025

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-layer perceptron with optional LayerNorm.
    The last linear layer has NO activation or norm (projection head).
    """

    def __init__(self, dims: list[int], norm: str | None = None):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if norm == "layernorm":
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AVAEAdapter(nn.Module):
    """
    Attentive Variational AutoEncoder adapter.
    Projects one modality's raw features into the shared latent space.

    At inference: embed() returns mu directly (no sampling, no decoder).
    At training:  forward() returns (mu, logvar, z, x_hat, z_prime).

    Decoder reconstructs back to the raw backbone output domain:
      VideoAdapter   -> MARLIN embedding space   (d_video,)
      ProsodyAdapter -> raw librosa feature space (d_prosody,)
    """

    def __init__(self, d_in: int, d_latent: int, hidden: int):
        super().__init__()
        self.d_in = d_in
        self.d_latent = d_latent
        self.input_norm = nn.LayerNorm(d_in)
        self.encoder = MLP([d_in, hidden, hidden], norm="layernorm")
        self.mu_head = nn.Linear(hidden, d_latent)
        self.logvar_head = nn.Linear(hidden, d_latent)
        self.decoder = MLP([d_latent, hidden, d_in], norm="layernorm")
        self.reencoder = MLP([d_in, hidden, d_latent])

    def encode(self, x: torch.Tensor):
        h = self.encoder(self.input_norm(x))
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(mu)

    def forward(self, x: torch.Tensor):
        """Training forward: returns (mu, logvar, z, x_hat, z_prime)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        z_prime = self.reencoder(x_hat)
        return mu, logvar, z, x_hat, z_prime

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Inference only — returns mu, no sampling, no decoder."""
        mu, _ = self.encode(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent back to input domain (validation / sanity check)."""
        return self.decoder(z)


class TemporalAttentionPool(nn.Module):
    """
    Collapses (W, d) MARLIN window embeddings to (d,) per chunk
    via learned softmax attention over time.
    """

    def __init__(self, d: int):
        super().__init__()
        self.scorer = nn.Linear(d, 1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (W, d) — one embedding per MARLIN window
        weights = F.softmax(self.scorer(H), dim=0)  # (W, 1)
        return (weights * H).sum(dim=0)              # (d,)
