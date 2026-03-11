# encoding/adapters/velocity.py
# Flow-matching velocity network for bidirectional transport (video <-> phoneme).
# Ref: AvaeFlow + FM-Refiner, 2025/26

import torch
import torch.nn as nn

from encoding.adapters.base import MLP


class VelocityNet(nn.Module):
    """
    Predicts the OT-path transport velocity from z_src to z_tgt at time t.
    Input:  concatenated [z_t, t] with shape (B, d+1)
    Output: velocity vector (B, d)
    """

    def __init__(self, d: int):
        super().__init__()
        self.net = MLP([d + 1, d, d, d], norm="layernorm")

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # z_t: (B, d), t: (B, 1)
        return self.net(torch.cat([z_t, t], dim=-1))
