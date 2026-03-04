# training/losses.py
# All loss functions for the CAMELS training protocol.
#   InfoNCE / NT-Xent  (SimCLR, Chen et al. 2020)
#   AVAE loss          (AvaeFlow, Li et al. 2025)
#   Bidirectional FM   (AvaeFlow + FM-Refiner, 2025/26)

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoding.utils.config import D_LATENT


# ── InfoNCE (NT-Xent) ─────────────────────────────────────────────────────────

def info_nce_loss(
    z_a:         torch.Tensor,   # (B, D) — first modality embeddings
    z_b:         torch.Tensor,   # (B, D) — second modality embeddings
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss (NT-Xent variant from SimCLR).
    Positives: same row index (same chunk, different modalities).
    Negatives: all off-diagonal entries in the batch.

    Temperature: 0.07 (as in SimCLR).
    Batch size should be >= 64 for reliable negative mining.

    Returns: scalar loss.
    """
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)

    B      = z_a.shape[0]
    logits = torch.matmul(z_a, z_b.T) / temperature   # (B, B)
    labels = torch.arange(B, device=z_a.device)

    loss_ab = F.cross_entropy(logits,   labels)
    loss_ba = F.cross_entropy(logits.T, labels)
    return (loss_ab + loss_ba) / 2.0


def all_pairs_nce(
    z_v: torch.Tensor,   # (B, 1024)
    z_a: torch.Tensor,   # (B, 1024)
    z_p: torch.Tensor,   # (B, 1024)
    z_t: torch.Tensor,   # (B, 1024)
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Sum of all 6 InfoNCE pairs across 4 modalities.
    Pairs: v-a, v-p, v-t, a-p, a-t, p-t.
    """
    return (
        info_nce_loss(z_v, z_a, temperature)   # video   <-> audio
      + info_nce_loss(z_v, z_p, temperature)   # video   <-> prosody
      + info_nce_loss(z_v, z_t, temperature)   # video   <-> text
      + info_nce_loss(z_a, z_p, temperature)   # audio   <-> prosody
      + info_nce_loss(z_a, z_t, temperature)   # audio   <-> text
      + info_nce_loss(z_p, z_t, temperature)   # prosody <-> text
    )


# ── AVAE loss ─────────────────────────────────────────────────────────────────

def avae_loss(
    x:          torch.Tensor,   # (B, d_in) — original raw feature
    x_hat:      torch.Tensor,   # (B, d_in) — AVAE decoder reconstruction
    mu:         torch.Tensor,   # (B, d_latent)
    logvar:     torch.Tensor,   # (B, d_latent)
    z:          torch.Tensor,   # (B, d_latent) — reparameterized sample
    z_prime:    torch.Tensor,   # (B, d_latent) — re-encoded sample
    kl_weight:  float = 1e-4,
) -> torch.Tensor:
    """
    AVAE loss = reconstruction + KL + self-consistency.

    recon   : MSE(x_hat, x)           — decoder reconstruction quality
    kl      : KL(q(z|x) || N(0,I))    — latent regularization (small weight)
    consist : MSE(z', z.detach())     — encoder-decoder-encoder loop consistency

    kl_weight=1e-4 is deliberately small to prevent over-regularization and
    posterior collapse, especially for the low-dimensional prosody adapter.
    """
    recon   = F.mse_loss(x_hat, x)
    kl      = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    consist = F.mse_loss(z_prime, z.detach())
    return recon + kl_weight * kl + consist


def all_avae_loss(
    v_raw,  xh_v, mu_v, lv_v, z_v, zp_v,
    a_raw,  xh_a, mu_a, lv_a, z_a, zp_a,
    p_raw,  xh_p, mu_p, lv_p, z_p, zp_p,
    t_raw,  xh_t, mu_t, lv_t, z_t, zp_t,
    kl_weight: float = 1e-4,
) -> torch.Tensor:
    """Sum of AVAE losses across all 4 modalities."""
    return (
        avae_loss(v_raw, xh_v, mu_v, lv_v, z_v, zp_v, kl_weight)
      + avae_loss(a_raw, xh_a, mu_a, lv_a, z_a, zp_a, kl_weight)
      + avae_loss(p_raw, xh_p, mu_p, lv_p, z_p, zp_p, kl_weight)
      + avae_loss(t_raw, xh_t, mu_t, lv_t, z_t, zp_t, kl_weight)
    )


# ── Flow matching loss ─────────────────────────────────────────────────────────

def flow_matching_loss(
    z_src:        torch.Tensor,   # (B, D) — source latent (detached)
    z_tgt:        torch.Tensor,   # (B, D) — target latent (detached)
    velocity_net,                  # VelocityNet
    sigma_min:    float = 1e-4,
) -> torch.Tensor:
    """
    Optimal Transport conditional flow matching loss (constant velocity).
    Interpolates z_t = (1 - (1-σ)t)·z_src + t·z_tgt, t ~ U[0,1].
    Target velocity: v* = z_tgt - (1-σ)·z_src.

    CRITICAL: z_src and z_tgt must be detached before this call.
    Only VelocityNet receives FM gradients — adapters are not updated by FM.

    Ref: Lipman et al. 2022; AvaeFlow + FM-Refiner 2025/26.
    """
    B   = z_src.shape[0]
    t   = torch.rand(B, 1, device=z_src.device)

    # Interpolated latent at time t
    z_t = (1.0 - (1.0 - sigma_min) * t) * z_src + t * z_tgt

    # Constant velocity target
    v_target = z_tgt - (1.0 - sigma_min) * z_src

    v_pred = velocity_net(z_t, t)
    return F.mse_loss(v_pred, v_target)


def bidirectional_fm_loss(
    z_v:        torch.Tensor,
    z_a:        torch.Tensor,
    velocity_va,            # VelocityNet: video → audio
    velocity_av,            # VelocityNet: audio → video
    sigma_min:  float = 1e-4,
) -> torch.Tensor:
    """
    Bidirectional FM between video and audio latent spaces.
    Adapters MUST be detached before this call (done here explicitly).
    Only VelocityNets receive gradients from this loss.
    """
    l_va = flow_matching_loss(z_v.detach(), z_a.detach(), velocity_va, sigma_min)
    l_av = flow_matching_loss(z_a.detach(), z_v.detach(), velocity_av, sigma_min)
    return l_va + l_av


# ── Per-pair InfoNCE monitoring ───────────────────────────────────────────────

def monitor_nce_pairs(
    z_v, z_a, z_p, z_t, temperature: float = 0.07
) -> dict:
    """
    Compute InfoNCE loss for each of the 6 pairs separately.
    Used during evaluation to identify which pairs are failing to align.
    Returns dict of {pair_name: loss_value}.
    """
    pairs = [
        ("v-a", z_v, z_a),
        ("v-p", z_v, z_p),
        ("v-t", z_v, z_t),
        ("a-p", z_a, z_p),
        ("a-t", z_a, z_t),
        ("p-t", z_p, z_t),
    ]
    return {
        name: info_nce_loss(za, zb, temperature).item()
        for name, za, zb in pairs
    }
