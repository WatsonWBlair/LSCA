# encoding/training/losses.py
# All loss functions for the CAMELS v8.1 training protocol.
#   InfoNCE / NT-Xent          (SimCLR, Chen et al. 2020)
#   Capacity-controlled AVAE   (AvaeFlow + Burgess et al. 2018)
#   Cross-modal orthogonality  (Barlow Twins, Zbontar et al. 2021)
#   Variance / Covariance reg  (VICReg, Barbe et al. 2022)
#   Bidirectional FM           (AvaeFlow + FM-Refiner 2025/26)
#   Phoneme probe loss         (v8.1 auxiliary verification)

from __future__ import annotations

from itertools import combinations

import torch
import torch.nn.functional as F


# ── InfoNCE (NT-Xent) ──────────────────────────────────────────────────────────

def info_nce_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss (NT-Xent).
    Positives: same row (same chunk, different modalities).
    Negatives: all off-diagonal.
    """
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    B = z_a.shape[0]
    logits = torch.matmul(z_a, z_b.T) / temperature
    labels = torch.arange(B, device=z_a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


def all_pairs_nce(
    z_dict: dict[str, torch.Tensor],
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Sum InfoNCE over all pairs of enabled modalities.
    z_dict maps modality name -> (B, d_latent) tensor.

    Returns: (total_loss, per_pair_losses_dict)
    """
    names = sorted(z_dict.keys())
    total = torch.tensor(0.0, device=next(iter(z_dict.values())).device)
    per_pair = {}
    for n1, n2 in combinations(names, 2):
        loss = info_nce_loss(z_dict[n1], z_dict[n2], temperature)
        per_pair[f"nce_{n1}_{n2}"] = loss
        total = total + loss
    return total, per_pair


# ── AVAE loss with capacity-controlled KL ───────────────────────────────────────

def avae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z: torch.Tensor,
    z_prime: torch.Tensor,
    capacity: float = 0.0,
    beta_cap: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    AVAE loss with capacity-controlled KL (Burgess et al. 2018).
    Returns dict of individual loss terms for per-modality tracking.
    """
    recon = F.mse_loss(x_hat, x)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    consist = F.mse_loss(z_prime, z.detach())
    kl_cap = beta_cap * torch.abs(kl - capacity)
    total = recon + kl_cap + consist
    return {
        "recon": recon,
        "kl": kl,
        "kl_cap": kl_cap,
        "consist": consist,
        "total": total,
    }


def get_capacity(
    epoch: int,
    stage_start: int,
    stage_end: int,
    c_max: float,
) -> float:
    """Linear capacity ramp from 0 to c_max over a training stage."""
    if epoch < stage_start:
        return 0.0
    progress = min(1.0, (epoch - stage_start) / max(1, stage_end - stage_start))
    return c_max * progress


# ── Cross-modal orthogonality (L_orth) ──────────────────────────────────────────

def cross_modal_orth_loss(z_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Penalizes cross-modal correlations (Barlow Twins style).
    z_list: list of (B, d) tensors for each modality.
    """
    loss = torch.tensor(0.0, device=z_list[0].device)
    n_pairs = 0
    for i in range(len(z_list)):
        for j in range(i + 1, len(z_list)):
            zi = z_list[i] - z_list[i].mean(dim=0)
            zj = z_list[j] - z_list[j].mean(dim=0)
            zi = F.normalize(zi, dim=0)
            zj = F.normalize(zj, dim=0)
            C_ij = torch.mm(zi.T, zj) / zi.shape[0]
            loss = loss + (C_ij ** 2).sum()
            n_pairs += 1
    return loss / max(n_pairs, 1)


# ── Within-modality variance regularizer (L_var) ───────────────────────────────

def variance_loss(
    z_list: list[torch.Tensor],
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg variance regularizer — prevents dimensional collapse."""
    loss = torch.tensor(0.0, device=z_list[0].device)
    for z in z_list:
        std = torch.sqrt(z.var(dim=0) + eps)
        loss = loss + torch.mean(F.relu(gamma - std))
    return loss / len(z_list)


# ── Within-modality covariance regularizer (L_cov) ─────────────────────────────

def covariance_loss(z_list: list[torch.Tensor]) -> torch.Tensor:
    """VICReg covariance regularizer — decorrelates dimensions within each modality."""
    loss = torch.tensor(0.0, device=z_list[0].device)
    for z in z_list:
        B, D = z.shape
        z_centered = z - z.mean(dim=0)
        cov = torch.mm(z_centered.T, z_centered) / max(B - 1, 1)
        off_diag = cov - torch.diag(cov.diagonal())
        loss = loss + (off_diag ** 2).sum() / D
    return loss / len(z_list)


# ── Flow matching loss ──────────────────────────────────────────────────────────

def flow_matching_loss(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    velocity_net,
    sigma_min: float = 1e-4,
) -> torch.Tensor:
    """
    OT conditional flow matching (constant velocity).
    CRITICAL: z_src and z_tgt must be detached before this call.
    """
    B = z_src.shape[0]
    t = torch.rand(B, 1, device=z_src.device)
    z_t = (1.0 - (1.0 - sigma_min) * t) * z_src + t * z_tgt
    v_target = z_tgt - (1.0 - sigma_min) * z_src
    v_pred = velocity_net(z_t, t)
    return F.mse_loss(v_pred, v_target)


def bidirectional_fm_loss(
    z_v: torch.Tensor,
    z_ph: torch.Tensor,
    velocity_vph,
    velocity_phv,
    sigma_min: float = 1e-4,
) -> torch.Tensor:
    """
    Bidirectional FM between video and phoneme latent spaces.
    Adapters DETACHED — only VelocityNets receive gradients.
    """
    l_vph = flow_matching_loss(z_v.detach(), z_ph.detach(), velocity_vph, sigma_min)
    l_phv = flow_matching_loss(z_ph.detach(), z_v.detach(), velocity_phv, sigma_min)
    return l_vph + l_phv


# ── Phoneme probe loss ──────────────────────────────────────────────────────────

def phoneme_probe_loss(
    z_ph_padded: torch.Tensor,
    phoneme_labels: torch.Tensor,
    mask: torch.Tensor,
    probe_head,
) -> torch.Tensor:
    """
    Cross-entropy on unmasked phoneme positions.
    z_ph_padded:    (B, MAX_PHONES, d)
    phoneme_labels: (B, MAX_PHONES) — ground truth IDs from CTC
    mask:           (B, MAX_PHONES) — 1 = real, 0 = padding
    """
    logits = probe_head(z_ph_padded)           # (B, MAX_PHONES, n_classes)
    logits_flat = logits[mask.bool()]           # (N_real, n_classes)
    labels_flat = phoneme_labels[mask.bool()]   # (N_real,)
    if labels_flat.numel() == 0:
        return torch.tensor(0.0, device=z_ph_padded.device)
    return F.cross_entropy(logits_flat, labels_flat)


# ── Per-pair monitoring ─────────────────────────────────────────────────────────

def monitor_nce_pairs(
    z_dict: dict[str, torch.Tensor],
    temperature: float = 0.07,
) -> dict[str, float]:
    """Compute InfoNCE for each pair separately (diagnostics)."""
    names = sorted(z_dict.keys())
    result = {}
    with torch.no_grad():
        for n1, n2 in combinations(names, 2):
            result[f"nce_{n1}_{n2}"] = info_nce_loss(z_dict[n1], z_dict[n2], temperature).item()
    return result
