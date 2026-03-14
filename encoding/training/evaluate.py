# encoding/training/evaluate.py
# Evaluation metrics for 3-modality CAMELS v8.1 training.
# Adapted from v7: 3 modalities, 3 pairs, 6 retrieval directions.

from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from encoding.config import CAMELSConfig

logger = logging.getLogger(__name__)


def eval_intra_chunk_alignment(z_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """
    Cosine similarity between same-chunk embeddings (positive pairs).
    Target: > 0.5 after Stage A.
    """
    names = sorted(z_dict.keys())
    results = {}
    for n1, n2 in combinations(names, 2):
        sim = F.cosine_similarity(
            F.normalize(z_dict[n1], dim=-1),
            F.normalize(z_dict[n2], dim=-1),
        )
        results[f"intra_alignment_{n1}_{n2}"] = sim.mean().item()
    results["intra_alignment_mean"] = float(np.mean(list(results.values())))
    return results


def eval_inter_chunk_separation(z_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """
    Mean cosine sim between same modality, different chunks. Target: < 0.2.
    """
    results = {}
    for name, z in z_dict.items():
        z_norm = F.normalize(z, dim=-1)
        sim = torch.matmul(z_norm, z_norm.T)
        mask = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
        results[f"inter_sep_{name}"] = sim[mask].mean().item()
    return results


def eval_retrieval(z_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Cross-modal R@1 and R@5 retrieval in all directions."""
    names = sorted(z_dict.keys())
    results = {}
    for src_name in names:
        for tgt_name in names:
            if src_name == tgt_name:
                continue
            src = z_dict[src_name]
            tgt = z_dict[tgt_name]
            sims = torch.matmul(F.normalize(src, dim=-1), F.normalize(tgt, dim=-1).T)
            ranked = sims.argsort(dim=1, descending=True)
            labels = torch.arange(len(src), device=src.device)
            r1 = (ranked[:, 0] == labels).float().mean().item()
            r5 = (ranked[:, :5] == labels.unsqueeze(1)).any(dim=1).float().mean().item()
            results[f"R@1_{src_name}->{tgt_name}"] = r1
            results[f"R@5_{src_name}->{tgt_name}"] = r5
    return results


def eval_uniformity(z: torch.Tensor, name: str = "") -> float:
    """Wang & Isola uniformity metric. Lower = more uniform on unit sphere."""
    z_norm = F.normalize(z, dim=-1)
    sq_dist = torch.pdist(z_norm, p=2).pow(2)
    return sq_dist.mul(-2).exp().mean().log().item()


def eval_uniformity_all(z_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    return {f"uniformity_{name}": eval_uniformity(z) for name, z in z_dict.items()}


def eval_cosine_margin(z_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """pos_mean - neg_mean per pair. Target: > 0.3."""
    names = sorted(z_dict.keys())
    results = {}
    for n1, n2 in combinations(names, 2):
        za_n = F.normalize(z_dict[n1], dim=-1)
        zb_n = F.normalize(z_dict[n2], dim=-1)
        sim = torch.matmul(za_n, zb_n.T)
        N = sim.shape[0]
        mask = torch.eye(N, dtype=torch.bool, device=za_n.device)
        pos_mean = sim[mask].mean().item()
        neg_mean = sim[~mask].mean().item()
        results[f"margin_{n1}_{n2}"] = pos_mean - neg_mean
    return results


def eval_roc_auc(z_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Binary ROC-AUC: same-chunk vs cross-chunk. Target: > 0.85."""
    from sklearn.metrics import roc_auc_score
    names = sorted(z_dict.keys())
    results = {}
    for n1, n2 in combinations(names, 2):
        za_n = F.normalize(z_dict[n1], dim=-1)
        zb_n = F.normalize(z_dict[n2], dim=-1)
        sim = torch.matmul(za_n, zb_n.T)
        N = sim.shape[0]
        mask = torch.eye(N, dtype=torch.bool, device=za_n.device)
        scores = sim.cpu().numpy().flatten()
        labels = mask.cpu().numpy().flatten().astype(int)
        try:
            results[f"roc_auc_{n1}_{n2}"] = roc_auc_score(labels, scores)
        except Exception:
            results[f"roc_auc_{n1}_{n2}"] = 0.5
    return results


def eval_reconstruction_mse(
    val_loader,
    adapters: dict,
    cfg: CAMELSConfig,
    device: str = "cpu",
) -> dict[str, float]:
    """Decode z back to input domain. Video < 0.10, prosody < 0.05."""
    mse = {"video": [], "prosody": []}
    for v_raw, ph_raw, ph_labels, ph_mask, p_raw in val_loader:
        v_raw = v_raw.to(device, non_blocking=True)
        p_raw = p_raw.to(device, non_blocking=True)
        with torch.no_grad():
            if "video_adapter" in adapters:
                mu_v, lv_v = adapters["video_adapter"].encode(v_raw)
                z_v = adapters["video_adapter"].reparameterize(mu_v, lv_v)
                xh_v = adapters["video_adapter"].decode(z_v)
                mse["video"].append(F.mse_loss(xh_v, v_raw).item())
            if "prosody_adapter" in adapters:
                mu_p, lv_p = adapters["prosody_adapter"].encode(p_raw)
                z_p = adapters["prosody_adapter"].reparameterize(mu_p, lv_p)
                xh_p = adapters["prosody_adapter"].decode(z_p)
                mse["prosody"].append(F.mse_loss(xh_p, p_raw).item())

    results = {}
    for name, vals in mse.items():
        if vals:
            results[f"recon_mse_{name}"] = float(np.mean(vals))
    return results


def eval_kl_per_modality(
    val_loader,
    adapters: dict,
    device: str = "cpu",
) -> dict[str, float]:
    """Monitor KL divergence per AVAE modality."""
    kl = {"video": [], "prosody": []}
    for v_raw, ph_raw, ph_labels, ph_mask, p_raw in val_loader:
        v_raw = v_raw.to(device, non_blocking=True)
        p_raw = p_raw.to(device, non_blocking=True)
        with torch.no_grad():
            for raw, key, name in [(v_raw, "video_adapter", "video"), (p_raw, "prosody_adapter", "prosody")]:
                if key in adapters:
                    mu, lv = adapters[key].encode(raw)
                    kl_val = -0.5 * torch.mean(1.0 + lv - mu.pow(2) - lv.exp())
                    kl[name].append(kl_val.item())
    return {f"kl_{name}": float(np.mean(vals)) for name, vals in kl.items() if vals}


def eval_modality_recovery(z_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Train a linear probe to predict modality identity from embeddings.
    In a perfectly disentangled space no dimension should encode modality — target ≈ 33%.
    Higher accuracy means more modality-identity information leaking into the shared space.
    """
    from sklearn.linear_model import LogisticRegression
    import warnings

    names = sorted(z_dict.keys())
    if len(names) < 2:
        return {}

    parts = []
    labels = []
    for idx, name in enumerate(names):
        z = z_dict[name].detach().cpu().numpy()
        parts.append(z)
        labels.extend([idx] * len(z))

    X = np.concatenate(parts, axis=0)
    y = np.array(labels)
    chance = 1.0 / len(names)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs", multi_class="auto")
        clf.fit(X, y)
        acc = clf.score(X, y)

    return {
        "modality_recovery_acc": float(acc),
        "modality_recovery_above_chance": float(acc - chance),
    }


def eval_cross_modal_mi_proxy(z_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Normalized cross-correlation proxy for mutual information between modality pairs.
    Uses ||C_ij||_F / (d * sqrt(B)) so values are comparable across batch sizes.
    Target: mean < 0.1 after Stage B. Lower = more disentangled.
    """
    names = sorted(z_dict.keys())
    results = {}
    for n1, n2 in combinations(names, 2):
        z1 = z_dict[n1]
        z2 = z_dict[n2]
        B, d = z1.shape[0], z1.shape[1]
        z1c = F.normalize(z1 - z1.mean(0), dim=0)
        z2c = F.normalize(z2 - z2.mean(0), dim=0)
        C = torch.mm(z1c.T, z2c) / B
        results[f"mi_proxy_{n1}_{n2}"] = (C.norm() / (d * (B ** 0.5))).item()
    results["mi_proxy_mean"] = float(np.mean([v for k, v in results.items() if k != "mi_proxy_mean"]))
    return results


def eval_dimension_utilization(
    z_dict: dict[str, torch.Tensor],
    d_latent: int,
    threshold: float = 0.01,
) -> dict[str, float]:
    """Active dimensions per modality. Target: > d_latent * 0.75 (75% utilization)."""
    results = {}
    for name, z in z_dict.items():
        var = z.var(dim=0)
        active = (var > threshold).sum().item()
        results[f"active_dims_{name}"] = active
        results[f"dim_util_{name}"] = active / d_latent
    results["dim_util_mean"] = float(np.mean([v for k, v in results.items() if "dim_util_" in k and "mean" not in k]))
    return results


def eval_phoneme_probe_accuracy(
    val_loader,
    adapters: dict,
    device: str = "cpu",
) -> dict[str, float]:
    """Classification accuracy of PhonemeProbeHead on val set. Target: > 85%."""
    if "phoneme_probe" not in adapters or "phoneme_adapter" not in adapters:
        return {}

    correct = 0
    total = 0
    for v_raw, ph_raw, ph_labels, ph_mask, p_raw in val_loader:
        ph_raw = ph_raw.to(device, non_blocking=True)
        ph_labels = ph_labels.to(device, non_blocking=True)
        ph_mask = ph_mask.to(device, non_blocking=True)
        with torch.no_grad():
            z_ph = adapters["phoneme_adapter"](ph_raw)
            logits = adapters["phoneme_probe"](z_ph)
            preds = logits.argmax(dim=-1)
            correct += (preds[ph_mask.bool()] == ph_labels[ph_mask.bool()]).sum().item()
            total += ph_mask.bool().sum().item()

    acc = correct / max(total, 1)
    return {"phoneme_probe_accuracy": acc}


def run_evaluation(
    val_loader,
    adapters: dict,
    cfg: CAMELSConfig,
    device: str = "cpu",
    stage: str = "A",
) -> dict[str, float]:
    """Run the full evaluation suite for a given training stage."""
    all_v, all_ph_pooled, all_p = [], [], []

    for v_raw, ph_raw, ph_labels, ph_mask, p_raw in val_loader:
        v_raw = v_raw.to(device, non_blocking=True)
        ph_raw = ph_raw.to(device, non_blocking=True)
        ph_mask = ph_mask.to(device, non_blocking=True)
        p_raw = p_raw.to(device, non_blocking=True)

        with torch.no_grad():
            z_v = adapters["video_adapter"].embed(v_raw)
            if cfg.modality.phoneme_adapter_type == "avae":
                mask_f = ph_mask.float().unsqueeze(-1)
                ph_pooled = (ph_raw * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
                z_ph_pooled = adapters["phoneme_adapter"].embed(ph_pooled)
            else:
                z_ph_seq = adapters["phoneme_adapter"](ph_raw)
                z_ph_pooled = adapters["phoneme_attn_pool"](z_ph_seq, ph_mask)
            z_p = adapters["prosody_adapter"].embed(p_raw)

        all_v.append(z_v.cpu())
        all_ph_pooled.append(z_ph_pooled.cpu())
        all_p.append(z_p.cpu())

    z_v = torch.cat(all_v)
    z_ph = torch.cat(all_ph_pooled)
    z_p = torch.cat(all_p)

    z_dict = {"phoneme": z_ph, "prosody": z_p, "video": z_v}

    metrics: dict[str, float] = {}
    metrics.update(eval_intra_chunk_alignment(z_dict))
    metrics.update(eval_inter_chunk_separation(z_dict))
    metrics.update(eval_retrieval(z_dict))
    metrics.update(eval_cosine_margin(z_dict))
    metrics.update(eval_roc_auc(z_dict))
    metrics.update(eval_dimension_utilization(z_dict, cfg.latent.d_latent))
    metrics.update(eval_phoneme_probe_accuracy(val_loader, adapters, device))

    metrics.update(eval_modality_recovery(z_dict))

    if stage in ("B", "C"):
        metrics.update(eval_reconstruction_mse(val_loader, adapters, cfg, device))
        metrics.update(eval_kl_per_modality(val_loader, adapters, device))
        metrics.update(eval_uniformity_all(z_dict))
        metrics.update(eval_cross_modal_mi_proxy(z_dict))

    logger.info(
        "[Stage %s] intra=%.3f  R@1(v->ph)=%.3f",
        stage,
        metrics.get("intra_alignment_mean", 0.0),
        metrics.get("R@1_video->phoneme", 0.0),
    )
    return metrics
