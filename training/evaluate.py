# training/evaluate.py
# All evaluation metrics for the CAMELS training protocol.
# Run after every 5 epochs during Stages A, B, C, and on the final test set.
# Ref: Wang & Isola 2020 (uniformity); SimCLR Chen et al. 2020 (linear probe)

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Intra-chunk alignment ─────────────────────────────────────────────────────

def eval_intra_chunk_alignment(
    z_v: torch.Tensor,
    z_a: torch.Tensor,
    z_p: torch.Tensor,
    z_t: torch.Tensor,
) -> Dict[str, float]:
    """
    For every chunk in the val set: compute pairwise cosine similarity between
    all 4 modality vectors for the SAME chunk (positive pairs).
    Average over all 6 pairs and all chunks.
    This is the primary quality metric for the HyperGNN handoff.

    Target: intra_alignment_mean > 0.5 after Stage A.
    """
    pairs = [
        (z_v, z_a, "v-a"),
        (z_v, z_p, "v-p"),
        (z_v, z_t, "v-t"),
        (z_a, z_p, "a-p"),
        (z_a, z_t, "a-t"),
        (z_p, z_t, "p-t"),
    ]
    results = {}
    for za, zb, name in pairs:
        sim = F.cosine_similarity(
            F.normalize(za, dim=-1),
            F.normalize(zb, dim=-1),
        )   # (N,)
        results[f"intra_alignment_{name}"] = sim.mean().item()

    results["intra_alignment_mean"] = float(np.mean(list(results.values())))
    return results


# ── Inter-chunk separation ────────────────────────────────────────────────────

def eval_inter_chunk_separation(
    z_v: torch.Tensor,
    z_a: torch.Tensor,
    z_p: torch.Tensor,
    z_t: torch.Tensor,
) -> Dict[str, float]:
    """
    Mean cosine similarity between the SAME modality across DIFFERENT chunks.
    Should be low — confirms the latent space is not collapsed.

    Target: < 0.2 for all modalities.
    Warning: if > 0.5, the latent space has collapsed.
    """
    results = {}
    for z, name in [(z_v, "v"), (z_a, "a"), (z_p, "p"), (z_t, "t")]:
        z_norm     = F.normalize(z, dim=-1)
        sim_matrix = torch.matmul(z_norm, z_norm.T)   # (N, N)
        # Exclude diagonal (self-similarity = 1.0)
        mask = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
        results[f"inter_sep_{name}"] = sim_matrix[mask].mean().item()
    return results


# ── Cross-modal retrieval (8 directions) ────────────────────────────────────

def eval_retrieval(
    z_v: torch.Tensor,
    z_a: torch.Tensor,
    z_p: torch.Tensor,
    z_t: torch.Tensor,
) -> Dict[str, float]:
    """
    Cross-modal R@1 and R@5 retrieval in all 12 non-self directions.
    Given a query embedding from one modality, retrieve the matching
    chunk from another modality by nearest-neighbour cosine search.

    Targets (after Stage A):
      R@1 > 0.50  (each of 12 directions)
      R@5 > 0.75
    """
    modalities = [("video", z_v), ("audio", z_a), ("prosody", z_p), ("text", z_t)]
    results    = {}

    for src_name, src in modalities:
        for tgt_name, tgt in modalities:
            if src_name == tgt_name:
                continue
            sims   = torch.matmul(
                F.normalize(src, dim=-1),
                F.normalize(tgt, dim=-1).T,
            )                                               # (N, N)
            ranked = sims.argsort(dim=1, descending=True)  # (N, N) — closest first
            labels = torch.arange(len(src), device=src.device)

            r1 = (ranked[:, 0] == labels).float().mean().item()
            r5 = (ranked[:, :5] == labels.unsqueeze(1)).any(dim=1).float().mean().item()

            results[f"R@1_{src_name}->{tgt_name}"] = r1
            results[f"R@5_{src_name}->{tgt_name}"] = r5

    return results


# ── Alignment uniformity ──────────────────────────────────────────────────────

def eval_uniformity(z: torch.Tensor, name: str = "") -> float:
    """
    Uniformity metric from Wang & Isola (ICML 2020).
    log( mean( exp(-2 * pairwise_sq_dist) ) )
    Lower value = more uniformly spread on unit sphere.
    Target: stable or decreasing across stages.
    """
    z_norm  = F.normalize(z, dim=-1)
    sq_dist = torch.pdist(z_norm, p=2).pow(2)
    u       = sq_dist.mul(-2).exp().mean().log().item()
    if name:
        logger.debug("Uniformity [%s]: %.4f", name, u)
    return u


def eval_uniformity_all(
    z_v: torch.Tensor,
    z_a: torch.Tensor,
    z_p: torch.Tensor,
    z_t: torch.Tensor,
) -> Dict[str, float]:
    return {
        "uniformity_v": eval_uniformity(z_v, "v"),
        "uniformity_a": eval_uniformity(z_a, "a"),
        "uniformity_p": eval_uniformity(z_p, "p"),
        "uniformity_t": eval_uniformity(z_t, "t"),
    }


# ── Cosine margin (per pair) ──────────────────────────────────────────────────

def eval_cosine_margin(
    z_v: torch.Tensor,
    z_a: torch.Tensor,
    z_p: torch.Tensor,
    z_t: torch.Tensor,
) -> Dict[str, float]:
    """
    For each pair: mean_sim(same chunk) - mean_sim(diff chunk).
    Target: > 0.3 for each pair after Stage A.
    """
    pairs = [
        (z_v, z_a, "v-a"),
        (z_v, z_p, "v-p"),
        (z_v, z_t, "v-t"),
        (z_a, z_p, "a-p"),
        (z_a, z_t, "a-t"),
        (z_p, z_t, "p-t"),
    ]
    results = {}
    for za, zb, name in pairs:
        za_n = F.normalize(za, dim=-1)
        zb_n = F.normalize(zb, dim=-1)
        sim_matrix = torch.matmul(za_n, zb_n.T)   # (N, N)

        N    = sim_matrix.shape[0]
        mask = torch.eye(N, dtype=torch.bool, device=za.device)

        pos_mean = sim_matrix[mask].mean().item()
        neg_mean = sim_matrix[~mask].mean().item()
        results[f"margin_{name}"] = pos_mean - neg_mean

    return results


# ── ROC-AUC per pair ──────────────────────────────────────────────────────────

def eval_roc_auc(
    z_v: torch.Tensor,
    z_a: torch.Tensor,
    z_p: torch.Tensor,
    z_t: torch.Tensor,
) -> Dict[str, float]:
    """
    Binary ROC-AUC: positive = same-chunk pair, negative = cross-chunk pair.
    Target: > 0.85 per pair.
    Uses only a random subsample of N^2 pairs for efficiency.
    """
    from sklearn.metrics import roc_auc_score

    pairs = [
        (z_v, z_a, "v-a"),
        (z_v, z_p, "v-p"),
        (z_v, z_t, "v-t"),
        (z_a, z_p, "a-p"),
        (z_a, z_t, "a-t"),
        (z_p, z_t, "p-t"),
    ]
    results = {}
    for za, zb, name in pairs:
        za_n = F.normalize(za, dim=-1)
        zb_n = F.normalize(zb, dim=-1)
        sim  = torch.matmul(za_n, zb_n.T)   # (N, N)
        N    = sim.shape[0]
        mask = torch.eye(N, dtype=torch.bool, device=za.device)

        scores = sim.cpu().numpy().flatten()
        labels = mask.cpu().numpy().flatten().astype(int)

        try:
            auc = roc_auc_score(labels, scores)
        except Exception:
            auc = 0.5
        results[f"roc_auc_{name}"] = auc

    return results


# ── Linear probe per modality ─────────────────────────────────────────────────

def eval_linear_probe(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val:   torch.Tensor,
    y_val:   torch.Tensor,
    name:    str = "",
) -> Dict[str, float]:
    """
    Fit a LogisticRegression on frozen embeddings to test discriminability.
    Ref: SimCLR, Chen et al. 2020.
    Returns accuracy and macro-F1.
    """
    from sklearn.linear_model  import LogisticRegression
    from sklearn.metrics        import f1_score

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", multi_class="auto")
    clf.fit(z_train.cpu().numpy(), y_train.cpu().numpy())

    y_pred = clf.predict(z_val.cpu().numpy())
    acc    = float(clf.score(z_val.cpu().numpy(), y_val.cpu().numpy()))
    f1     = float(f1_score(y_val.cpu().numpy(), y_pred, average="macro", zero_division=0))

    if name:
        logger.info("Linear probe [%s]: acc=%.3f  macro-F1=%.3f", name, acc, f1)
    return {f"probe_acc_{name}": acc, f"probe_f1_{name}": f1}


# ── AVAE reconstruction MSE ───────────────────────────────────────────────────

def eval_reconstruction_mse(
    val_loader,
    adapters: dict,
    device:   str = "cpu",
) -> Dict[str, float]:
    """
    Decode z back to the original input domain and measure MSE.
    Expected targets:
      video  < 0.05
      audio  < 0.10
      prosody < 0.02
      text   < 0.05
    """
    mse_v, mse_a, mse_p, mse_t = [], [], [], []

    for v_raw, a_raw, p_raw, t_raw in val_loader:
        v_raw = v_raw.to(device)
        a_raw = a_raw.to(device)
        p_raw = p_raw.to(device)
        t_raw = t_raw.to(device)

        with torch.no_grad():
            mu_v, lv_v = adapters["video_adapter"].encode(v_raw)
            z_v        = adapters["video_adapter"].reparameterize(mu_v, lv_v)
            xh_v       = adapters["video_adapter"].decode(z_v)
            mse_v.append(F.mse_loss(xh_v, v_raw).item())

            mu_a, lv_a = adapters["audio_adapter"].encode(a_raw)
            z_a        = adapters["audio_adapter"].reparameterize(mu_a, lv_a)
            xh_a       = adapters["audio_adapter"].decode(z_a)
            mse_a.append(F.mse_loss(xh_a, a_raw).item())

            mu_p, lv_p = adapters["prosody_adapter"].encode(p_raw)
            z_p        = adapters["prosody_adapter"].reparameterize(mu_p, lv_p)
            xh_p       = adapters["prosody_adapter"].decode(z_p)
            mse_p.append(F.mse_loss(xh_p, p_raw).item())

            mu_t, lv_t = adapters["text_adapter"].encode(t_raw)
            z_t        = adapters["text_adapter"].reparameterize(mu_t, lv_t)
            xh_t       = adapters["text_adapter"].decode(z_t)
            mse_t.append(F.mse_loss(xh_t, t_raw).item())

    return {
        "recon_mse_video":   float(np.mean(mse_v)),
        "recon_mse_audio":   float(np.mean(mse_a)),
        "recon_mse_prosody": float(np.mean(mse_p)),
        "recon_mse_text":    float(np.mean(mse_t)),
    }


# ── KL per modality ───────────────────────────────────────────────────────────

def eval_kl_per_modality(
    val_loader,
    adapters: dict,
    device:   str = "cpu",
) -> Dict[str, float]:
    """Monitor KL divergence per modality — should be stable, not exploding."""
    kl_v, kl_a, kl_p, kl_t = [], [], [], []

    for v_raw, a_raw, p_raw, t_raw in val_loader:
        v_raw = v_raw.to(device); a_raw = a_raw.to(device)
        p_raw = p_raw.to(device); t_raw = t_raw.to(device)

        with torch.no_grad():
            for raw, key, bucket in [
                (v_raw, "video_adapter",   kl_v),
                (a_raw, "audio_adapter",   kl_a),
                (p_raw, "prosody_adapter", kl_p),
                (t_raw, "text_adapter",    kl_t),
            ]:
                mu, lv = adapters[key].encode(raw)
                kl     = -0.5 * torch.mean(1.0 + lv - mu.pow(2) - lv.exp())
                bucket.append(kl.item())

    return {
        "kl_video":   float(np.mean(kl_v)),
        "kl_audio":   float(np.mean(kl_a)),
        "kl_prosody": float(np.mean(kl_p)),
        "kl_text":    float(np.mean(kl_t)),
    }


# ── Full evaluation suite ─────────────────────────────────────────────────────

def run_evaluation(
    val_loader,
    adapters:     dict,
    device:       str = "cpu",
    stage:        str = "A",
    y_train:      Optional[torch.Tensor] = None,
    y_val:        Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Run the complete evaluation suite for a given training stage.

    Stage A: alignment + separation + retrieval + linear probe + margin + AUC
    Stage B: all Stage A + reconstruction MSE + KL + uniformity
    Stage C: all Stage B + (FM losses logged separately in train.py)

    Returns: flat dict of {metric_name: value}.
    """
    # Collect full val set embeddings
    all_v, all_a, all_p, all_t = [], [], [], []
    all_v_raw, all_a_raw, all_p_raw, all_t_raw = [], [], [], []

    for v_raw, a_raw, p_raw, t_raw in val_loader:
        v_raw = v_raw.to(device)
        a_raw = a_raw.to(device)
        p_raw = p_raw.to(device)
        t_raw = t_raw.to(device)
        with torch.no_grad():
            z_v = adapters["video_adapter"].embed(v_raw)
            z_a = adapters["audio_adapter"].embed(a_raw)
            z_p = adapters["prosody_adapter"].embed(p_raw)
            z_t = adapters["text_adapter"].embed(t_raw)
        all_v.append(z_v.cpu()); all_a.append(z_a.cpu())
        all_p.append(z_p.cpu()); all_t.append(z_t.cpu())
        all_v_raw.append(v_raw.cpu())

    z_v = torch.cat(all_v); z_a = torch.cat(all_a)
    z_p = torch.cat(all_p); z_t = torch.cat(all_t)

    metrics = {}

    # Stage A metrics (always computed)
    metrics.update(eval_intra_chunk_alignment(z_v, z_a, z_p, z_t))
    metrics.update(eval_inter_chunk_separation(z_v, z_a, z_p, z_t))
    metrics.update(eval_retrieval(z_v, z_a, z_p, z_t))
    metrics.update(eval_cosine_margin(z_v, z_a, z_p, z_t))
    metrics.update(eval_roc_auc(z_v, z_a, z_p, z_t))

    if y_train is not None and y_val is not None:
        z_train_all = torch.cat(all_v)   # use video embeddings for probe
        for z_all, name in [(z_v, "video"), (z_a, "audio"), (z_p, "prosody"), (z_t, "text")]:
            metrics.update(eval_linear_probe(z_all, y_val, z_all, y_val, name))

    # Stage B+ metrics
    if stage in ("B", "C"):
        metrics.update(eval_reconstruction_mse(val_loader, adapters, device))
        metrics.update(eval_kl_per_modality(val_loader, adapters, device))
        metrics.update(eval_uniformity_all(z_v, z_a, z_p, z_t))

    # Log a concise summary
    logger.info(
        "[Stage %s] intra=%.3f  inter_v=%.3f  R@1(v->a)=%.3f  R@5(v->a)=%.3f",
        stage,
        metrics.get("intra_alignment_mean", 0.0),
        metrics.get("inter_sep_v", 0.0),
        metrics.get("R@1_video->audio", 0.0),
        metrics.get("R@5_video->audio", 0.0),
    )
    return metrics
