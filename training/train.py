# training/train.py
# Three-stage training protocol for CAMELS AVAE adapters.
#   Stage A: InfoNCE only — aligns all 4 modalities in shared 1024-D space
#   Stage B: + AVAE reconstruction + KL + z-consistency
#   Stage C: + bidirectional flow matching (video ↔ audio)
# Ref: SimCLR Chen et al. 2020; AvaeFlow Li et al. 2025; FM-Refiner 2025/26

import json
import logging
import os
from typing import Optional

import torch
import torch.optim as optim

from pipeline.config  import D_LATENT
from training.losses  import all_pairs_nce, all_avae_loss, bidirectional_fm_loss, monitor_nce_pairs
from training.evaluate import run_evaluation

logger = logging.getLogger(__name__)


# ── Hyperparameters ───────────────────────────────────────────────────────────

DEFAULTS = dict(
    lr              = 1e-4,
    weight_decay    = 1e-4,
    temperature     = 0.07,
    kl_weight       = 1e-4,
    sigma_min       = 1e-4,
    stage_a_epochs  = 20,
    stage_b_epochs  = 20,
    stage_c_epochs  = 20,
    eval_every      = 5,     # epochs between evaluation runs
    checkpoint_dir  = "checkpoints",
)


# ── Stage A: Contrastive alignment ───────────────────────────────────────────

def train_stage_a(
    train_loader,
    val_loader,
    adapters:    dict,
    device:      str  = "cpu",
    epochs:      int  = 20,
    lr:          float = 1e-4,
    weight_decay: float = 1e-4,
    temperature:  float = 0.07,
    eval_every:   int  = 5,
    checkpoint_dir: str = "checkpoints",
) -> list:
    """
    Stage A — InfoNCE contrastive alignment only.
    All 6 pairs active. mu only — no sampling, no decoder, no FM.
    Goal: bring all 4 modalities into the same geometric neighbourhood.
    This is the most important stage for HyperGNN handoff quality.

    Returns: list of per-epoch loss dicts.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    history = []

    # Trainable params: all 4 adapters + TemporalAttentionPool (no VelocityNets yet)
    params = []
    for key in ["temporal_pool", "video_adapter", "audio_adapter",
                "prosody_adapter", "text_adapter"]:
        params += list(adapters[key].parameters())
        adapters[key].to(device)

    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────────────────
        for key in ["temporal_pool", "video_adapter", "audio_adapter",
                    "prosody_adapter", "text_adapter"]:
            adapters[key].train()

        epoch_loss = 0.0
        n_batches  = 0

        for v_raw, a_raw, p_raw, t_raw in train_loader:
            v_raw = v_raw.to(device); a_raw = a_raw.to(device)
            p_raw = p_raw.to(device); t_raw = t_raw.to(device)

            z_v = adapters["video_adapter"].embed(v_raw)     # (B, 1024)
            z_a = adapters["audio_adapter"].embed(a_raw)     # (B, 1024)
            z_p = adapters["prosody_adapter"].embed(p_raw)   # (B, 1024)
            z_t = adapters["text_adapter"].embed(t_raw)      # (B, 1024)

            loss = all_pairs_nce(z_v, z_a, z_p, z_t, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history.append({"stage": "A", "epoch": epoch, "loss": avg_loss})
        logger.info("Stage A | Epoch %3d/%d | loss=%.4f", epoch, epochs, avg_loss)

        # ── Evaluation ────────────────────────────────────────────────────
        if epoch % eval_every == 0 or epoch == epochs:
            for key in ["temporal_pool", "video_adapter", "audio_adapter",
                        "prosody_adapter", "text_adapter"]:
                adapters[key].eval()

            # Per-pair InfoNCE monitoring
            _log_pair_losses(train_loader, adapters, device, temperature, epoch, "A")

            metrics = run_evaluation(val_loader, adapters, device=device, stage="A")
            history[-1]["eval"] = metrics

            # Checkpoint
            _save_checkpoint(adapters, checkpoint_dir, "stage_a", epoch)

    return history


# ── Stage B: + AVAE ───────────────────────────────────────────────────────────

def train_stage_b(
    train_loader,
    val_loader,
    adapters:    dict,
    device:      str   = "cpu",
    epochs:      int   = 20,
    lr:          float = 1e-4,
    weight_decay: float = 1e-4,
    temperature:  float = 0.07,
    kl_weight:   float = 1e-4,
    eval_every:  int   = 5,
    checkpoint_dir: str = "checkpoints",
) -> list:
    """
    Stage B — InfoNCE + AVAE (reconstruction + KL + z-consistency).
    Adds the decoder and reencoder to the training graph.
    Reconstruction targets:
      video   → MARLIN embedding space  (target MSE < 0.05)
      audio   → wav2vec2 space           (target MSE < 0.10)
      prosody → normalized prosody       (target MSE < 0.02)
      text    → SONAR space              (target MSE < 0.05)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    history = []

    params = []
    for key in ["temporal_pool", "video_adapter", "audio_adapter",
                "prosody_adapter", "text_adapter"]:
        params += list(adapters[key].parameters())
        adapters[key].to(device)

    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        for key in ["temporal_pool", "video_adapter", "audio_adapter",
                    "prosody_adapter", "text_adapter"]:
            adapters[key].train()

        epoch_nce  = 0.0
        epoch_avae = 0.0
        n_batches  = 0

        for v_raw, a_raw, p_raw, t_raw in train_loader:
            v_raw = v_raw.to(device); a_raw = a_raw.to(device)
            p_raw = p_raw.to(device); t_raw = t_raw.to(device)

            mu_v, lv_v, z_v, xh_v, zp_v = adapters["video_adapter"](v_raw)
            mu_a, lv_a, z_a, xh_a, zp_a = adapters["audio_adapter"](a_raw)
            mu_p, lv_p, z_p, xh_p, zp_p = adapters["prosody_adapter"](p_raw)
            mu_t, lv_t, z_t, xh_t, zp_t = adapters["text_adapter"](t_raw)

            l_nce  = all_pairs_nce(mu_v, mu_a, mu_p, mu_t, temperature=temperature)
            l_avae = all_avae_loss(
                v_raw, xh_v, mu_v, lv_v, z_v, zp_v,
                a_raw, xh_a, mu_a, lv_a, z_a, zp_a,
                p_raw, xh_p, mu_p, lv_p, z_p, zp_p,
                t_raw, xh_t, mu_t, lv_t, z_t, zp_t,
                kl_weight=kl_weight,
            )
            loss = l_nce + l_avae

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_nce  += l_nce.item()
            epoch_avae += l_avae.item()
            n_batches  += 1

        avg_nce  = epoch_nce  / max(n_batches, 1)
        avg_avae = epoch_avae / max(n_batches, 1)
        history.append({
            "stage": "B", "epoch": epoch,
            "loss_nce": avg_nce, "loss_avae": avg_avae,
            "loss_total": avg_nce + avg_avae,
        })
        logger.info(
            "Stage B | Epoch %3d/%d | nce=%.4f  avae=%.4f",
            epoch, epochs, avg_nce, avg_avae,
        )

        if epoch % eval_every == 0 or epoch == epochs:
            for key in ["temporal_pool", "video_adapter", "audio_adapter",
                        "prosody_adapter", "text_adapter"]:
                adapters[key].eval()

            metrics = run_evaluation(val_loader, adapters, device=device, stage="B")
            history[-1]["eval"] = metrics
            _save_checkpoint(adapters, checkpoint_dir, "stage_b", epoch)

    return history


# ── Stage C: + Bidirectional Flow Matching ────────────────────────────────────

def train_stage_c(
    train_loader,
    val_loader,
    adapters:    dict,
    device:      str   = "cpu",
    epochs:      int   = 20,
    lr:          float = 1e-4,
    weight_decay: float = 1e-4,
    temperature:  float = 0.07,
    kl_weight:   float = 1e-4,
    sigma_min:   float = 1e-4,
    eval_every:  int   = 5,
    checkpoint_dir: str = "checkpoints",
) -> list:
    """
    Stage C — InfoNCE + AVAE + bidirectional flow matching (video ↔ audio).
    CRITICAL: adapter gradients are DETACHED before FM loss.
    Only VelocityNets receive FM gradients — adapters are not updated by FM.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    history = []

    # Adapter params (same as Stage B)
    adapter_params = []
    for key in ["temporal_pool", "video_adapter", "audio_adapter",
                "prosody_adapter", "text_adapter"]:
        adapter_params += list(adapters[key].parameters())
        adapters[key].to(device)

    # VelocityNet params (separate optimizer so FM doesn't touch adapters)
    vel_params = (
        list(adapters["velocity_va"].parameters()) +
        list(adapters["velocity_av"].parameters())
    )
    adapters["velocity_va"].to(device)
    adapters["velocity_av"].to(device)

    opt_adapters  = optim.AdamW(adapter_params, lr=lr, weight_decay=weight_decay)
    opt_velocity  = optim.AdamW(vel_params,     lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        for key in adapters:
            adapters[key].train()

        epoch_nce  = 0.0
        epoch_avae = 0.0
        epoch_fm   = 0.0
        n_batches  = 0

        for v_raw, a_raw, p_raw, t_raw in train_loader:
            v_raw = v_raw.to(device); a_raw = a_raw.to(device)
            p_raw = p_raw.to(device); t_raw = t_raw.to(device)

            mu_v, lv_v, z_v, xh_v, zp_v = adapters["video_adapter"](v_raw)
            mu_a, lv_a, z_a, xh_a, zp_a = adapters["audio_adapter"](a_raw)
            mu_p, lv_p, z_p, xh_p, zp_p = adapters["prosody_adapter"](p_raw)
            mu_t, lv_t, z_t, xh_t, zp_t = adapters["text_adapter"](t_raw)

            l_nce  = all_pairs_nce(mu_v, mu_a, mu_p, mu_t, temperature=temperature)
            l_avae = all_avae_loss(
                v_raw, xh_v, mu_v, lv_v, z_v, zp_v,
                a_raw, xh_a, mu_a, lv_a, z_a, zp_a,
                p_raw, xh_p, mu_p, lv_p, z_p, zp_p,
                t_raw, xh_t, mu_t, lv_t, z_t, zp_t,
                kl_weight=kl_weight,
            )

            # FM loss: adapters detached — only VelocityNets updated
            l_fm = bidirectional_fm_loss(
                mu_v, mu_a,
                adapters["velocity_va"], adapters["velocity_av"],
                sigma_min=sigma_min,
            )

            # Update adapters with NCE + AVAE
            opt_adapters.zero_grad()
            (l_nce + l_avae).backward(retain_graph=True)
            opt_adapters.step()

            # Update VelocityNets with FM only
            opt_velocity.zero_grad()
            l_fm.backward()
            opt_velocity.step()

            epoch_nce  += l_nce.item()
            epoch_avae += l_avae.item()
            epoch_fm   += l_fm.item()
            n_batches  += 1

        avg_nce  = epoch_nce  / max(n_batches, 1)
        avg_avae = epoch_avae / max(n_batches, 1)
        avg_fm   = epoch_fm   / max(n_batches, 1)
        history.append({
            "stage": "C", "epoch": epoch,
            "loss_nce": avg_nce, "loss_avae": avg_avae, "loss_fm": avg_fm,
            "loss_total": avg_nce + avg_avae + avg_fm,
        })
        logger.info(
            "Stage C | Epoch %3d/%d | nce=%.4f  avae=%.4f  fm=%.4f",
            epoch, epochs, avg_nce, avg_avae, avg_fm,
        )

        if epoch % eval_every == 0 or epoch == epochs:
            for key in adapters:
                adapters[key].eval()
            metrics = run_evaluation(val_loader, adapters, device=device, stage="C")
            history[-1]["eval"] = metrics
            _save_checkpoint(adapters, checkpoint_dir, "stage_c", epoch)

    return history


# ── Full training run (A → B → C) ────────────────────────────────────────────

def train_all_stages(
    train_loader,
    val_loader,
    adapters:      dict,
    device:        str = "cpu",
    hparams:       dict = None,
    checkpoint_dir: str = "checkpoints",
    history_path:  str = "training_history.json",
) -> dict:
    """
    Run the complete 3-stage training protocol.
    Saves history to history_path as JSON for inspection.
    Returns combined history dict.
    """
    hp = {**DEFAULTS, **(hparams or {})}

    logger.info("=" * 60)
    logger.info("CAMELS Training — device=%s", device)
    logger.info("Stage A: %d epochs | Stage B: %d epochs | Stage C: %d epochs",
                hp["stage_a_epochs"], hp["stage_b_epochs"], hp["stage_c_epochs"])
    logger.info("=" * 60)

    hist_a = train_stage_a(
        train_loader, val_loader, adapters, device=device,
        epochs=hp["stage_a_epochs"], lr=hp["lr"],
        weight_decay=hp["weight_decay"], temperature=hp["temperature"],
        eval_every=hp["eval_every"], checkpoint_dir=checkpoint_dir,
    )
    hist_b = train_stage_b(
        train_loader, val_loader, adapters, device=device,
        epochs=hp["stage_b_epochs"], lr=hp["lr"],
        weight_decay=hp["weight_decay"], temperature=hp["temperature"],
        kl_weight=hp["kl_weight"], eval_every=hp["eval_every"],
        checkpoint_dir=checkpoint_dir,
    )
    hist_c = train_stage_c(
        train_loader, val_loader, adapters, device=device,
        epochs=hp["stage_c_epochs"], lr=hp["lr"],
        weight_decay=hp["weight_decay"], temperature=hp["temperature"],
        kl_weight=hp["kl_weight"], sigma_min=hp["sigma_min"],
        eval_every=hp["eval_every"], checkpoint_dir=checkpoint_dir,
    )

    history = {"A": hist_a, "B": hist_b, "C": hist_c}
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=_json_default)
    logger.info("Training complete. History saved to %s", history_path)
    return history


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_pair_losses(loader, adapters, device, temperature, epoch, stage):
    """Log per-pair InfoNCE on first batch (fast diagnostic)."""
    try:
        v_raw, a_raw, p_raw, t_raw = next(iter(loader))
        v_raw = v_raw.to(device); a_raw = a_raw.to(device)
        p_raw = p_raw.to(device); t_raw = t_raw.to(device)
        with torch.no_grad():
            z_v = adapters["video_adapter"].embed(v_raw)
            z_a = adapters["audio_adapter"].embed(a_raw)
            z_p = adapters["prosody_adapter"].embed(p_raw)
            z_t = adapters["text_adapter"].embed(t_raw)
        pairs = monitor_nce_pairs(z_v, z_a, z_p, z_t, temperature)
        parts = "  ".join(f"{k}={v:.3f}" for k, v in pairs.items())
        logger.info("Stage %s | Epoch %d | pair losses: %s", stage, epoch, parts)
    except Exception as e:
        logger.warning("_log_pair_losses failed: %s", e)


def _save_checkpoint(adapters, checkpoint_dir, stage_name, epoch):
    from pipeline.adapters import save_adapters
    path = os.path.join(checkpoint_dir, f"{stage_name}_epoch{epoch:03d}.pt")
    save_adapters(adapters, path)
    logger.info("Checkpoint saved: %s", path)


def _json_default(obj):
    """JSON serializer for non-serializable types (e.g., numpy floats)."""
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)
