# encoding/training/train.py
# Three-stage training protocol for CAMELS v8.1.
#   Stage A: InfoNCE + L_var + L_cov + L_aux
#   Stage B: + AVAE (capacity-controlled) + L_orth
#   Stage C: + bidirectional FM (video <-> phoneme)
# Dynamic loss assembly based on active modalities and current stage.

from __future__ import annotations

import json
import logging
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from encoding.config import CAMELSConfig
from encoding.adapters.registry import save_adapters
from encoding.training.losses import (
    all_pairs_nce, all_pairs_moco, avae_loss, get_capacity,
    cross_modal_orth_loss, variance_loss, covariance_loss,
    bidirectional_fm_loss, phoneme_probe_loss, monitor_nce_pairs,
)
from encoding.training.evaluate import run_evaluation

logger = logging.getLogger(__name__)


def _get_z_dict(z_v, z_ph_pooled, z_p, cfg: CAMELSConfig) -> dict[str, torch.Tensor]:
    """Build modality -> embedding dict for enabled modalities."""
    z_dict = {}
    if cfg.modality.video_enabled:
        z_dict["video"] = z_v
    if cfg.modality.phoneme_enabled:
        z_dict["phoneme"] = z_ph_pooled
    if cfg.modality.prosody_enabled:
        z_dict["prosody"] = z_p
    return z_dict


def _forward_batch(batch, adapters, cfg, device, stage):
    """Run forward pass for one batch. Returns all needed tensors."""
    v_raw, ph_raw, ph_labels, ph_mask, p_raw = batch
    v_raw = v_raw.to(device)
    ph_raw = ph_raw.to(device)
    ph_labels = ph_labels.to(device)
    ph_mask = ph_mask.to(device)
    p_raw = p_raw.to(device)

    result = {"v_raw": v_raw, "ph_raw": ph_raw, "ph_labels": ph_labels, "ph_mask": ph_mask, "p_raw": p_raw}

    # Video
    if stage == "A":
        z_v = adapters["video_adapter"].embed(v_raw)
        result.update({"z_v": z_v})
    else:
        mu_v, lv_v, z_v, xh_v, zp_v = adapters["video_adapter"](v_raw)
        result.update({"mu_v": mu_v, "lv_v": lv_v, "z_v": z_v, "xh_v": xh_v, "zp_v": zp_v})

    # Phoneme — branch on adapter type
    if cfg.modality.phoneme_enabled:
        if cfg.modality.phoneme_adapter_type == "avae":
            # Mean-pool raw features over unmasked positions → (B, d_phoneme)
            mask_f = ph_mask.float().unsqueeze(-1)  # (B, MAX, 1)
            ph_pooled = (ph_raw * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            if stage == "A":
                z_ph_pooled = adapters["phoneme_adapter"].embed(ph_pooled)
                result.update({"z_ph_pooled": z_ph_pooled})
            else:
                mu_ph, lv_ph, z_ph_pooled, xh_ph, zp_ph = adapters["phoneme_adapter"](ph_pooled)
                result.update({
                    "ph_pooled": ph_pooled,
                    "mu_ph": mu_ph, "lv_ph": lv_ph,
                    "z_ph_pooled": z_ph_pooled, "xh_ph": xh_ph, "zp_ph": zp_ph,
                })
        else:
            z_ph_seq = adapters["phoneme_adapter"](ph_raw)          # (B, MAX, d_latent)
            z_ph_pooled = adapters["phoneme_attn_pool"](z_ph_seq, ph_mask)  # (B, d_latent)
            result.update({"z_ph_seq": z_ph_seq, "z_ph_pooled": z_ph_pooled})
            if stage in ("B", "C"):
                mask_f = ph_mask.float().unsqueeze(-1)
                ph_pooled_for_recon = (ph_raw * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
                result["ph_pooled_for_recon"] = ph_pooled_for_recon

    # Prosody
    if stage == "A":
        z_p = adapters["prosody_adapter"].embed(p_raw)
        result.update({"z_p": z_p})
    else:
        mu_p, lv_p, z_p, xh_p, zp_p = adapters["prosody_adapter"](p_raw)
        result.update({"mu_p": mu_p, "lv_p": lv_p, "z_p": z_p, "xh_p": xh_p, "zp_p": zp_p})

    return result


def _compute_losses(
    fwd, adapters, cfg, stage, epoch, stage_b_start, stage_b_end,
    z_k_dict=None, momentum_manager=None,
):
    """Compute all active losses for the current stage. Returns (total, losses_dict)."""
    tc = cfg.training
    losses = {}

    # Contrastive loss — MoCo if enabled, else InfoNCE
    z_v_for_nce = fwd.get("mu_v", fwd.get("z_v"))
    z_p_for_nce = fwd.get("mu_p", fwd.get("z_p"))
    z_dict = _get_z_dict(z_v_for_nce, fwd["z_ph_pooled"], z_p_for_nce, cfg)
    if cfg.moco.enabled and z_k_dict is not None and momentum_manager is not None:
        l_nce, nce_pairs = all_pairs_moco(
            z_dict, z_k_dict, momentum_manager.queues, temperature=tc.temperature
        )
    else:
        l_nce, nce_pairs = all_pairs_nce(z_dict, temperature=tc.temperature)
    losses["nce"] = l_nce
    losses.update(nce_pairs)

    # L_var + L_cov — Stage A+
    z_list = list(z_dict.values())
    losses["var"] = variance_loss(z_list, gamma=tc.gamma_var)
    losses["cov"] = covariance_loss(z_list)

    # L_aux (phoneme probe) — Stage A+
    if "phoneme_probe" in adapters:
        losses["aux"] = phoneme_probe_loss(
            fwd["z_ph_seq"], fwd["ph_labels"], fwd["ph_mask"], adapters["phoneme_probe"],
        )
    else:
        losses["aux"] = torch.tensor(0.0, device=l_nce.device)

    total = l_nce + tc.lambda_var * losses["var"] + tc.lambda_cov * losses["cov"] + tc.lambda_aux * losses["aux"]

    # Stage B+: AVAE + L_orth
    if stage in ("B", "C"):
        # Video AVAE
        C_v = get_capacity(epoch, stage_b_start, stage_b_end, tc.c_max_video)
        v_avae = avae_loss(fwd["v_raw"], fwd["xh_v"], fwd["mu_v"], fwd["lv_v"], fwd["z_v"], fwd["zp_v"], C_v, tc.beta_cap)
        for k, v in v_avae.items():
            losses[f"avae_video_{k}"] = v

        # Prosody AVAE
        C_p = get_capacity(epoch, stage_b_start, stage_b_end, tc.c_max_prosody)
        p_avae = avae_loss(fwd["p_raw"], fwd["xh_p"], fwd["mu_p"], fwd["lv_p"], fwd["z_p"], fwd["zp_p"], C_p, tc.beta_cap)
        for k, v in p_avae.items():
            losses[f"avae_prosody_{k}"] = v

        l_avae = v_avae["total"] + p_avae["total"]

        # Phoneme AVAE (only when adapter type is avae)
        if cfg.modality.phoneme_enabled and cfg.modality.phoneme_adapter_type == "avae":
            C_ph = get_capacity(epoch, stage_b_start, stage_b_end, tc.c_max_phoneme)
            ph_avae = avae_loss(
                fwd["ph_pooled"], fwd["xh_ph"], fwd["mu_ph"], fwd["lv_ph"],
                fwd["z_ph_pooled"], fwd["zp_ph"], C_ph, tc.beta_cap,
            )
            for k, v in ph_avae.items():
                losses[f"avae_phoneme_{k}"] = v
            l_avae = l_avae + ph_avae["total"]

        losses["avae_total"] = l_avae

        # L_orth
        losses["orth"] = cross_modal_orth_loss(z_list)

        # Phoneme decoder reconstruction (linear adapter path only)
        if "phoneme_decoder" in adapters and "ph_pooled_for_recon" in fwd:
            ph_hat = adapters["phoneme_decoder"](fwd["z_ph_pooled"])
            losses["recon_phoneme"] = F.mse_loss(ph_hat, fwd["ph_pooled_for_recon"])
        else:
            losses["recon_phoneme"] = torch.tensor(0.0, device=l_nce.device)

        total = total + l_avae + tc.lambda_orth * losses["orth"] + losses["recon_phoneme"]

    return total, losses


def train_stage_a(train_loader, val_loader, adapters, cfg, device="cpu", momentum_manager=None):
    """Stage A: contrastive alignment + geometric regularization."""
    tc = cfg.training
    os.makedirs(tc.checkpoint_dir, exist_ok=True)
    history = []

    # Trainable: adapters + temporal_pool + phoneme components (NO velocity)
    params = []
    train_keys = [k for k in adapters if "velocity" not in k]
    for key in train_keys:
        adapters[key].to(device)
        params.extend(adapters[key].parameters())

    optimizer = optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)

    for epoch in range(1, tc.stage_a_epochs + 1):
        for key in train_keys:
            adapters[key].train()

        epoch_losses = {}
        n_batches = 0

        for batch in train_loader:
            z_k_dict = momentum_manager.encode_keys(batch, cfg, device, "A") if momentum_manager else None
            fwd = _forward_batch(batch, adapters, cfg, device, "A")
            total, losses = _compute_losses(
                fwd, adapters, cfg, "A", epoch, 0, 0,
                z_k_dict=z_k_dict, momentum_manager=momentum_manager,
            )

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            if momentum_manager is not None:
                momentum_manager.ema_update(adapters, cfg.moco.momentum)
                momentum_manager.enqueue(z_k_dict)

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + (v.item() if torch.is_tensor(v) else v)
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        history.append({"stage": "A", "epoch": epoch, **avg})
        logger.info("Stage A | Epoch %3d/%d | nce=%.4f var=%.4f cov=%.4f aux=%.4f",
                     epoch, tc.stage_a_epochs, avg.get("nce", 0), avg.get("var", 0), avg.get("cov", 0), avg.get("aux", 0))

        if epoch % tc.eval_every == 0 or epoch == tc.stage_a_epochs:
            for key in train_keys:
                adapters[key].eval()
            metrics = run_evaluation(val_loader, adapters, cfg, device=device, stage="A")
            history[-1]["eval"] = metrics
            save_adapters(adapters, os.path.join(tc.checkpoint_dir, f"stage_a_epoch{epoch:03d}.pt"))

    return history


def train_stage_b(train_loader, val_loader, adapters, cfg, device="cpu", momentum_manager=None):
    """Stage B: + AVAE reconstruction + L_orth + capacity control."""
    tc = cfg.training
    os.makedirs(tc.checkpoint_dir, exist_ok=True)
    history = []

    params = []
    train_keys = [k for k in adapters if "velocity" not in k]
    for key in train_keys:
        adapters[key].to(device)
        params.extend(adapters[key].parameters())

    optimizer = optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    stage_b_start = 1
    stage_b_end = tc.stage_b_epochs

    for epoch in range(1, tc.stage_b_epochs + 1):
        for key in train_keys:
            adapters[key].train()

        epoch_losses = {}
        n_batches = 0

        for batch in train_loader:
            z_k_dict = momentum_manager.encode_keys(batch, cfg, device, "B") if momentum_manager else None
            fwd = _forward_batch(batch, adapters, cfg, device, "B")
            total, losses = _compute_losses(
                fwd, adapters, cfg, "B", epoch, stage_b_start, stage_b_end,
                z_k_dict=z_k_dict, momentum_manager=momentum_manager,
            )

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            if momentum_manager is not None:
                momentum_manager.ema_update(adapters, cfg.moco.momentum)
                momentum_manager.enqueue(z_k_dict)

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + (v.item() if torch.is_tensor(v) else v)
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        history.append({"stage": "B", "epoch": epoch, **avg})
        logger.info("Stage B | Epoch %3d/%d | nce=%.4f avae=%.4f orth=%.4f",
                     epoch, tc.stage_b_epochs, avg.get("nce", 0), avg.get("avae_total", 0), avg.get("orth", 0))

        if epoch % tc.eval_every == 0 or epoch == tc.stage_b_epochs:
            for key in train_keys:
                adapters[key].eval()
            metrics = run_evaluation(val_loader, adapters, cfg, device=device, stage="B")
            history[-1]["eval"] = metrics
            save_adapters(adapters, os.path.join(tc.checkpoint_dir, f"stage_b_epoch{epoch:03d}.pt"))

    return history


def train_stage_c(train_loader, val_loader, adapters, cfg, device="cpu", momentum_manager=None):
    """Stage C: + bidirectional flow matching (video <-> phoneme)."""
    tc = cfg.training
    os.makedirs(tc.checkpoint_dir, exist_ok=True)
    history = []

    # Adapter params (same as Stage B)
    adapter_keys = [k for k in adapters if "velocity" not in k]
    adapter_params = []
    for key in adapter_keys:
        adapters[key].to(device)
        adapter_params.extend(adapters[key].parameters())

    # VelocityNet params (separate optimizer)
    vel_keys = [k for k in adapters if "velocity" in k]
    vel_params = []
    for key in vel_keys:
        adapters[key].to(device)
        vel_params.extend(adapters[key].parameters())

    opt_adapters = optim.AdamW(adapter_params, lr=tc.lr, weight_decay=tc.weight_decay)
    opt_velocity = optim.AdamW(vel_params, lr=tc.lr, weight_decay=tc.weight_decay)

    stage_b_start = 1
    stage_b_end = tc.stage_b_epochs  # capacity continues from B schedule

    for epoch in range(1, tc.stage_c_epochs + 1):
        for key in adapters:
            adapters[key].train()

        epoch_losses = {}
        n_batches = 0

        for batch in train_loader:
            z_k_dict = momentum_manager.encode_keys(batch, cfg, device, "C") if momentum_manager else None
            fwd = _forward_batch(batch, adapters, cfg, device, "C")
            total, losses = _compute_losses(
                fwd, adapters, cfg, "B", epoch + tc.stage_b_epochs, stage_b_start, stage_b_end,
                z_k_dict=z_k_dict, momentum_manager=momentum_manager,
            )

            # FM loss (adapters detached)
            l_fm = bidirectional_fm_loss(
                fwd.get("mu_v", fwd["z_v"]),
                fwd["z_ph_pooled"],
                adapters["velocity_vph"],
                adapters["velocity_phv"],
                sigma_min=tc.sigma_min,
            )
            losses["fm"] = l_fm

            # Update adapters with NCE + AVAE + geometric losses
            opt_adapters.zero_grad()
            total.backward(retain_graph=True)
            opt_adapters.step()

            # EMA update after adapter step, before velocity step
            if momentum_manager is not None:
                momentum_manager.ema_update(adapters, cfg.moco.momentum)
                momentum_manager.enqueue(z_k_dict)

            # Update VelocityNets with FM only
            opt_velocity.zero_grad()
            l_fm.backward()
            opt_velocity.step()

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + (v.item() if torch.is_tensor(v) else v)
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        history.append({"stage": "C", "epoch": epoch, **avg})
        logger.info("Stage C | Epoch %3d/%d | nce=%.4f avae=%.4f fm=%.4f",
                     epoch, tc.stage_c_epochs, avg.get("nce", 0), avg.get("avae_total", 0), avg.get("fm", 0))

        if epoch % tc.eval_every == 0 or epoch == tc.stage_c_epochs:
            for key in adapters:
                adapters[key].eval()
            metrics = run_evaluation(val_loader, adapters, cfg, device=device, stage="C")
            history[-1]["eval"] = metrics
            save_adapters(adapters, os.path.join(tc.checkpoint_dir, f"stage_c_epoch{epoch:03d}.pt"))

    return history


def train_all_stages(
    train_loader,
    val_loader,
    adapters: dict,
    cfg: CAMELSConfig,
    device: str = "cpu",
    history_path: str = "training_history.json",
) -> dict:
    """Run the complete 3-stage training protocol."""
    logger.info("=" * 60)
    logger.info("CAMELS v8.1 Training — device=%s, d_latent=%d", device, cfg.latent.d_latent)
    logger.info("Stage A: %d | B: %d | C: %d epochs",
                cfg.training.stage_a_epochs, cfg.training.stage_b_epochs, cfg.training.stage_c_epochs)
    logger.info("=" * 60)

    from encoding.training.momentum import MomentumEncoderManager
    momentum_manager = MomentumEncoderManager(adapters, cfg, device) if cfg.moco.enabled else None

    hist_a = train_stage_a(train_loader, val_loader, adapters, cfg, device, momentum_manager=momentum_manager)
    hist_b = train_stage_b(train_loader, val_loader, adapters, cfg, device, momentum_manager=momentum_manager)
    hist_c = train_stage_c(train_loader, val_loader, adapters, cfg, device, momentum_manager=momentum_manager)

    history = {"A": hist_a, "B": hist_b, "C": hist_c}
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    logger.info("Training complete. History saved to %s", history_path)
    return history
