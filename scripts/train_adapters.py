#!/usr/bin/env python3
# scripts/train_adapters.py
# Run the full 3-stage training protocol (A -> B -> C) on pre-extracted features.
#
# Usage:
#   python scripts/train_adapters.py \
#       --feature-dir outputs/features \
#       --checkpoint-dir checkpoints/ \
#       --device cuda \
#       --batch-size 64

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_adapters")


def parse_args():
    p = argparse.ArgumentParser(description="Train CAMELS adapters (Stages A/B/C)")
    p.add_argument("--feature-dir",    default="outputs/features", help="Dir with v/ph/p_raw.npy")
    p.add_argument("--checkpoint-dir", default="checkpoints/",     help="Where to save checkpoints")
    p.add_argument("--history-path",   default="training_history.json")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--batch-size",     default=64,   type=int)
    p.add_argument("--d-latent",       default=768,  type=int,   help="Latent dimension")
    p.add_argument("--stage-a-epochs", default=20,   type=int)
    p.add_argument("--stage-b-epochs", default=20,   type=int)
    p.add_argument("--stage-c-epochs", default=20,   type=int)
    p.add_argument("--lr",             default=1e-4, type=float)
    p.add_argument("--eval-every",     default=5,    type=int)
    # Modality / adapter flags
    p.add_argument("--modalities",     nargs="+",    default=["video", "phoneme", "prosody"],
                   help="Active modalities (e.g. --modalities video phoneme)")
    p.add_argument("--phoneme-adapter-type", choices=["linear", "avae"], default="linear")
    # MoCo flags
    p.add_argument("--no-moco",        action="store_true",  help="Disable MoCo; fall back to InfoNCE")
    p.add_argument("--moco-momentum",  default=0.999, type=float)
    p.add_argument("--moco-queue-size", default=4096, type=int)
    # Loss hyperparameters
    p.add_argument("--temperature",    default=0.07,  type=float)
    p.add_argument("--weight-decay",   default=1e-4,  type=float)
    p.add_argument("--lambda-var",     default=0.04,  type=float)
    p.add_argument("--lambda-cov",     default=0.04,  type=float)
    p.add_argument("--lambda-orth",    default=0.01,  type=float)
    p.add_argument("--lambda-aux",     default=0.1,   type=float)
    p.add_argument("--c-max-video",    default=25.0,  type=float)
    p.add_argument("--c-max-prosody",  default=10.0,  type=float)
    p.add_argument("--c-max-phoneme",  default=10.0,  type=float)
    return p.parse_args()


def main():
    args = parse_args()

    from encoding.config import CAMELSConfig, LatentConfig, ModalityConfig, TrainingConfig, MoCoConfig
    from encoding.adapters.registry import build_adapters
    from encoding.training.dataset import make_dataloaders
    from encoding.training.train import train_all_stages

    mods = set(args.modalities)
    cfg = CAMELSConfig(
        latent=LatentConfig(d_latent=args.d_latent),
        modality=ModalityConfig(
            video_enabled="video" in mods,
            phoneme_enabled="phoneme" in mods,
            prosody_enabled="prosody" in mods,
            phoneme_adapter_type=args.phoneme_adapter_type,
        ),
        training=TrainingConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            lambda_var=args.lambda_var,
            lambda_cov=args.lambda_cov,
            lambda_orth=args.lambda_orth,
            lambda_aux=args.lambda_aux,
            c_max_video=args.c_max_video,
            c_max_prosody=args.c_max_prosody,
            c_max_phoneme=args.c_max_phoneme,
            stage_a_epochs=args.stage_a_epochs,
            stage_b_epochs=args.stage_b_epochs,
            stage_c_epochs=args.stage_c_epochs,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            checkpoint_dir=args.checkpoint_dir,
        ),
        moco=MoCoConfig(
            enabled=not args.no_moco,
            momentum=args.moco_momentum,
            queue_size=args.moco_queue_size,
            temperature=args.temperature,
        ),
    )
    cfg.validate()

    logger.info("Building adapters (d_latent=%d) ...", cfg.latent.d_latent)
    adapters = build_adapters(cfg)

    logger.info("Loading dataset from %s ...", args.feature_dir)
    train_loader, val_loader, _ = make_dataloaders(
        feature_dir=args.feature_dir,
        cfg=cfg,
        batch_size=cfg.training.batch_size,
    )

    history = train_all_stages(
        train_loader=train_loader,
        val_loader=val_loader,
        adapters=adapters,
        cfg=cfg,
        device=args.device,
        history_path=args.history_path,
    )

    logger.info("Training complete. %d total epochs across all stages.",
                sum(len(v) for v in history.values()))


if __name__ == "__main__":
    main()
