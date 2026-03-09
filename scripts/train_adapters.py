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
    return p.parse_args()


def main():
    args = parse_args()

    from encoding.config import CAMELSConfig, LatentConfig, TrainingConfig
    from encoding.adapters.registry import build_adapters
    from encoding.training.dataset import make_dataloaders
    from encoding.training.train import train_all_stages

    cfg = CAMELSConfig(
        latent=LatentConfig(d_latent=args.d_latent),
        training=TrainingConfig(
            lr=args.lr,
            stage_a_epochs=args.stage_a_epochs,
            stage_b_epochs=args.stage_b_epochs,
            stage_c_epochs=args.stage_c_epochs,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            checkpoint_dir=args.checkpoint_dir,
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
