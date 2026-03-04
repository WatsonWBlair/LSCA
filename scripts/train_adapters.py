#!/usr/bin/env python3
# scripts/train_adapters.py
# Run the full 3-stage training protocol (A → B → C) on pre-extracted features.
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
    p = argparse.ArgumentParser(description="Train CAMELS AVAE adapters (Stages A/B/C)")
    p.add_argument("--feature-dir",    default="outputs/features", help="Dir with v/a/p/t_raw.npy")
    p.add_argument("--checkpoint-dir", default="checkpoints/",     help="Where to save checkpoints")
    p.add_argument("--history-path",   default="training_history.json")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--batch-size",     default=64,  type=int)
    p.add_argument("--stage-a-epochs", default=20,  type=int)
    p.add_argument("--stage-b-epochs", default=20,  type=int)
    p.add_argument("--stage-c-epochs", default=20,  type=int)
    p.add_argument("--lr",             default=1e-4, type=float)
    p.add_argument("--kl-weight",      default=1e-4, type=float)
    return p.parse_args()


def main():
    args = parse_args()

    from src.encoding.adapters  import build_adapters
    from src.encoding.utils.config    import D_AUDIO
    from training.dataset   import make_dataloaders
    from training.train     import train_all_stages

    logger.info("Building adapters ...")
    adapters = build_adapters(d_audio=D_AUDIO)
    for mod in adapters.values():
        mod.to(args.device)

    logger.info("Loading dataset from %s ...", args.feature_dir)
    train_loader, val_loader, test_loader = make_dataloaders(
        feature_dir=args.feature_dir,
        batch_size=args.batch_size,
    )

    hparams = dict(
        stage_a_epochs = args.stage_a_epochs,
        stage_b_epochs = args.stage_b_epochs,
        stage_c_epochs = args.stage_c_epochs,
        lr             = args.lr,
        kl_weight      = args.kl_weight,
    )

    train_all_stages(
        train_loader   = train_loader,
        val_loader     = val_loader,
        adapters       = adapters,
        device         = args.device,
        hparams        = hparams,
        checkpoint_dir = args.checkpoint_dir,
        history_path   = args.history_path,
    )


if __name__ == "__main__":
    main()
