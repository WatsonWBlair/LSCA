#!/usr/bin/env python3
# scripts/run_ablations.py
# Ablation harness: sweeps over d_latent, modality combos, MoCo, and phoneme adapter type.
#
# Usage:
#   python scripts/run_ablations.py --feature-dir datasets/pregenerated
#   python scripts/run_ablations.py --dry-run --feature-dir datasets/pregenerated
#   python scripts/run_ablations.py --variants d_latent_256 no_moco

import argparse
import json
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_ablations")

# ---------------------------------------------------------------------------
# Ablation matrix — keys are variant names, values are extra arg dicts.
# True-valued bool flags are emitted as bare flags (e.g. --no-moco).
# List-valued flags are emitted space-separated (e.g. --modalities video phoneme).
# ---------------------------------------------------------------------------
ABLATION_MATRIX: dict[str, dict] = {
    # --- d_latent variants ---
    "d_latent_256": {"--d-latent": "256"},
    "d_latent_512": {"--d-latent": "512"},
    "d_latent_768": {},  # baseline (default)
    # --- modality combos (6 valid pairs + full) ---
    "mod_video_phoneme":  {"--modalities": ["video", "phoneme"]},
    "mod_video_prosody":  {"--modalities": ["video", "prosody"]},
    "mod_phoneme_prosody": {"--modalities": ["phoneme", "prosody"]},
    "mod_all":            {},  # baseline (default: all 3)
    # --- single-modality (expected to fail validate()) ---
    "mod_video_only":   {"--modalities": ["video"]},
    "mod_phoneme_only": {"--modalities": ["phoneme"]},
    "mod_prosody_only": {"--modalities": ["prosody"]},
    # --- MoCo ablation ---
    "no_moco": {"--no-moco": True},
    # --- phoneme adapter type ---
    "phoneme_avae":   {"--phoneme-adapter-type": "avae"},
    "phoneme_linear": {"--phoneme-adapter-type": "linear"},  # baseline (default)
}


def build_command(base_args: list[str], extra: dict) -> list[str]:
    """Append extra ablation args to the base command."""
    cmd = list(base_args)
    for flag, val in extra.items():
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        elif isinstance(val, list):
            cmd.append(flag)
            cmd.extend(val)
        else:
            cmd.extend([flag, str(val)])
    return cmd


def run_variant(name: str, cmd: list[str], dry_run: bool) -> dict:
    """Execute one variant. Returns result dict."""
    logger.info("Variant: %s", name)
    logger.info("  Command: %s", " ".join(cmd))
    if dry_run:
        return {"variant": name, "status": "dry_run", "command": cmd}

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            logger.info("  [OK] %s", name)
            return {"variant": name, "status": "ok", "returncode": 0}
        else:
            logger.warning("  [FAIL] %s — exit %d", name, result.returncode)
            stderr_tail = result.stderr.strip().splitlines()[-5:] if result.stderr else []
            return {
                "variant": name,
                "status": "error",
                "returncode": result.returncode,
                "error": "\n".join(stderr_tail),
            }
    except subprocess.TimeoutExpired:
        logger.error("  [TIMEOUT] %s", name)
        return {"variant": name, "status": "timeout"}
    except Exception as exc:  # noqa: BLE001
        logger.error("  [EXCEPTION] %s — %s", name, exc)
        return {"variant": name, "status": "exception", "error": str(exc)}


def parse_args():
    p = argparse.ArgumentParser(description="CAMELS ablation harness")
    p.add_argument("--feature-dir",  required=True,    help="Dir with pregenerated .npy features")
    p.add_argument("--device",       default="cpu")
    p.add_argument("--dry-run",      action="store_true", help="Print commands without executing")
    p.add_argument("--variants",     nargs="*",         help="Run only these variants (default: all)")
    p.add_argument("--output",       default="checkpoints/ablations/ablation_results.json")
    # Pass-through training flags
    p.add_argument("--stage-a-epochs", default=20, type=int)
    p.add_argument("--stage-b-epochs", default=20, type=int)
    p.add_argument("--stage-c-epochs", default=20, type=int)
    p.add_argument("--batch-size",     default=64, type=int)
    return p.parse_args()


def main():
    args = parse_args()

    # Select variants
    variants = args.variants if args.variants else list(ABLATION_MATRIX.keys())
    unknown = set(variants) - set(ABLATION_MATRIX)
    if unknown:
        logger.error("Unknown variant(s): %s. Valid: %s", unknown, list(ABLATION_MATRIX))
        sys.exit(1)

    train_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "train_adapters.py"
    )

    results = []
    for name in variants:
        extra = ABLATION_MATRIX[name]
        ckpt_dir = f"checkpoints/ablations/{name}"
        history_path = f"{ckpt_dir}/training_history.json"

        base_cmd = [
            sys.executable, train_script,
            "--feature-dir", args.feature_dir,
            "--device", args.device,
            "--checkpoint-dir", ckpt_dir,
            "--history-path", history_path,
            "--stage-a-epochs", str(args.stage_a_epochs),
            "--stage-b-epochs", str(args.stage_b_epochs),
            "--stage-c-epochs", str(args.stage_c_epochs),
            "--batch-size", str(args.batch_size),
        ]
        cmd = build_command(base_cmd, extra)
        result = run_variant(name, cmd, args.dry_run)
        results.append(result)

    # Aggregate results
    if not args.dry_run:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results written to %s", args.output)

        ok = sum(1 for r in results if r["status"] == "ok")
        failed = sum(1 for r in results if r["status"] == "error")
        logger.info("Summary: %d ok, %d failed, %d total", ok, failed, len(results))
    else:
        logger.info("Dry-run complete. %d commands printed.", len(results))


if __name__ == "__main__":
    main()
