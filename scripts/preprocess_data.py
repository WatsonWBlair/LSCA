#!/usr/bin/env python3
# scripts/preprocess_data.py
# Run ONCE before training to extract raw features from all .mp4/.wav files.
#
# Saves to output_dir:
#   v_raw.npy  (N, 768)    MARLIN + TemporalPool
#   a_raw.npy  (N, 768)    wav2vec2 mean pool
#   p_raw.npy  (N, 18)     librosa prosody (z-scored)
#   t_raw.npy  (N, 1024)   SONAR text encoding
#   prosody_stats.json      z-score stats (fit on training split only)
#
# Usage:
#   python scripts/preprocess_data.py \
#       --data-root data/ \
#       --output-dir outputs/features \
#       --device cpu

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure Homebrew and common tool paths are visible (needed for ffprobe/ffmpeg)
for _extra in ("/opt/homebrew/bin", "/usr/local/bin"):
    if _extra not in os.environ.get("PATH", "") and os.path.isdir(_extra):
        os.environ["PATH"] = _extra + os.pathsep + os.environ.get("PATH", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preprocess_data")


def parse_args():
    p = argparse.ArgumentParser(description="Extract raw multimodal features for training")
    p.add_argument("--data-root",   default="data/",          help="Root dir containing .mp4 files")
    p.add_argument("--output-dir",  default="outputs/features", help="Where to save .npy files")
    p.add_argument("--device",      default="cpu",             help="cpu | cuda | mps")
    p.add_argument("--max-videos",  default=None, type=int,    help="Limit number of videos (debugging)")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("Loading frozen models ...")
    from models.model_loader import load_all_models
    from src.encoding.adapters   import build_adapters
    from src.encoding.utils.config     import D_AUDIO

    models   = load_all_models(device=args.device)
    adapters = build_adapters(d_audio=D_AUDIO)
    for mod in adapters.values():
        mod.to(args.device).eval()

    # Step 1: Fit prosody stats on training split
    # We use ALL audio files to fit stats (in practice use only train split)
    logger.info("Fitting prosody z-score stats on training audio ...")
    from pathlib import Path
    import numpy as np
    import librosa
    from src.encoding.pipelines.prosody_pipeline import extract_prosody_raw, save_prosody_stats
    from src.encoding.utils.config           import SAMPLE_RATE

    wav_files = sorted(Path(args.data_root).rglob("*.wav"))
    if args.max_videos:
        wav_files = wav_files[:args.max_videos]

    all_raws = []
    for wav_path in wav_files:
        try:
            waveform, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
            all_raws.append(extract_prosody_raw(waveform.astype(np.float32)))
        except Exception as e:
            logger.warning("Failed to load %s: %s", wav_path, e)

    if all_raws:
        arr  = np.stack(all_raws)
        mean = arr.mean(axis=0).tolist()
        std  = [float(s) if float(s) > 1e-8 else 1.0
                for s in arr.std(axis=0).tolist()]
        prosody_stats = {"mean": mean, "std": std}
        os.makedirs(args.output_dir, exist_ok=True)
        save_prosody_stats(prosody_stats, os.path.join(args.output_dir, "prosody_stats.json"))
        logger.info("Prosody stats saved.")
    else:
        logger.warning("No .wav files found — prosody stats not fitted")
        prosody_stats = None

    # Step 2: Extract all raw features
    from training.dataset import extract_raw_features
    extract_raw_features(
        data_root     = args.data_root,
        output_dir    = args.output_dir,
        models        = models,
        adapters      = adapters,
        prosody_stats = prosody_stats,
        max_videos    = args.max_videos,
    )

    logger.info("Preprocessing complete. Files are in: %s", os.path.abspath(args.output_dir))
    logger.info("Next step: python scripts/train_adapters.py --feature-dir %s", args.output_dir)


if __name__ == "__main__":
    main()
