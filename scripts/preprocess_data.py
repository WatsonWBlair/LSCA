#!/usr/bin/env python3
# scripts/preprocess_data.py
# DEPRECATED: Use `invoke generate-wrangled-tokens` for new work.
#   This script operates on raw .mp4/.wav files directly and predates the
#   wrangled-dataset pipeline. Prefer the invoke task for all new training runs.
#
# Run ONCE before training to extract raw features from all .mp4/.wav files.
#
# Saves to output_dir:
#   v_raw.npy      (N, 768)              MARLIN + TemporalPool
#   ph_raw.npy     (N, MAX_PHONES, 768)  per-phoneme wav2vec2-CTC hidden states
#   ph_labels.npy  (N, MAX_PHONES)       phoneme class IDs
#   ph_mask.npy    (N, MAX_PHONES)        1 = real phoneme, 0 = padding
#   p_raw.npy      (N, 22)               librosa prosody (z-scored)
#   prosody_stats.json                    z-score stats (fit on training split only)
#
# Usage:
#   python scripts/preprocess_data.py \
#       --data-root data/ \
#       --output-dir outputs/features \
#       --device cpu

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure Homebrew and common tool paths are visible (needed for ffprobe/ffmpeg)
for _extra in ("/opt/homebrew/bin", "/usr/local/bin"):
    if _extra not in os.environ.get("PATH", "") and os.path.isdir(_extra):
        os.environ["PATH"] = _extra + os.pathsep + os.environ.get("PATH", "")

import librosa
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preprocess_data")


def parse_args():
    p = argparse.ArgumentParser(description="Extract raw multimodal features for training")
    p.add_argument("--data-root",   default="data/",            help="Root dir containing .mp4/.wav files")
    p.add_argument("--output-dir",  default="outputs/features", help="Where to save .npy files")
    p.add_argument("--device",      default="cpu",              help="cpu | cuda | mps")
    p.add_argument("--max-videos",  default=None, type=int,     help="Limit number of videos (debugging)")
    p.add_argument("--d-latent",    default=768, type=int,      help="Latent dimension (default: 768)")
    p.add_argument("--max-phones",  default=50, type=int,       help="Max phoneme positions per chunk")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from encoding.config import CAMELSConfig, LatentConfig
    from encoding.models.loader import load_all_models
    from encoding.adapters.registry import build_adapters
    from encoding.pipelines.video import extract_video_file
    from encoding.pipelines.phoneme import phoneme_pipeline, pad_phonemes
    from encoding.pipelines.prosody import extract_prosody_raw, fit_prosody_stats, z_score

    cfg = CAMELSConfig(
        latent=LatentConfig(d_latent=args.d_latent, max_phones=args.max_phones),
    )

    # ── 1. Load frozen models ──────────────────────────────────────────
    logger.info("Loading frozen models (device=%s) ...", args.device)
    models = load_all_models(cfg, device=args.device)

    # Auto-detect phoneme classes from CTC model vocab
    if "num_phoneme_classes" in models:
        cfg.latent.num_phoneme_classes = models["num_phoneme_classes"]
        logger.info("Detected %d phoneme classes from CTC model", cfg.latent.num_phoneme_classes)

    # ── 2. Discover files ──────────────────────────────────────────────
    mp4_files = sorted(Path(args.data_root).rglob("*.mp4"))
    wav_files = sorted(Path(args.data_root).rglob("*.wav"))
    if args.max_videos:
        mp4_files = mp4_files[:args.max_videos]
        wav_files = wav_files[:args.max_videos]

    logger.info("Found %d .mp4 files, %d .wav files", len(mp4_files), len(wav_files))

    # ── 3. Fit prosody stats on training audio ─────────────────────────
    logger.info("Fitting prosody z-score stats ...")
    all_prosody_raws = []
    for wav_path in wav_files:
        try:
            waveform, _ = librosa.load(str(wav_path), sr=cfg.streaming.sample_rate, mono=True)
            raw = extract_prosody_raw(
                waveform.astype(np.float32),
                sr=cfg.streaming.sample_rate,
                d_prosody=cfg.latent.d_prosody,
            )
            all_prosody_raws.append(raw)
        except Exception as e:
            logger.warning("Failed to extract prosody from %s: %s", wav_path, e)

    prosody_stats = None
    if all_prosody_raws:
        prosody_stats = fit_prosody_stats(all_prosody_raws)
        stats_path = os.path.join(args.output_dir, "prosody_stats.json")
        with open(stats_path, "w") as f:
            json.dump(prosody_stats, f, indent=2)
        logger.info("Prosody stats saved to %s", stats_path)
    else:
        logger.warning("No .wav files found — prosody stats not fitted")

    # ── 4. Extract video features ──────────────────────────────────────
    v_rows = []
    logger.info("Extracting video features ...")
    marlin_model = models.get("marlin")
    adapters = build_adapters(cfg)
    temporal_pool = adapters["temporal_pool"]
    for i, mp4_path in enumerate(mp4_files):
        try:
            z_v = extract_video_file(str(mp4_path), marlin_model, temporal_pool, cfg)
            v_rows.append(z_v.detach().cpu().numpy())
            if (i + 1) % 10 == 0:
                logger.info("  Video: %d/%d", i + 1, len(mp4_files))
        except Exception as e:
            logger.warning("Failed video extraction for %s: %s", mp4_path, e)

    # ── 5. Extract phoneme features ────────────────────────────────────
    ph_emb_rows, ph_label_rows, ph_mask_rows = [], [], []
    logger.info("Extracting phoneme features ...")
    wav2vec2_ctc = models.get("wav2vec2_ctc")
    wav2vec2_proc = models.get("wav2vec2_processor")
    for i, wav_path in enumerate(wav_files):
        try:
            waveform, _ = librosa.load(str(wav_path), sr=cfg.streaming.sample_rate, mono=True)
            audio_tensor = torch.from_numpy(waveform).unsqueeze(0).to(args.device)
            embs, labels, mask, _ = phoneme_pipeline(audio_tensor, wav2vec2_ctc, wav2vec2_proc, cfg)
            padded_embs, padded_labels, padded_mask = pad_phonemes(
                embs, labels, mask, cfg.latent.max_phones, cfg.latent.d_phoneme,
            )
            ph_emb_rows.append(padded_embs.detach().cpu().numpy())
            ph_label_rows.append(padded_labels.detach().cpu().numpy())
            ph_mask_rows.append(padded_mask.detach().cpu().numpy())
            if (i + 1) % 10 == 0:
                logger.info("  Phoneme: %d/%d", i + 1, len(wav_files))
        except Exception as e:
            logger.warning("Failed phoneme extraction for %s: %s", wav_path, e)

    # ── 6. Extract prosody features ────────────────────────────────────
    p_rows = []
    logger.info("Extracting prosody features ...")
    for i, wav_path in enumerate(wav_files):
        try:
            waveform, _ = librosa.load(str(wav_path), sr=cfg.streaming.sample_rate, mono=True)
            raw = extract_prosody_raw(
                waveform.astype(np.float32),
                sr=cfg.streaming.sample_rate,
                d_prosody=cfg.latent.d_prosody,
            )
            if prosody_stats is not None:
                raw = z_score(raw, prosody_stats)
            p_rows.append(raw)
            if (i + 1) % 10 == 0:
                logger.info("  Prosody: %d/%d", i + 1, len(wav_files))
        except Exception as e:
            logger.warning("Failed prosody extraction for %s: %s", wav_path, e)

    # ── 7. Save .npy files ─────────────────────────────────────────────
    n = min(len(v_rows), len(ph_emb_rows), len(p_rows))
    if n == 0:
        logger.error("No features extracted — check data paths")
        return

    logger.info("Saving %d aligned samples to %s", n, args.output_dir)
    np.save(os.path.join(args.output_dir, "v_raw.npy"),      np.stack(v_rows[:n]))
    np.save(os.path.join(args.output_dir, "ph_raw.npy"),     np.stack(ph_emb_rows[:n]))
    np.save(os.path.join(args.output_dir, "ph_labels.npy"),  np.stack(ph_label_rows[:n]))
    np.save(os.path.join(args.output_dir, "ph_mask.npy"),    np.stack(ph_mask_rows[:n]))
    np.save(os.path.join(args.output_dir, "p_raw.npy"),      np.stack(p_rows[:n]))

    logger.info("Preprocessing complete. Files saved to: %s", os.path.abspath(args.output_dir))
    logger.info("Next step: python scripts/train_adapters.py --feature-dir %s", args.output_dir)


if __name__ == "__main__":
    main()
