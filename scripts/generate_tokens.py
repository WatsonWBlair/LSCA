#!/usr/bin/env python3
# scripts/generate_tokens.py
# Pre-generate backbone tokens from wrangled conversation videos for adapter training.
#
# Runs frozen MARLIN and wav2vec2-CTC over all wrangled mp4/wav pairs,
# chunks them into sliding windows (2s window, 1s stride), and saves the
# raw backbone outputs as .npy files compatible with MultimodalDataset.
#
# Output layout (output_dir/):
#   v_raw.npy          (N, 768)        MARLIN + TemporalAttentionPool per chunk
#   ph_raw.npy         (N, 50, 768)    wav2vec2-CTC per-phoneme hidden states
#   ph_labels.npy      (N, 50)         phoneme class IDs
#   ph_mask.npy        (N, 50)         1=real phoneme, 0=padding
#   p_raw.npy          (N, 22)         librosa features (z-scored)
#   prosody_stats.json                 z-score stats (mean/std over all chunks)
#
# Usage:
#   python scripts/generate_tokens.py \
#       --data-root datasets/ \
#       --output-dir outputs/features \
#       --device cpu \
#       --max-pairs 2

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure Homebrew and common tool paths are visible (needed for ffprobe/ffmpeg)
for _extra in ("/opt/homebrew/bin", "/usr/local/bin"):
    if _extra not in os.environ.get("PATH", "") and os.path.isdir(_extra):
        os.environ["PATH"] = _extra + os.pathsep + os.environ.get("PATH", "")

import librosa
import numpy as np
import torch
import torchvision.transforms.functional as TF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_tokens")


def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-generate backbone tokens from wrangled conversations"
    )
    p.add_argument("--data-root",   default="datasets/",       help="Root dir to search for mp4/wav pairs")
    p.add_argument("--output-dir",  default="outputs/features", help="Where to save .npy files")
    p.add_argument("--device",      default="cpu",              help="cpu | cuda | mps")
    p.add_argument("--max-pairs",   default=None, type=int,     help="Limit pairs (debugging)")
    p.add_argument("--d-latent",    default=768, type=int,      help="Latent dimension (default: 768)")
    p.add_argument("--max-phones",  default=50, type=int,       help="Max phoneme positions per chunk")
    return p.parse_args()


def discover_pairs(data_root: str, max_pairs: int | None) -> list[tuple[Path, Path]]:
    """
    Find all .mp4 files under data_root. For each, check if a .wav with the
    same stem exists in the same directory. Warn and skip unmatched files.
    """
    mp4_files = sorted(Path(data_root).rglob("*.mp4"))
    pairs = []
    skipped = 0

    for mp4_path in mp4_files:
        wav_path = mp4_path.with_suffix(".wav")
        if wav_path.exists():
            pairs.append((mp4_path, wav_path))
        else:
            logger.warning("No matching .wav for %s — skipping", mp4_path)
            skipped += 1

    if skipped:
        logger.warning("Skipped %d .mp4 files with no matching .wav", skipped)

    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    logger.info("Found %d matched mp4/wav pairs", len(pairs))
    return pairs


def load_all_frames(mp4_path: Path, cfg) -> tuple[list[torch.Tensor], float]:
    """
    Load and preprocess all frames from an mp4: BGR→RGB, resize to marlin_size,
    ImageNet normalize. Returns (frames, video_fps).
    """
    import cv2

    marlin_size = cfg.streaming.marlin_size
    im_mean = list(cfg.streaming.imagenet_mean)
    im_std = list(cfg.streaming.imagenet_std)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        logger.warning("load_all_frames: cannot open %s", mp4_path)
        return [], 30.0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[torch.Tensor] = []

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (marlin_size, marlin_size), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = TF.normalize(t, mean=im_mean, std=im_std)
        frames.append(t)

    cap.release()
    return frames, video_fps


def iter_chunks(
    mp4_path: Path,
    wav_path: Path,
    cfg,
) -> Iterator[tuple[list[torch.Tensor], np.ndarray, float]]:
    """
    Yield (chunk_frames, audio_chunk, video_fps) for each time-aligned sliding window.
    Window: 2s, stride: 1s.
    """
    # Load full audio at sample_rate
    waveform, _ = librosa.load(str(wav_path), sr=cfg.streaming.sample_rate, mono=True)

    # Load + preprocess all video frames
    frames, video_fps = load_all_frames(mp4_path, cfg)

    if not frames:
        logger.warning("iter_chunks: no frames from %s — skipping", mp4_path)
        return

    # Chunk boundaries
    audio_win    = int(cfg.streaming.window_sec * cfg.streaming.sample_rate)
    audio_stride = int(cfg.streaming.stride_sec * cfg.streaming.sample_rate)
    video_win    = int(cfg.streaming.window_sec * video_fps)
    video_stride = int(cfg.streaming.stride_sec * video_fps)

    audio_start = 0
    frame_start = 0

    while audio_start + audio_win <= len(waveform):
        yield (
            frames[frame_start : frame_start + video_win],
            waveform[audio_start : audio_start + audio_win],
            video_fps,
        )
        audio_start += audio_stride
        frame_start += video_stride


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from encoding.config import CAMELSConfig, LatentConfig
    from encoding.models.loader import load_all_models
    from encoding.adapters.registry import build_adapters
    from encoding.pipelines.video import video_pipeline
    from encoding.pipelines.phoneme import phoneme_pipeline, pad_phonemes
    from encoding.pipelines.prosody import extract_prosody_raw, z_score

    cfg = CAMELSConfig(
        latent=LatentConfig(d_latent=args.d_latent, max_phones=args.max_phones),
    )

    # ── 1. Load frozen models ──────────────────────────────────────────────
    logger.info("Loading frozen models (device=%s) ...", args.device)
    models = load_all_models(cfg, device=args.device)

    if "num_phoneme_classes" in models:
        cfg.latent.num_phoneme_classes = models["num_phoneme_classes"]
        logger.info("Detected %d phoneme classes", cfg.latent.num_phoneme_classes)

    marlin_model   = models["marlin"]
    wav2vec2_ctc   = models["wav2vec2_ctc"]
    wav2vec2_proc  = models["wav2vec2_processor"]

    # ── 2. Build TemporalAttentionPool ─────────────────────────────────────
    adapters = build_adapters(cfg)
    temporal_pool = adapters["temporal_pool"]

    # ── 3. Discover pairs ─────────────────────────────────────────────────
    pairs = discover_pairs(args.data_root, args.max_pairs)
    if not pairs:
        logger.error("No matched mp4/wav pairs found under %s", args.data_root)
        return

    # ── 4. Single-pass extraction ─────────────────────────────────────────
    v_rows, ph_rows, ph_label_rows, ph_mask_rows, p_raw_rows = [], [], [], [], []
    total_chunks = 0

    for pair_idx, (mp4_path, wav_path) in enumerate(pairs):
        logger.info("Processing pair %d/%d: %s", pair_idx + 1, len(pairs), mp4_path.name)
        pair_chunks = 0

        for chunk_frames, audio_chunk, fps in iter_chunks(mp4_path, wav_path, cfg):
            try:
                # Video: frozen MARLIN + TemporalAttentionPool
                v = video_pipeline(chunk_frames, fps, marlin_model, temporal_pool, cfg)

                # Phoneme: frozen wav2vec2-CTC
                embs, labels, mask, _ = phoneme_pipeline(
                    audio_chunk, wav2vec2_ctc, wav2vec2_proc, cfg,
                )
                padded_embs, padded_labels, padded_mask = pad_phonemes(
                    embs, labels, mask,
                    cfg.latent.max_phones, cfg.latent.d_phoneme,
                )

                # Prosody: frozen librosa (raw; z-scoring done after all chunks)
                p = extract_prosody_raw(
                    audio_chunk,
                    sr=cfg.streaming.sample_rate,
                    d_prosody=cfg.latent.d_prosody,
                )

                v_rows.append(v.detach().cpu().numpy())
                ph_rows.append(padded_embs.detach().cpu().numpy())
                ph_label_rows.append(padded_labels.detach().cpu().numpy())
                ph_mask_rows.append(padded_mask.detach().cpu().numpy())
                p_raw_rows.append(p)

                pair_chunks += 1
                total_chunks += 1

            except Exception as e:
                logger.warning("Chunk error in %s: %s", mp4_path.name, e)

        logger.info("  → %d chunks from this pair (total: %d)", pair_chunks, total_chunks)

    if total_chunks == 0:
        logger.error("No chunks extracted — check data paths and formats")
        return

    # ── 5. Fit prosody z-score stats from extracted raws ──────────────────
    logger.info("Fitting prosody z-score stats over %d chunks ...", total_chunks)
    arr = np.stack(p_raw_rows)           # (N, d_prosody)
    mean = arr.mean(axis=0).tolist()
    std  = arr.std(axis=0).tolist()
    std  = [s if s > 1e-8 else 1.0 for s in std]
    prosody_stats = {"mean": mean, "std": std}

    stats_path = os.path.join(args.output_dir, "prosody_stats.json")
    with open(stats_path, "w") as f:
        json.dump(prosody_stats, f, indent=2)
    logger.info("Prosody stats saved to %s", stats_path)

    # Z-score all prosody rows
    p_rows = [z_score(p, prosody_stats) for p in p_raw_rows]

    # ── 6. Save .npy files ────────────────────────────────────────────────
    logger.info("Saving %d aligned chunks to %s ...", total_chunks, args.output_dir)
    np.save(os.path.join(args.output_dir, "v_raw.npy"),     np.stack(v_rows))
    np.save(os.path.join(args.output_dir, "ph_raw.npy"),    np.stack(ph_rows))
    np.save(os.path.join(args.output_dir, "ph_labels.npy"), np.stack(ph_label_rows))
    np.save(os.path.join(args.output_dir, "ph_mask.npy"),   np.stack(ph_mask_rows))
    np.save(os.path.join(args.output_dir, "p_raw.npy"),     np.stack(p_rows))

    logger.info("Token generation complete. Files saved to: %s", os.path.abspath(args.output_dir))
    logger.info("Next step: python scripts/train_adapters.py --feature-dir %s", args.output_dir)


if __name__ == "__main__":
    main()
