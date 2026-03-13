"""Extract CAMELS frozen backbone features for MERBench evaluation.

For each segment in a MERBench dataset manifest, runs the frozen MARLIN,
wav2vec2-CTC, and librosa pipelines and saves results in MERBench's expected
{segment_id: np.ndarray} dict format as a pickle file per split.

Usage:
    python scripts/extract_merbench_features.py \
        --merbench-root datasets/merbench \
        --datasets cmu_mosei \
        --device cpu \
        --feature-mode concat_768
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CAMELS features for MERBench")
    parser.add_argument("--merbench-root", default="datasets/merbench",
                        help="Root of MERBench installation (default: datasets/merbench)")
    parser.add_argument("--datasets", default="iemocap cmu_mosei meld",
                        help="Space-separated dataset names to process")
    parser.add_argument("--device", default="cpu",
                        help="Torch device (default: cpu)")
    parser.add_argument(
        "--feature-mode",
        default="concat_768",
        choices=["concat_768", "separate"],
        help=(
            "concat_768: concatenate video(768) + phoneme_mean(768) + prosody(22) into one vector; "
            "separate: save three arrays per segment (default: concat_768)"
        ),
    )
    return parser.parse_args()


def _read_manifest(dataset_dir: Path) -> list[dict]:
    """Read MERBench segment manifest (CSV or JSON).

    Returns list of dicts with keys: segment_id, video_path, split.
    Supports:
      - label.csv / labels.csv — comma-separated with header
      - manifest.json           — list of dicts
    """
    segments: list[dict] = []

    for csv_name in ("label.csv", "labels.csv"):
        csv_path = dataset_dir / csv_name
        if csv_path.exists():
            import csv as csv_mod
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    seg_id = row.get("segment_id") or row.get("id") or row.get("name", "")
                    video_path = row.get("video_path") or row.get("video", "")
                    split = row.get("split") or row.get("set") or "train"
                    if seg_id:
                        segments.append({
                            "segment_id": seg_id,
                            "video_path": dataset_dir / video_path if video_path else None,
                            "split": split,
                        })
            return segments

    json_path = dataset_dir / "manifest.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        for entry in data:
            seg_id = entry.get("segment_id") or entry.get("id", "")
            video_path = entry.get("video_path") or entry.get("video", "")
            split = entry.get("split") or entry.get("set") or "train"
            if seg_id:
                segments.append({
                    "segment_id": seg_id,
                    "video_path": dataset_dir / video_path if video_path else None,
                    "split": split,
                })
        return segments

    logger.warning("No manifest found in %s — skipping", dataset_dir)
    return segments


def _load_models(device: str):
    """Load frozen backbone models using the same pattern as generate_wrangled_tokens.py."""
    from encoding.config import CAMELSConfig
    from encoding.models.loader import load_all_models
    from encoding.adapters.registry import build_adapters

    cfg = CAMELSConfig()
    models = load_all_models(cfg, device=device)
    if "num_phoneme_classes" in models:
        cfg.latent.num_phoneme_classes = models["num_phoneme_classes"]
    adapters = build_adapters(cfg)
    return cfg, models, adapters


def _extract_segment_features(
    video_path: Path,
    cfg,
    marlin_model,
    wav2vec2_ctc,
    wav2vec2_proc,
    temporal_pool,
    feature_mode: str,
) -> np.ndarray | dict[str, np.ndarray] | None:
    """Run all three frozen pipelines on a single video segment.

    Returns:
        concat_768 mode: 1-D numpy array of shape (1558,) = 768+768+22
        separate mode:   dict with keys "video"(768,), "phoneme"(768,), "prosody"(22,)
        None if the segment could not be processed.
    """
    import librosa

    from encoding.pipelines.video import video_pipeline
    from encoding.pipelines.phoneme import phoneme_pipeline, pad_phonemes
    from encoding.pipelines.prosody import extract_prosody_raw

    if video_path is None or not video_path.exists():
        logger.warning("Missing video: %s — skipping", video_path)
        return None

    # Derive WAV path: same stem, .wav extension alongside video
    wav_path = video_path.with_suffix(".wav")
    if not wav_path.exists():
        # Try extracting on-the-fly with ffmpeg
        import subprocess
        wav_path = video_path.with_suffix("._tmp.wav")
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ar", "16000", "-ac", "1", "-vn", str(wav_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            logger.warning("ffmpeg failed for %s — skipping", video_path)
            return None

    try:
        waveform, _ = librosa.load(str(wav_path), sr=cfg.streaming.sample_rate, mono=True)
    except Exception as e:
        logger.warning("Audio load failed for %s: %s", wav_path, e)
        return None

    # Load video frames
    import cv2
    import torchvision.transforms.functional as TF

    marlin_size = cfg.streaming.marlin_size
    im_mean = list(cfg.streaming.imagenet_mean)
    im_std = list(cfg.streaming.imagenet_std)

    cap = cv2.VideoCapture(str(video_path))
    frames: list[torch.Tensor] = []
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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

    if not frames:
        logger.warning("No frames from %s — skipping", video_path)
        return None

    try:
        v = video_pipeline(frames, video_fps, marlin_model, temporal_pool, cfg)
        embs, labels, mask, _ = phoneme_pipeline(
            waveform, wav2vec2_ctc, wav2vec2_proc, cfg,
        )
        padded_embs, _, padded_mask = pad_phonemes(
            embs, labels, mask,
            cfg.latent.max_phones, cfg.latent.d_phoneme,
        )
        p = extract_prosody_raw(
            waveform,
            sr=cfg.streaming.sample_rate,
            d_prosody=cfg.latent.d_prosody,
        )

        v_np = v.detach().cpu().numpy()                  # (768,)
        ph_np = padded_embs.detach().cpu().numpy()       # (max_phones, 768)
        mask_np = padded_mask.detach().cpu().numpy()     # (max_phones,)

        # Mean-pool phoneme embeddings over valid (unmasked) positions
        valid = mask_np.astype(bool)
        ph_mean = ph_np[valid].mean(axis=0) if valid.any() else np.zeros(v_np.shape, dtype=np.float32)

        if feature_mode == "concat_768":
            return np.concatenate([v_np, ph_mean, p]).astype(np.float32)
        else:
            return {"video": v_np, "phoneme": ph_mean, "prosody": p}

    except Exception as e:
        logger.warning("Feature extraction failed for %s: %s", video_path, e)
        return None


def main() -> None:
    args = parse_args()
    merbench_root = Path(args.merbench_root)
    dataset_names = args.datasets.split()
    feature_mode = args.feature_mode

    logger.info("Loading frozen models (device=%s) ...", args.device)
    cfg, models, adapters = _load_models(args.device)
    marlin_model  = models["marlin"]
    wav2vec2_ctc  = models["wav2vec2_ctc"]
    wav2vec2_proc = models["wav2vec2_processor"]
    temporal_pool = adapters["temporal_pool"]

    for dataset_name in dataset_names:
        dataset_dir = merbench_root / dataset_name
        if not dataset_dir.exists():
            logger.warning("Dataset directory not found: %s — skipping", dataset_dir)
            continue

        segments = _read_manifest(dataset_dir)
        if not segments:
            logger.warning("No segments in manifest for %s — skipping", dataset_name)
            continue

        # Group by split
        splits: dict[str, list[dict]] = {}
        for seg in segments:
            split = seg["split"]
            splits.setdefault(split, []).append(seg)

        for split, split_segs in splits.items():
            out_dir = merbench_root / "features" / "camels" / dataset_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{split}.pkl"

            features: dict[str, np.ndarray | dict] = {}
            logger.info(
                "Extracting %s / %s (%d segments) → %s",
                dataset_name, split, len(split_segs), out_path,
            )

            for i, seg in enumerate(split_segs):
                seg_id = seg["segment_id"]
                video_path = seg.get("video_path")
                if video_path is not None:
                    video_path = Path(video_path)

                result = _extract_segment_features(
                    video_path, cfg, marlin_model, wav2vec2_ctc, wav2vec2_proc,
                    temporal_pool, feature_mode,
                )
                if result is not None:
                    features[seg_id] = result

                if (i + 1) % 50 == 0:
                    logger.info("  %d / %d processed", i + 1, len(split_segs))

            with open(out_path, "wb") as f:
                pickle.dump(features, f, protocol=4)

            logger.info("  Saved %d feature vectors → %s", len(features), out_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
