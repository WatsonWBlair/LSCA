#!/usr/bin/env python3
# scripts/generate_wrangled_tokens.py
# Pregenerate backbone tokens from wrangled Seamless Interaction sessions.
#
# Reads datasets/wrangled/S*/  — expects .mp4, .wav, .json per stem.
# Writes datasets/pregenerated/{backbone_tag}/S*/stem/ with:
#   v_raw.npy       (N, 768)       MARLIN last latent per chunk
#   ph_raw.npy      (N, 50, 768)   wav2vec2 phoneme embeddings
#   ph_labels.npy   (N, 50)        CTC phoneme class IDs
#   ph_mask.npy     (N, 50)        1=real, 0=padding
#   p_raw.npy       (N, 22)        raw librosa features (NOT z-scored)
#   chunks.jsonl                   N lines, one JSON object per chunk
#
# Usage:
#   python scripts/generate_wrangled_tokens.py \
#       --wrangled-root datasets/wrangled \
#       --pregen-root   datasets/pregenerated \
#       --device        cpu \
#       --max-pairs     2

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _extra in ("/opt/homebrew/bin", "/usr/local/bin"):
    if _extra not in os.environ.get("PATH", "") and os.path.isdir(_extra):
        os.environ["PATH"] = _extra + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
import torchvision.transforms.functional as TF
import librosa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_wrangled_tokens")


def parse_args():
    p = argparse.ArgumentParser(
        description="Pregenerate backbone tokens from wrangled sessions"
    )
    p.add_argument("--wrangled-root", default="datasets/wrangled",
                   help="Root of wrangled data (default: datasets/wrangled)")
    p.add_argument("--pregen-root",   default="datasets/pregenerated",
                   help="Output root (default: datasets/pregenerated)")
    p.add_argument("--device",        default="cpu",
                   help="cpu | cuda | mps (default: cpu)")
    p.add_argument("--max-pairs",     default=None, type=int,
                   help="Limit pairs for debugging (default: None)")
    return p.parse_args()


def make_backbone_tag(cfg) -> str:
    marlin_slug = cfg.streaming.marlin_model_name.replace("_", "-")
    wav_slug = cfg.streaming.wav2vec2_ctc_model.split("/")[-1]
    return f"{marlin_slug}__{wav_slug[:20]}"


def discover_triplets(
    wrangled_root: str, max_pairs: int | None
) -> list[tuple[Path, Path, Path]]:
    """Find all (mp4, wav, json) triplets under wrangled_root/S*/."""
    session_dirs = sorted(Path(wrangled_root).glob("S*/"))
    triplets = []
    skipped = 0

    for session_dir in session_dirs:
        for mp4_path in sorted(session_dir.glob("*.mp4")):
            stem = mp4_path.stem
            wav_path = mp4_path.with_suffix(".wav")
            json_path = mp4_path.with_suffix(".json")
            if not wav_path.exists():
                logger.warning("No .wav for %s — skipping", mp4_path.name)
                skipped += 1
                continue
            if not json_path.exists():
                logger.warning("No .json for %s — skipping", mp4_path.name)
                skipped += 1
                continue
            triplets.append((mp4_path, wav_path, json_path))

    if skipped:
        logger.warning("Skipped %d stems with missing files", skipped)

    if max_pairs is not None:
        triplets = triplets[:max_pairs]

    logger.info("Found %d matched triplets", len(triplets))
    return triplets


def load_labels(json_path: Path) -> tuple[list, list]:
    """Load VAD segments and transcript word entries from wrangled JSON."""
    with open(json_path) as f:
        data = json.load(f)
    vad_segs = data.get("metadata:vad", [])
    # Flatten transcript segments into a flat word list
    transcript_segs = []
    for seg in data.get("metadata:transcript", []):
        transcript_segs.extend(seg.get("words", []))
    return vad_segs, transcript_segs


def chunk_labels(
    vad_segs: list,
    word_list: list,
    start_sec: float,
    end_sec: float,
) -> dict:
    """Derive per-chunk labels from time-aligned VAD and transcript data."""
    window_sec = end_sec - start_sec

    # VAD coverage
    vad_overlap = 0.0
    for seg in vad_segs:
        seg_start = seg["start"]
        seg_end = seg["end"]
        overlap = max(0.0, min(seg_end, end_sec) - max(seg_start, start_sec))
        vad_overlap += overlap
    vad_coverage = min(1.0, vad_overlap / window_sec) if window_sec > 0 else 0.0
    vad_active = vad_coverage > 0.0

    # Words overlapping this chunk
    overlapping_words = [
        w for w in word_list
        if w["start"] < end_sec and w["end"] > start_sec
    ]
    transcript = " ".join(w["word"] for w in overlapping_words)
    asr_confidence = (
        float(np.mean([w["score"] for w in overlapping_words]))
        if overlapping_words else 0.0
    )

    return {
        "vad_active": vad_active,
        "vad_coverage": round(vad_coverage, 4),
        "words": overlapping_words,
        "transcript": transcript,
        "asr_confidence": round(asr_confidence, 4),
    }


def load_all_frames(mp4_path: Path, cfg) -> tuple[list[torch.Tensor], float]:
    import cv2
    marlin_size = cfg.streaming.marlin_size
    im_mean = list(cfg.streaming.imagenet_mean)
    im_std = list(cfg.streaming.imagenet_std)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        logger.warning("Cannot open %s", mp4_path)
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


def iter_chunks(mp4_path: Path, wav_path: Path, cfg):
    """Yield (chunk_frames, audio_chunk, video_fps, chunk_id, start_sec, end_sec)."""
    waveform, _ = librosa.load(str(wav_path), sr=cfg.streaming.sample_rate, mono=True)
    frames, video_fps = load_all_frames(mp4_path, cfg)

    if not frames:
        logger.warning("No frames from %s — skipping", mp4_path)
        return

    audio_win    = int(cfg.streaming.window_sec * cfg.streaming.sample_rate)
    audio_stride = int(cfg.streaming.stride_sec * cfg.streaming.sample_rate)
    video_win    = int(cfg.streaming.window_sec * video_fps)
    video_stride = int(cfg.streaming.stride_sec * video_fps)

    audio_start = 0
    frame_start = 0
    chunk_id = 0

    while audio_start + audio_win <= len(waveform):
        start_sec = audio_start / cfg.streaming.sample_rate
        end_sec   = (audio_start + audio_win) / cfg.streaming.sample_rate
        yield (
            frames[frame_start : frame_start + video_win],
            waveform[audio_start : audio_start + audio_win],
            video_fps,
            chunk_id,
            start_sec,
            end_sec,
        )
        audio_start += audio_stride
        frame_start += video_stride
        chunk_id += 1


def main():
    args = parse_args()

    from encoding.config import CAMELSConfig
    from encoding.models.loader import load_all_models
    from encoding.adapters.registry import build_adapters
    from encoding.pipelines.video import video_pipeline
    from encoding.pipelines.phoneme import phoneme_pipeline, pad_phonemes
    from encoding.pipelines.prosody import extract_prosody_raw

    cfg = CAMELSConfig()
    backbone_tag = make_backbone_tag(cfg)
    tag_root = Path(args.pregen_root) / backbone_tag

    logger.info("Loading frozen models (device=%s) ...", args.device)
    models = load_all_models(cfg, device=args.device)

    if "num_phoneme_classes" in models:
        cfg.latent.num_phoneme_classes = models["num_phoneme_classes"]
        logger.info("Detected %d phoneme classes", cfg.latent.num_phoneme_classes)

    marlin_model  = models["marlin"]
    wav2vec2_ctc  = models["wav2vec2_ctc"]
    wav2vec2_proc = models["wav2vec2_processor"]

    adapters = build_adapters(cfg)
    temporal_pool = adapters["temporal_pool"]

    triplets = discover_triplets(args.wrangled_root, args.max_pairs)
    if not triplets:
        logger.error("No triplets found under %s", args.wrangled_root)
        return

    sessions_processed: list[str] = []
    total_chunks = 0

    for tri_idx, (mp4_path, wav_path, json_path) in enumerate(triplets):
        session_name = mp4_path.parent.name       # e.g. S0172
        stem_name    = mp4_path.stem              # e.g. I00001226_P1316
        out_dir = tag_root / session_name / stem_name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Processing %d/%d: %s/%s",
            tri_idx + 1, len(triplets), session_name, stem_name,
        )

        vad_segs, word_list = load_labels(json_path)

        v_rows, ph_rows, ph_label_rows, ph_mask_rows, p_rows = [], [], [], [], []
        chunk_records: list[dict] = []
        pair_chunks = 0

        jsonl_path = out_dir / "chunks.jsonl"
        with open(jsonl_path, "w") as jsonl_fh:
            for chunk_frames, audio_chunk, fps, chunk_id, start_sec, end_sec in iter_chunks(
                mp4_path, wav_path, cfg
            ):
                try:
                    v = video_pipeline(chunk_frames, fps, marlin_model, temporal_pool, cfg)
                    embs, labels, mask, _ = phoneme_pipeline(
                        audio_chunk, wav2vec2_ctc, wav2vec2_proc, cfg,
                    )
                    padded_embs, padded_labels, padded_mask = pad_phonemes(
                        embs, labels, mask,
                        cfg.latent.max_phones, cfg.latent.d_phoneme,
                    )
                    p = extract_prosody_raw(
                        audio_chunk,
                        sr=cfg.streaming.sample_rate,
                        d_prosody=cfg.latent.d_prosody,
                    )

                    v_rows.append(v.detach().cpu().numpy())
                    ph_rows.append(padded_embs.detach().cpu().numpy())
                    ph_label_rows.append(padded_labels.detach().cpu().numpy())
                    ph_mask_rows.append(padded_mask.detach().cpu().numpy())
                    p_rows.append(p)

                    lbl = chunk_labels(vad_segs, word_list, start_sec, end_sec)
                    record = {
                        "chunk_id": chunk_id,
                        "start_sec": round(start_sec, 4),
                        "end_sec": round(end_sec, 4),
                        **lbl,
                    }
                    jsonl_fh.write(json.dumps(record) + "\n")
                    pair_chunks += 1
                    total_chunks += 1

                except Exception as e:
                    logger.warning("Chunk %d error in %s: %s", chunk_id, stem_name, e)

        if pair_chunks == 0:
            logger.warning("No chunks from %s/%s — skipping save", session_name, stem_name)
            continue

        np.save(str(out_dir / "v_raw.npy"),      np.stack(v_rows))
        np.save(str(out_dir / "ph_raw.npy"),     np.stack(ph_rows))
        np.save(str(out_dir / "ph_labels.npy"),  np.stack(ph_label_rows))
        np.save(str(out_dir / "ph_mask.npy"),    np.stack(ph_mask_rows))
        np.save(str(out_dir / "p_raw.npy"),      np.stack(p_rows))

        logger.info("  → %d chunks saved to %s", pair_chunks, out_dir)
        sessions_processed.append(f"{session_name}/{stem_name}")

    # Write config.json at backbone_tag root
    config = {
        "backbone_tag": backbone_tag,
        "marlin_model_name": cfg.streaming.marlin_model_name,
        "wav2vec2_ctc_model": cfg.streaming.wav2vec2_ctc_model,
        "window_sec": cfg.streaming.window_sec,
        "stride_sec": cfg.streaming.stride_sec,
        "d_latent": cfg.latent.d_latent,
        "max_phones": cfg.latent.max_phones,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sessions": sessions_processed,
        "total_chunks": total_chunks,
    }
    config_path = tag_root / "config.json"
    tag_root.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(
        "Done. %d chunks from %d stems → %s",
        total_chunks, len(sessions_processed), tag_root,
    )


if __name__ == "__main__":
    main()
