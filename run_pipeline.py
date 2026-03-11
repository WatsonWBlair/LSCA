#!/usr/bin/env python3
# run_pipeline.py
# Live streaming entry point for the CAMELS pipeline (v8.1).
# 3 modalities: video, phoneme, prosody + transcript utility.
#
# Usage:
#   python run_pipeline.py [--device cuda] [--output-dir ./run_output]
#                          [--camera 0] [--checkpoint checkpoints/stage_c_epoch020.pt]

import argparse
import json
import logging
import os
import signal
import sys
import time

# Ensure Homebrew and common tool paths are visible (needed for ffprobe/ffmpeg)
for _extra in ("/opt/homebrew/bin", "/usr/local/bin"):
    if _extra not in os.environ.get("PATH", "") and os.path.isdir(_extra):
        os.environ["PATH"] = _extra + os.pathsep + os.environ.get("PATH", "")

import sounddevice as sd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="CAMELS v8.1 live streaming pipeline")
    p.add_argument("--device",        default="cpu",        help="cpu | cuda | mps")
    p.add_argument("--output-dir",    default="run_output", help="Directory for .npy output files")
    p.add_argument("--camera",        default=0, type=int,  help="Camera index (default: 0)")
    p.add_argument("--checkpoint",    default=None,         help="Path to .pt adapter checkpoint")
    p.add_argument("--prosody-stats", default="prosody_stats.json",
                   help="Path to prosody_stats.json (from training)")
    p.add_argument("--d-latent",      default=768, type=int, help="Latent dimension")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from encoding.config import CAMELSConfig, LatentConfig
    from encoding.models.loader import load_all_models
    from encoding.adapters.registry import build_adapters, load_adapters
    from encoding.streaming.buffers import AudioBuffer, FrameBuffer
    from encoding.streaming.scheduler import FixedStrideScheduler
    from encoding.streaming.dispatch import run_all_pipelines, handle_silent_chunk
    from encoding.pipelines.transcript import EmformerASR

    cfg = CAMELSConfig(
        latent=LatentConfig(d_latent=args.d_latent),
    )

    # ── 1. Load frozen models ─────────────────────────────────────────
    logger.info("Loading frozen models (device=%s) ...", args.device)
    models = load_all_models(cfg, device=args.device)

    # Auto-detect phoneme classes
    if "num_phoneme_classes" in models:
        cfg.latent.num_phoneme_classes = models["num_phoneme_classes"]

    # ── 2. Load / build adapters ──────────────────────────────────────
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info("Loading adapters from checkpoint: %s", args.checkpoint)
        adapters = build_adapters(cfg)
        load_adapters(adapters, args.checkpoint, device=args.device)
    else:
        logger.info("No checkpoint provided — using untrained adapters")
        adapters = build_adapters(cfg)

    for mod in adapters.values():
        mod.to(args.device).eval()

    # ── 3. Load prosody stats ─────────────────────────────────────────
    prosody_stats = None
    if os.path.exists(args.prosody_stats):
        with open(args.prosody_stats) as f:
            prosody_stats = json.load(f)
        logger.info("Loaded prosody stats from %s", args.prosody_stats)
    else:
        logger.warning("prosody_stats.json not found — prosody will be un-normalized")

    # ── 4. Start AudioBuffer + sounddevice stream ─────────────────────
    audio_buffer = AudioBuffer(cfg)
    stream = sd.InputStream(
        samplerate=cfg.streaming.sample_rate,
        channels=1,
        dtype="float32",
        callback=audio_buffer.callback,
    )

    # ── 5. Start FrameBuffer (camera + MediaPipe face detect) ─────────
    frame_buffer = FrameBuffer(cfg)
    frame_buffer.start_capture(camera_index=args.camera)

    # ── 6. Start Emformer ASR ─────────────────────────────────────────
    asr = EmformerASR(models, output_dir=args.output_dir)
    asr.start(audio_buffer._buf)

    # ── 7. Set up dispatcher callbacks ────────────────────────────────
    chunk_registry: list = []

    def on_chunk(chunk_id, start_sec, end_sec):
        run_all_pipelines(
            chunk_id=chunk_id,
            start_sec=start_sec,
            end_sec=end_sec,
            frame_buffer=frame_buffer,
            audio_buffer=audio_buffer,
            asr=asr,
            models=models,
            adapters=adapters,
            prosody_stats=prosody_stats,
            output_dir=args.output_dir,
            chunk_registry=chunk_registry,
            cfg=cfg,
        )

    def on_silent(chunk_id, start_sec, end_sec):
        handle_silent_chunk(
            chunk_id=chunk_id,
            start_sec=start_sec,
            end_sec=end_sec,
            output_dir=args.output_dir,
            chunk_registry=chunk_registry,
            cfg=cfg,
        )

    # ── 8. Start FixedStrideScheduler ─────────────────────────────────
    scheduler = FixedStrideScheduler(
        audio_buffer=audio_buffer,
        on_chunk=on_chunk,
        on_silent=on_silent,
        cfg=cfg,
    )

    # ── 9. Graceful shutdown ──────────────────────────────────────────
    def shutdown(sig=None, frame=None):
        logger.info("\nShutting down CAMELS pipeline ...")
        scheduler.stop()
        asr.stop()
        frame_buffer.stop()
        stream.stop()
        stream.close()
        logger.info(
            "Session complete: %d chunks written to %s",
            scheduler.chunk_id, args.output_dir,
        )
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── 10. Start streaming ───────────────────────────────────────────
    logger.info("Starting CAMELS v8.1 live streaming — press Ctrl+C to stop")
    logger.info("Output dir: %s", os.path.abspath(args.output_dir))
    logger.info("Modalities: video=%s, phoneme=%s, prosody=%s",
                cfg.modality.video_enabled, cfg.modality.phoneme_enabled, cfg.modality.prosody_enabled)
    stream.start()
    scheduler.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
