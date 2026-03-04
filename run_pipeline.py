#!/usr/bin/env python3
# run_pipeline.py
# Live streaming entry point for the CAMELS pipeline.
# Starts all threads (audio buffer, frame buffer, Emformer ASR, scheduler)
# and runs until interrupted.
#
# Usage:
#   python run_pipeline.py [--device cuda] [--output-dir ./run_output]
#                          [--camera 0] [--checkpoint checkpoints/stage_c_epoch020.pt]
#                          [--language eng_Latn]

import argparse
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
    p = argparse.ArgumentParser(description="CAMELS live streaming pipeline")
    p.add_argument("--device",      default="cpu",   help="cpu | cuda | mps")
    p.add_argument("--output-dir",  default="run_output", help="Directory for .npy output files")
    p.add_argument("--camera",      default=0, type=int, help="Camera index (default: 0)")
    p.add_argument("--checkpoint",  default=None, help="Path to .pt adapter checkpoint")
    p.add_argument("--language",    default="eng_Latn", help="SONAR language code")
    p.add_argument("--prosody-stats", default="prosody_stats.json",
                   help="Path to prosody_stats.json (from training)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load frozen models ─────────────────────────────────────────────
    logger.info("Loading frozen models (device=%s) ...", args.device)
    from models.model_loader import load_all_models
    models = load_all_models(device=args.device)

    # ── 2. Load / build adapters ──────────────────────────────────────────
    from pipeline.adapters import build_adapters, load_adapters
    from pipeline.config   import D_AUDIO

    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info("Loading adapters from checkpoint: %s", args.checkpoint)
        adapters = load_adapters(args.checkpoint, d_audio=D_AUDIO, device=args.device)
    else:
        logger.info("No checkpoint provided — using untrained adapters")
        adapters = build_adapters(d_audio=D_AUDIO)
        for mod in adapters.values():
            mod.to(args.device).eval()

    # ── 3. Load prosody stats ─────────────────────────────────────────────
    from pipeline.prosody_pipeline import load_prosody_stats
    prosody_stats = None
    if os.path.exists(args.prosody_stats):
        prosody_stats = load_prosody_stats(args.prosody_stats)
    else:
        logger.warning("prosody_stats.json not found — prosody will be un-normalized")

    # ── 4. Start AudioBuffer + sounddevice stream ─────────────────────────
    from pipeline.buffers  import AudioBuffer, FrameBuffer
    from pipeline.config   import SAMPLE_RATE

    audio_buffer = AudioBuffer()
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_buffer.callback,
    )

    # ── 5. Start FrameBuffer (camera + MediaPipe face detect) ─────────────
    frame_buffer = FrameBuffer()
    frame_buffer.start_capture(camera_index=args.camera)

    # ── 6. Start Emformer ASR background thread ───────────────────────────
    from pipeline.text_pipeline import EmformerASR
    asr = EmformerASR(models, output_dir=args.output_dir)
    asr.start(audio_buffer._buf)

    # ── 7. Set up dispatcher callbacks ────────────────────────────────────
    from pipeline.dispatch   import run_all_pipelines, handle_silent_chunk
    chunk_registry: list = []

    def on_chunk(chunk_id, start_sec, end_sec):
        run_all_pipelines(
            chunk_id     = chunk_id,
            start_sec    = start_sec,
            end_sec      = end_sec,
            frame_buffer = frame_buffer,
            audio_buffer = audio_buffer,
            asr          = asr,
            models       = models,
            adapters     = adapters,
            prosody_stats = prosody_stats,
            output_dir   = args.output_dir,
            chunk_registry = chunk_registry,
            language     = args.language,
        )

    def on_silent(chunk_id, start_sec, end_sec):
        handle_silent_chunk(
            chunk_id       = chunk_id,
            start_sec      = start_sec,
            end_sec        = end_sec,
            output_dir     = args.output_dir,
            chunk_registry = chunk_registry,
        )

    # ── 8. Start FixedStrideScheduler ─────────────────────────────────────
    from pipeline.scheduler import FixedStrideScheduler
    scheduler = FixedStrideScheduler(
        audio_buffer = audio_buffer,
        on_chunk     = on_chunk,
        on_silent    = on_silent,
    )

    # ── 9. Graceful shutdown ──────────────────────────────────────────────
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

    # ── 10. Start streaming ───────────────────────────────────────────────
    logger.info("Starting live streaming — press Ctrl+C to stop")
    logger.info("Output dir: %s", os.path.abspath(args.output_dir))
    stream.start()
    scheduler.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
