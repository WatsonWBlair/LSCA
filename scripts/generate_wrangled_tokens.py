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
import queue
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _extra in ("/opt/homebrew/bin", "/usr/local/bin"):
    if _extra not in os.environ.get("PATH", "") and os.path.isdir(_extra):
        os.environ["PATH"] = _extra + os.pathsep + os.environ.get("PATH", "")

import cv2  # must import before torchvision to avoid Windows DLL conflict
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
    p.add_argument("--batch-size",    default=32, type=int,
                   help="Chunks per inference batch (default: 32)")
    p.add_argument("--num-workers",   default=6, type=int,
                   help="CPU preprocessing threads (default: 6)")
    return p.parse_args()


def make_backbone_tag(cfg) -> str:
    marlin_slug = cfg.streaming.marlin_model_name.replace("_", "-")
    wav_slug = cfg.streaming.wav2vec2_ctc_model.split("/")[-1]
    return f"{marlin_slug}__{wav_slug[:20]}"


def discover_triplets(
    wrangled_root: str, max_pairs: int | None
) -> list[tuple[Path, Path, Path]]:
    """Find all (mp4, wav, json) triplets under wrangled_root/*/."""
    session_dirs = sorted(d for d in Path(wrangled_root).iterdir() if d.is_dir())
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


def extract_convo_id(stem_name: str) -> str | None:
    """Extract conversation/interaction ID from stem name for dyadic pairing.

    Seamless: "I{interaction}_P{participant}" -> "I{interaction}"
    CANDOR:   "{conv_uuid}_{user_id}"         -> "{conv_uuid}"
    """
    if stem_name.startswith("I") and "_P" in stem_name:
        return stem_name.split("_P")[0]
    if "_" in stem_name:
        return stem_name.rsplit("_", 1)[0]
    return None


def build_partner_map(triplets: list) -> dict[str, str]:
    """Return {str(mp4_path): partner_stem_name} for each matched dyadic pair."""
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for mp4_path, _, _ in triplets:
        convo_id = extract_convo_id(mp4_path.stem)
        if convo_id:
            groups[(str(mp4_path.parent), convo_id)].append((mp4_path.stem, str(mp4_path)))
    partner_map: dict[str, str] = {}
    for members in groups.values():
        if len(members) == 2:
            (stem_a, path_a), (stem_b, path_b) = members
            partner_map[path_a] = stem_b
            partner_map[path_b] = stem_a
    return partner_map


def load_seamless_emotion_data(npz_path: Path) -> dict:
    """Load per-frame emotion arrays from a Seamless NPZ. Returns {} if unavailable."""
    if not npz_path.exists():
        return {}
    try:
        npz = np.load(str(npz_path), allow_pickle=False)
        result = {}
        for key in (
            "movement_v4:emotion_valence",
            "movement_v4:emotion_arousal",
            "movement_v4:emotion_scores",
            "movement_v4:FAUValue",
        ):
            if key in npz:
                result[key] = npz[key]
        return result
    except Exception as e:
        logger.debug("Could not load Seamless NPZ %s: %s", npz_path, e)
        return {}


def chunk_seamless_emotion(
    emotion_data: dict, start_sec: float, end_sec: float, fps: float
) -> tuple[dict, "np.ndarray | None"]:
    """Per-chunk averages of Seamless per-frame emotion arrays.

    Returns (record_fields_dict, fau_mean_or_None).
    """
    if not emotion_data:
        return {}, None
    start_frame = int(start_sec * fps)
    end_frame = max(start_frame + 1, int(end_sec * fps))
    result = {}

    val_arr = emotion_data.get("movement_v4:emotion_valence")
    if val_arr is not None and len(val_arr) > start_frame:
        window = val_arr[start_frame:end_frame]
        result["valence"] = float(np.mean(window)) if len(window) > 0 else 0.0

    aro_arr = emotion_data.get("movement_v4:emotion_arousal")
    if aro_arr is not None and len(aro_arr) > start_frame:
        window = aro_arr[start_frame:end_frame]
        result["arousal"] = float(np.mean(window)) if len(window) > 0 else 0.0

    scores_arr = emotion_data.get("movement_v4:emotion_scores")
    if scores_arr is not None and len(scores_arr) > start_frame:
        window = scores_arr[start_frame:end_frame]
        if len(window) > 0:
            result["emotion_probs"] = np.mean(window, axis=0).tolist()

    fau_mean = None
    fau_arr = emotion_data.get("movement_v4:FAUValue")
    if fau_arr is not None and len(fau_arr) > start_frame:
        window = fau_arr[start_frame:end_frame]
        if len(window) > 0:
            fau_mean = np.mean(window, axis=0).astype(np.float32)

    return result, fau_mean


def chunk_candor_emotion(avf_entries: list, start_sec: float, end_sec: float) -> dict:
    """Per-chunk averages from CANDOR audio_video_features entries."""
    if not avf_entries:
        return {}
    window = [
        e for e in avf_entries
        if start_sec <= float(e.get("time_sec", -1)) < end_sec
    ]
    if not window:
        return {}
    result = {}
    emotion_keys = sorted(k for k in window[0] if k.startswith("prob_face_"))
    if emotion_keys:
        try:
            result["emotion_probs"] = [
                float(np.mean([float(e.get(k, 0)) for e in window]))
                for k in emotion_keys
            ]
        except (ValueError, TypeError):
            pass
    for scalar_key in ("is_speaking", "gaze_on"):
        if scalar_key in window[0]:
            try:
                result[scalar_key] = float(
                    np.mean([float(e.get(scalar_key, 0)) for e in window])
                )
            except (ValueError, TypeError):
                pass
    return result


_SESSION_META_FIELDS = (
    "session_interaction_idx",
    "session_total_interactions",
    "session_id",
    "prompt_a",
    "prompt_b",
    "ipc_a",
    "ipc_b",
    "interaction_type",
)


def load_labels(json_path: Path) -> tuple[list, list, list, dict, dict, dict]:
    """Load VAD segments, transcript words, AVF entries, interaction metadata, segment labels, and base metadata from wrangled JSON."""
    with open(json_path) as f:
        data = json.load(f)
    vad_segs = data.get("metadata:vad", [])
    # Flatten transcript segments into a flat word list
    transcript_segs = []
    for seg in data.get("metadata:transcript", []):
        transcript_segs.extend(seg.get("words", []))
    avf_entries = data.get("metadata:audio_video_features", [])
    interaction_meta = {k: data[k] for k in _SESSION_META_FIELDS if k in data}
    segment_labels = data.get("metadata:labels", {})
    base_meta = {}
    if "id" in data:
        base_meta["source_id"] = data["id"]
    if "source" in data:
        base_meta["source"] = data["source"]
    survey = data.get("metadata:survey")
    if survey:
        base_meta["survey"] = survey
    return vad_segs, transcript_segs, avf_entries, interaction_meta, segment_labels, base_meta


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

    # Words overlapping this chunk (guard against None timestamps — see DATA_WRANGLING.md)
    overlapping_words = [
        w for w in word_list
        if w.get("start") is not None and w.get("end") is not None
        and w["start"] < end_sec and w["end"] > start_sec
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


def _preprocess_frame(bgr, marlin_size: int, im_mean: list, im_std: list) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (marlin_size, marlin_size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return TF.normalize(t, mean=im_mean, std=im_std)


def iter_chunks(mp4_path: Path, wav_path: Path, cfg):
    """Yield (chunk_frames, audio_chunk, video_fps, chunk_id, start_sec, end_sec).

    Streams video frames through a deque to avoid loading the entire video into RAM.
    Audio is loaded fully upfront (small, ~25 MB max).
    """
    from collections import deque

    waveform, _ = librosa.load(str(wav_path), sr=cfg.streaming.sample_rate, mono=True)

    marlin_size = cfg.streaming.marlin_size
    im_mean = list(cfg.streaming.imagenet_mean)
    im_std = list(cfg.streaming.imagenet_std)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        logger.warning("No frames from %s — skipping", mp4_path)
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    audio_win    = int(cfg.streaming.window_sec * cfg.streaming.sample_rate)
    audio_stride = int(cfg.streaming.stride_sec * cfg.streaming.sample_rate)
    video_win    = int(cfg.streaming.window_sec * video_fps)
    video_stride = int(cfg.streaming.stride_sec * video_fps)

    # Decode only every n_sub-th frame — uniform_sample picks evenly anyway,
    # so decoding all frames is wasteful. n_sub ≈ video_win / marlin_frames.
    n_sub = max(1, video_win // cfg.streaming.marlin_frames)
    video_win_sub    = max(1, video_win // n_sub)
    video_stride_sub = max(1, video_stride // n_sub)

    # Prime the deque with video_win_sub subsampled frames
    frame_buffer: deque[torch.Tensor] = deque()
    frame_counter = 0
    eof = False
    while len(frame_buffer) < video_win_sub:
        ret, bgr = cap.read()
        if not ret:
            eof = True
            break
        if frame_counter % n_sub == 0:
            frame_buffer.append(_preprocess_frame(bgr, marlin_size, im_mean, im_std))
        frame_counter += 1

    if not frame_buffer:
        cap.release()
        logger.warning("No frames from %s — skipping", mp4_path)
        return

    audio_start = 0
    chunk_id = 0

    while audio_start + audio_win <= len(waveform):
        if len(frame_buffer) < video_win_sub and eof:
            break

        start_sec = audio_start / cfg.streaming.sample_rate
        end_sec   = (audio_start + audio_win) / cfg.streaming.sample_rate
        yield (
            list(frame_buffer)[:video_win_sub],
            waveform[audio_start : audio_start + audio_win],
            video_fps,
            chunk_id,
            start_sec,
            end_sec,
        )

        # Advance: read real frames until video_stride_sub new subsampled frames collected
        new_sub = 0
        while new_sub < video_stride_sub:
            ret, bgr = cap.read()
            if not ret:
                eof = True
                break
            if frame_counter % n_sub == 0:
                frame_buffer.append(_preprocess_frame(bgr, marlin_size, im_mean, im_std))
                new_sub += 1
            frame_counter += 1
        for _ in range(min(video_stride_sub, len(frame_buffer))):
            frame_buffer.popleft()

        audio_start += audio_stride
        chunk_id += 1

    cap.release()


def _stem_worker(mp4_path, wav_path, json_path, partner_map, cfg, work_queue):
    """Per-stem CPU preprocessing worker. Pushes chunk items + sentinel to work_queue."""
    from encoding.pipelines.prosody import extract_prosody_raw

    stem_key = str(mp4_path)
    stem_name = mp4_path.stem
    count = 0

    try:
        vad_segs, word_list, avf_entries, interaction_meta, segment_labels, base_meta = (
            load_labels(json_path)
        )
        convo_id = extract_convo_id(stem_name)
        partner_stem = partner_map.get(stem_key)

        npz_path = mp4_path.with_suffix(".npz")
        seamless_emotion = load_seamless_emotion_data(npz_path)
        is_seamless = bool(seamless_emotion) or npz_path.exists()

        for chunk_frames, audio_chunk, fps_i, chunk_id, start_sec, end_sec in iter_chunks(
            mp4_path, wav_path, cfg
        ):
            prosody = extract_prosody_raw(
                audio_chunk,
                sr=cfg.streaming.sample_rate,
                d_prosody=cfg.latent.d_prosody,
            )
            lbl = chunk_labels(vad_segs, word_list, start_sec, end_sec)
            record = {
                "chunk_id": chunk_id,
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec, 4),
                **lbl,
                **interaction_meta,
                **base_meta,
            }
            if segment_labels:
                record["segment_labels"] = segment_labels
            if convo_id:
                record["convo_id"] = convo_id
            if partner_stem:
                record["partner_stem"] = partner_stem

            fau_mean = None
            if seamless_emotion:
                emotion_fields, fau_mean = chunk_seamless_emotion(
                    seamless_emotion, start_sec, end_sec, fps_i
                )
                record.update(emotion_fields)
            elif avf_entries:
                record.update(chunk_candor_emotion(avf_entries, start_sec, end_sec))

            work_queue.put({
                "stem_key": stem_key,
                "frame_list": chunk_frames,
                "audio_chunk": audio_chunk,
                "fps": fps_i,
                "prosody": prosody,
                "record": record,
                "fau_mean": fau_mean,
                "is_seamless": is_seamless,
            })
            count += 1

    except Exception as e:
        logger.error("Worker error for %s: %s", stem_key, e)
    finally:
        work_queue.put({"stem_key": stem_key, "done": True, "count": count})


def _flush_batch(batch_buf, stem_states, marlin_model, temporal_pool,
                 wav2vec2_ctc, wav2vec2_proc, cfg):
    """Run GPU inference on batch_buf and append results to per-stem state."""
    from encoding.pipelines.video import batch_video_pipeline
    from encoding.pipelines.phoneme import pad_phonemes, batch_phoneme_pipeline

    if not batch_buf:
        return

    frame_lists = [item["frame_list"] for item in batch_buf]
    audio_chunks = [item["audio_chunk"] for item in batch_buf]
    fps_val = batch_buf[0]["fps"]

    try:
        v_batch = batch_video_pipeline(frame_lists, fps_val, marlin_model, temporal_pool, cfg)
        ph_batch = batch_phoneme_pipeline(audio_chunks, wav2vec2_ctc, wav2vec2_proc, cfg)
    except Exception as e:
        logger.warning("Batch inference failed: %s — skipping %d items", e, len(batch_buf))
        batch_buf.clear()
        return

    for i, item in enumerate(batch_buf):
        sk = item["stem_key"]
        state = stem_states[sk]
        try:
            v = v_batch[i]
            embs, labels, mask = ph_batch[i]
            padded_embs, padded_labels, padded_mask = pad_phonemes(
                embs, labels, mask,
                cfg.latent.max_phones, cfg.latent.d_phoneme,
            )
            # Open jsonl lazily on first chunk for this stem
            if state["jsonl_fh"] is None:
                state["out_dir"].mkdir(parents=True, exist_ok=True)
                state["jsonl_fh"] = open(state["out_dir"] / "chunks.jsonl", "w")

            # Atomic write: numpy lists + jsonl in same try block (Row-N invariant)
            state["v_rows"].append(v.detach().cpu().numpy())
            state["ph_rows"].append(padded_embs.detach().cpu().numpy())
            state["ph_label_rows"].append(padded_labels.detach().cpu().numpy())
            state["ph_mask_rows"].append(padded_mask.detach().cpu().numpy())
            state["p_rows"].append(item["prosody"])

            if item["is_seamless"]:
                fau_mean = item["fau_mean"]
                state["fau_rows"].append(
                    fau_mean if fau_mean is not None else np.zeros(24, dtype=np.float32)
                )

            state["jsonl_fh"].write(json.dumps(item["record"]) + "\n")
            state["processed"] += 1

        except Exception as e:
            logger.warning("Item error in %s chunk %d: %s",
                           sk, item["record"].get("chunk_id", "?"), e)

    batch_buf.clear()


def _try_save_complete(stem_states, sessions_processed):
    """Save arrays and close jsonl for any stem where processed == expected.

    Returns the number of chunks written across all newly completed stems.
    """
    done_keys = []
    total = 0
    for sk, state in stem_states.items():
        if state["expected"] is None or state["processed"] != state["expected"]:
            continue
        if state["processed"] == 0:
            logger.warning(
                "No chunks from %s/%s — skipping save",
                state["session_name"], state["stem_name"],
            )
        else:
            out_dir = state["out_dir"]
            np.save(str(out_dir / "v_raw.npy"),     np.stack(state["v_rows"]))
            np.save(str(out_dir / "ph_raw.npy"),    np.stack(state["ph_rows"]))
            np.save(str(out_dir / "ph_labels.npy"), np.stack(state["ph_label_rows"]))
            np.save(str(out_dir / "ph_mask.npy"),   np.stack(state["ph_mask_rows"]))
            np.save(str(out_dir / "p_raw.npy"),     np.stack(state["p_rows"]))
            if state["is_seamless"] and state["fau_rows"]:
                np.save(str(out_dir / "fau.npy"), np.stack(state["fau_rows"]))
            logger.info("  → %d chunks saved to %s", state["processed"], out_dir)
            sessions_processed.append(f"{state['session_name']}/{state['stem_name']}")
            total += state["processed"]
        if state["jsonl_fh"] is not None:
            state["jsonl_fh"].close()
        done_keys.append(sk)
    for sk in done_keys:
        del stem_states[sk]
    return total


def main():
    args = parse_args()

    from encoding.config import CAMELSConfig
    from encoding.models.loader import load_all_models
    from encoding.adapters.registry import build_adapters

    cfg = CAMELSConfig()
    backbone_tag = make_backbone_tag(cfg)
    tag_root = Path(args.pregen_root) / backbone_tag

    logger.info("Loading frozen models (device=%s, half=True, emformer=False) ...", args.device)
    models = load_all_models(cfg, device=args.device, load_emformer=False, half=True)

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

    partner_map = build_partner_map(triplets)
    logger.info("Dyadic partner map: %d matched stems", len(partner_map))

    # Build lookup: stem_key -> (session_name, stem_name, out_dir)
    triplet_lookup: dict[str, tuple[str, str, Path]] = {}
    for mp4_path, _, _ in triplets:
        session_name = mp4_path.parent.name
        stem_name = mp4_path.stem
        out_dir = tag_root / session_name / stem_name
        triplet_lookup[str(mp4_path)] = (session_name, stem_name, out_dir)

    # Filter already-processed stems before submitting workers
    skipped_existing = 0
    non_skipped_triplets = []
    for tri_idx, (mp4_path, wav_path, json_path) in enumerate(triplets):
        _, _, out_dir = triplet_lookup[str(mp4_path)]
        if (out_dir / "v_raw.npy").exists():
            session_name, stem_name, _ = triplet_lookup[str(mp4_path)]
            logger.info("Skipping %d/%d: %s/%s (already processed)",
                        tri_idx + 1, len(triplets), session_name, stem_name)
            skipped_existing += 1
        else:
            non_skipped_triplets.append((mp4_path, wav_path, json_path))

    sessions_processed: list[str] = []
    total_chunks = 0

    if not non_skipped_triplets:
        logger.info("All stems already processed.")
    else:
        work_queue: queue.Queue = queue.Queue(
            maxsize=args.num_workers * args.batch_size * 2
        )
        # stem_states: keyed by str(mp4_path), lazily initialized on first data item
        stem_states: dict[str, dict] = {}
        pending_sentinels = {str(mp4) for mp4, _, _ in non_skipped_triplets}
        batch_buf: list[dict] = []

        logger.info(
            "Submitting %d stems to %d worker threads (batch_size=%d) ...",
            len(non_skipped_triplets), args.num_workers, args.batch_size,
        )

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for mp4_path, wav_path, json_path in non_skipped_triplets:
                executor.submit(
                    _stem_worker, mp4_path, wav_path, json_path,
                    partner_map, cfg, work_queue,
                )

            # Consumer loop: drain work_queue, batch GPU inference, save completed stems
            while pending_sentinels or batch_buf:
                try:
                    item = work_queue.get(timeout=5.0)
                except queue.Empty:
                    if batch_buf:
                        _flush_batch(
                            batch_buf, stem_states,
                            marlin_model, temporal_pool,
                            wav2vec2_ctc, wav2vec2_proc, cfg,
                        )
                        total_chunks += _try_save_complete(stem_states, sessions_processed)
                    if not pending_sentinels:
                        break
                    continue

                sk = item["stem_key"]

                if item.get("done"):
                    stem_states.setdefault(sk, {
                        "v_rows": [], "ph_rows": [], "ph_label_rows": [],
                        "ph_mask_rows": [], "p_rows": [], "fau_rows": [],
                        "jsonl_fh": None,
                        "expected": None,
                        "processed": 0,
                        "out_dir": triplet_lookup[sk][2],
                        "session_name": triplet_lookup[sk][0],
                        "stem_name": triplet_lookup[sk][1],
                        "is_seamless": False,
                    })
                    stem_states[sk]["expected"] = item["count"]
                    pending_sentinels.discard(sk)
                    total_chunks += _try_save_complete(stem_states, sessions_processed)
                else:
                    # Lazily init state on first data item for this stem
                    if sk not in stem_states:
                        session_name, stem_name, out_dir = triplet_lookup[sk]
                        stem_states[sk] = {
                            "v_rows": [], "ph_rows": [], "ph_label_rows": [],
                            "ph_mask_rows": [], "p_rows": [], "fau_rows": [],
                            "jsonl_fh": None,
                            "expected": None,
                            "processed": 0,
                            "out_dir": out_dir,
                            "session_name": session_name,
                            "stem_name": stem_name,
                            "is_seamless": item["is_seamless"],
                        }
                    batch_buf.append(item)
                    if len(batch_buf) >= args.batch_size:
                        _flush_batch(
                            batch_buf, stem_states,
                            marlin_model, temporal_pool,
                            wav2vec2_ctc, wav2vec2_proc, cfg,
                        )
                        total_chunks += _try_save_complete(stem_states, sessions_processed)

    if skipped_existing:
        logger.info("Skipped %d already-processed stem(s)", skipped_existing)

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
