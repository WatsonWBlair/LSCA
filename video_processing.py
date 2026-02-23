#VIDEO PROCESSOR — FACE CROP + AUDIO EXTRACTION + TIMESTAMPS
import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
import json
from pathlib import Path
import shutil 

VIDEO_PATH = Path("/Users/katyasha1/capstone/data/naturalistic/dev/0001/0003/V00_S1823_I00000265_P0115.mp4")
OUTPUT_DIR = Path("/Users/katyasha1/capstone/data/processed_frames")


# Video settings
TARGET_FPS    = 15       # downsample from original (usually 30) to 15
FRAME_SIZE    = (224, 224)  # width x height — standard for vision models
FACE_PADDING  = 0.4      # 0.4 = 40% extra space around face (gets chest/shoulders)

# Audio settings
AUDIO_SAMPLE_RATE = 16000   # 16kHz — what Whisper expects

# =============================================================
#  EXTRACT AUDIO
# =============================================================

def extract_audio(video_path, output_dir, sample_rate=16000):
    """
    Pulls the audio stream out of the video file and saves it as a .wav file.

    Parameters:
      video_path  : str or Path → path to your .mp4 file
      output_dir  : str or Path → folder to save the .wav file
      sample_rate : 16000 Hz (Whisper standard)

    Returns:
      Path to the saved .wav file (or None if ffmpeg fails)
    """

    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "audio.wav")

    print(f"\n[AUDIO] Extracting audio → {audio_path}")

    # ffmpeg command:
    cmd = [
        "ffmpeg", "-y",           # overwrite existing audio.wav
        "-i", str(video_path),    # input video
        "-vn",                    # remove video stream
        "-ac", "1",               # mono
        "-ar", str(sample_rate),  # sample rate
        "-f", "wav",              # output format
        audio_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # If ffmpeg failed
    if result.returncode != 0:
        print("[AUDIO] ERROR: ffmpeg failed.")
        print(result.stderr[-500:])  # last 500 chars for debugging
        return None

    # Confirm success
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"[AUDIO] ✓ Saved audio ({size_mb:.2f} MB)")
    print(f"[AUDIO]   Sample rate: {sample_rate} Hz")
    print(f"[AUDIO]   Channels: 1 (mono)")

    return audio_path

# =============================================================
#  GET VIDEO PROPERTIES
# =============================================================

def get_video_info(video_path):
    """
    Reads basic properties of the video before processing.
    Always do this first — you need fps to calculate timestamps.

    Returns:
        dict with:
            - fps
            - total_frames
            - width
            - height
            - duration_sec
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[VIDEO] ERROR: Cannot open {video_path}")
        print("         Make sure the file path is correct.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # avoid division by zero
    duration_sec = total_frames / fps if fps > 0 else 0.0

    info = {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }

    cap.release()

    print(f"\n[VIDEO] File: {video_path}")
    print(f"[VIDEO] Resolution: {width}x{height}")
    print(f"[VIDEO] FPS: {fps}")
    print(f"[VIDEO] Total frames: {total_frames}")
    print(f"[VIDEO] Duration: {duration_sec:.2f} seconds")

    return info


# =============================================================
# FACE DETECTION SETUP
# =============================================================
#
# MediaPipe is Google's lightweight face detection library.
# It can run in real-time on a laptop (no GPU needed).
#
# How it works:
#   1. You give it a frame (a numpy array of pixels)
#   2. It returns bounding boxes around detected faces
#   3. Each bounding box is given as RELATIVE coordinates:
#      x=0.5, y=0.5 means the center of the image
#      x=0.0 means left edge, x=1.0 means right edge
#   4. You multiply by actual width/height to get pixel coords
#
# We create the detector ONCE (expensive)
# and reuse it per frame (cheap).
# -------------------------------------------------------------

def create_face_detector():
    """
    Creates a MediaPipe face detector.
    Call this once before your frame loop.
    """
    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(
        model_selection=1,             # 1 = long-range detector
        min_detection_confidence=0.3   # lower threshold → detect more faces
    )
    
    print("\n[FACE] Face detector ready (MediaPipe, long-range, conf=0.3)")
    return detector


# =============================================================
#  PROCESS A SINGLE FRAME
# =============================================================

def process_frame(frame, frame_number, original_fps, face_detector,
                  target_size=(224, 224), padding=0.4):
    """
    Takes one raw video frame and returns the processed face crop.

    This is the heart of the pipeline. In real-time deployment,
    this function runs once per incoming video frame.

    Parameters:
      frame         : numpy array, shape (H, W, 3), values 0-255 (BGR)
      frame_number  : which frame this is (0, 1, 2, ...)
      original_fps  : frames per second of the source video
      face_detector : MediaPipe detector object
      target_size   : (width, height) to resize crop to
      padding       : extra space around face bbox

    Returns a dict (or None if no face found):
      "frame_number"  : int
      "timestamp_sec" : float — exact time this frame occurred
      "timestamp_ms"  : int   — same in milliseconds (useful for alignment)
      "face_crop"     : numpy array (H, W) grayscale, or None
      "bbox_pixels"   : (x1, y1, x2, y2) face bounding box in pixel coords
      "face_found"    : bool
    """

    # ── TIMESTAMP ─────────────────────────────────────────────
    # frame_number / fps = seconds from start of video
    # Guard against fps = 0
    if original_fps and original_fps > 0:
        timestamp_sec = frame_number / original_fps
    else:
        timestamp_sec = 0.0
    timestamp_ms  = int(timestamp_sec * 1000)  # convert to milliseconds

    # ── FACE DETECTION ────────────────────────────────────────
    h, w = frame.shape[:2]  # actual pixel height and width

    # MediaPipe needs RGB, but OpenCV loads as BGR
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(frame_rgb)

    # No result object or no detections
    if not results or not results.detections:
        # No face detected — return record with face_found=False
        return {
            "frame_number":  frame_number,
            "timestamp_sec": timestamp_sec,
            "timestamp_ms":  timestamp_ms,
            "face_crop":     None,
            "bbox_pixels":   None,
            "face_found":    False
        }

    # ── PICK BEST FACE ────────────────────────────────────────
    # If multiple faces, take the highest score
    detection = max(results.detections,
                    key=lambda d: d.score[0])

    # ── BOUNDING BOX: relative → pixel coords ─────────────────
    bbox = detection.location_data.relative_bounding_box

    # MediaPipe gives top-left corner + width/height (relative)
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    bw = int(bbox.width  * w)
    bh = int(bbox.height * h)
    x2 = x1 + bw
    y2 = y1 + bh

    # ── PADDING ───────────────────────────────────────────────
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(w, x2 + pad_x)
    y2_pad = min(h, y2 + pad_y)

    # ── CROP ──────────────────────────────────────────────────
    face_bgr = frame[y1_pad:y2_pad, x1_pad:x2_pad]

    if face_bgr.size == 0:
        # Crop came out empty (weird, but be safe)
        return {
            "frame_number":  frame_number,
            "timestamp_sec": timestamp_sec,
            "timestamp_ms":  timestamp_ms,
            "face_crop":     None,
            "bbox_pixels":   (x1_pad, y1_pad, x2_pad, y2_pad),
            "face_found":    False
        }

    # ── GRAYSCALE ─────────────────────────────────────────────
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    # ── RESIZE ────────────────────────────────────────────────
    face_resized = cv2.resize(face_gray, target_size)

    return {
        "frame_number":  frame_number,
        "timestamp_sec": round(timestamp_sec, 4),
        "timestamp_ms":  timestamp_ms,
        "face_crop":     face_resized,  # numpy array (224, 224) uint8
        "bbox_pixels":   (x1_pad, y1_pad, x2_pad, y2_pad),
        "face_found":    True
    }

# =============================================================
#  PROCESS THE WHOLE VIDEO
# =============================================================
def process_video(video_path, output_dir, config):
    """
    Processes the entire video:
      - Extracts audio
      - Loops through frames (downsampled)
      - Detects face, crops, grayscales, timestamps
      - Saves frames + a timestamp manifest
    """

    # Clean old output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Re-create clean folders
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)


    # ── GET VIDEO INFO ─────────────────────────────────────────
    info = get_video_info(video_path)
    if info is None:
        return

    # Save video info
    with open(os.path.join(output_dir, "video_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    original_fps = info["fps"] or 0.0
    if original_fps <= 0:
        print("[VIDEO] ERROR: original_fps <= 0, cannot process safely.")
        return

    # ── AUDIO EXTRACTION ──────────────────────────────────────
    audio_path = extract_audio(video_path, output_dir, config["sample_rate"])
    if audio_path is None:
        print("[AUDIO] Warning: audio extraction failed; continuing with video only.")

    # ── DOWNSAMPLING SETUP ────────────────────────────────────
    # Example: original 30fps, target 15fps → frame_step ≈ 2
    frame_step = max(1, round(original_fps / config["target_fps"]))
    effective_fps = original_fps / frame_step
    print(f"\n[VIDEO] Downsampling: {original_fps:.2f}fps → {effective_fps:.2f}fps")
    print(f"[VIDEO] Keeping every {frame_step} frame(s)")

    # ── FACE DETECTOR ─────────────────────────────────────────
    face_detector = create_face_detector()

    # ── DATA STRUCTURES FOR TRACKING ──────────────────────────
    timestamps = {}  # frame filename → timestamp + face info
    log = {
        "total_frames_read":  0,
        "frames_kept":        0,    # after downsampling
        "faces_found":        0,
        "faces_missed":       0,
        "saved_frame_files":  []
    }

    # ── OPEN VIDEO AND LOOP THROUGH FRAMES ────────────────────
    print(f"\n[VIDEO] Processing frames...")
    cap = cv2.VideoCapture(str(video_path))  # ensure string path

    frame_number = 0   # counts frames in original video
    saved_count  = 0   # counts only frames we actually write to disk

    while True:
        ret, frame = cap.read()  # read one frame
        if not ret:
            break  # end of video

        log["total_frames_read"] += 1

        # ── DOWNSAMPLING ─────────────────────────────────────
        # Only process every Nth frame
        if frame_number % frame_step != 0:
            frame_number += 1
            continue

        log["frames_kept"] += 1

        # ── PROCESS THE FRAME ─────────────────────────────────
        result = process_frame(
            frame,
            frame_number,
            original_fps,
            face_detector,
            target_size=config["frame_size"],
            padding=config["face_padding"]
        )

        # ── SAVE THE FRAME ────────────────────────────────────
        # We save something for every kept frame (face or blank)
        filename = f"frame_{frame_number:06d}.png"
        frame_path = os.path.join(frames_dir, filename)

        if result["face_found"] and result["face_crop"] is not None:
            cv2.imwrite(frame_path, result["face_crop"])
            log["faces_found"] += 1
        else:
            # Save a blank black frame as placeholder
            blank = np.zeros(config["frame_size"], dtype=np.uint8)
            cv2.imwrite(frame_path, blank)
            log["faces_missed"] += 1

        # ── RECORD TIMESTAMP ─────────────────────────────────
        timestamps[filename] = {
            "frame_number":  result["frame_number"],
            "timestamp_sec": result["timestamp_sec"],
            "timestamp_ms":  result["timestamp_ms"],
            "face_found":    result["face_found"],
            "bbox":          result["bbox_pixels"],
        }

        log["saved_frame_files"].append(filename)
        saved_count += 1

        # Print progress every 50 saved frames
        if saved_count % 50 == 0:
            pct = (frame_number / info["total_frames"]) * 100
            print(
                f"  {pct:5.1f}% — frame {frame_number}/{info['total_frames']} "
                f"| saved {saved_count} frames "
                f"| faces found: {log['faces_found']}"
            )

        frame_number += 1

    cap.release()

    # ── SAVE TIMESTAMPS FILE ──────────────────────────────────
    timestamps_path = os.path.join(output_dir, "timestamps.json")
    with open(timestamps_path, "w") as f:
        json.dump(timestamps, f, indent=2)

    # ── SAVE PROCESSING LOG ───────────────────────────────────
    log_path = os.path.join(output_dir, "processing_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # ── PRINT SUMMARY ─────────────────────────────────────────
    face_rate = (log["faces_found"] / log["frames_kept"] * 100) if log["frames_kept"] > 0 else 0

    print(f"\n{'='*55}")
    print("DONE")
    print(f"{'='*55}")
    print(f"Total frames in video : {log['total_frames_read']}")
    print(f"Frames kept (after ds): {log['frames_kept']}")
    print(f"Faces detected        : {log['faces_found']} ({face_rate:.1f}%)")
    print(f"Frames with no face   : {log['faces_missed']}  (saved as black)")
    print("\nOutput files:")
    print(f"  {os.path.join(output_dir, 'audio.wav')}")
    print(f"  {os.path.join(output_dir, 'frames/')}  ({saved_count} PNGs)")
    print(f"  {timestamps_path}")
    print(f"  {log_path}")
    print(f"{'='*55}\n")

    return timestamps_path


# OPTIONAL — VISUAL PREVIEW (sanity check)
# =============================================================

def save_preview_strip(output_dir, n_frames=10):
    """
    Saves a single image showing N evenly-spaced face crops side by side.
    Useful for quickly checking if face detection worked correctly.
    Call this after process_video().
    """
    frames_dir = os.path.join(output_dir, "frames")
    timestamps_path = os.path.join(output_dir, "timestamps.json")

    if not os.path.exists(timestamps_path):
        print("No timestamps.json found — run process_video() first")
        return

    with open(timestamps_path) as f:
        timestamps = json.load(f)

    # Get frames with detected faces (sorted in chronological order)
    good_frames = sorted(
        [fname for fname, d in timestamps.items() if d["face_found"]],
        key=lambda name: int(name.replace("frame_", "").replace(".png", ""))
    )

    if len(good_frames) == 0:
        print("No frames with detected faces found")
        return

    # Pick N evenly spaced frames
    indices = np.linspace(0, len(good_frames) - 1, n_frames, dtype=int)
    selected = [good_frames[i] for i in indices]

    # Load and stack horizontally
    images = []
    for filename in selected:
        img_path = os.path.join(frames_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

    if len(images) == 0:
        print("No images loaded — something went wrong.")
        return

    strip = np.hstack(images)
    preview_path = os.path.join(output_dir, "preview_strip.png")
    cv2.imwrite(preview_path, strip)

    print(f"\n[PREVIEW] Saved {len(images)}-frame strip → {preview_path}")
    print("          Open this image to visually verify face cropping worked.")


# =============================================================
# RUN IT
# =============================================================
if __name__ == "__main__":

    config = {
        "target_fps":    TARGET_FPS,
        "frame_size":    FRAME_SIZE,
        "face_padding":  FACE_PADDING,
        "sample_rate":   AUDIO_SAMPLE_RATE,
    }

    # 1. Process the video
    timestamps_path = process_video(VIDEO_PATH, OUTPUT_DIR, config)

    if timestamps_path:
        # 2. OPTIONAL: Show alignment example (if implemented)
        # align_example(timestamps_path)

        # 3. Save a visual preview strip
        save_preview_strip(OUTPUT_DIR, n_frames=10)
