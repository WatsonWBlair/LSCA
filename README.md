# video-face-pipeline

**Branch:** `feature/video-processor`

A preprocessing pipeline for naturalistic video data. Extracts face crops (grayscale, 224×224) at a target frame rate, pulls the audio track as a 16 kHz mono WAV, and emits a timestamp manifest for downstream alignment with audio/transcription models (e.g. Whisper).

---

## What it does

1. **Audio extraction** — uses `ffmpeg` to rip the audio stream to `audio.wav` (16 kHz, mono).
2. **Frame downsampling** — reads the original video (typically 30 fps) and keeps every Nth frame to hit a configurable target fps (default 15).
3. **Face detection** — runs Google MediaPipe's long-range face detector on each kept frame.
4. **Face cropping** — pads the bounding box by 40% (captures shoulders/chest context), crops, converts to grayscale, and resizes to 224×224.
5. **Timestamp manifest** — writes `timestamps.json` mapping each saved PNG filename to its exact `timestamp_sec` and `timestamp_ms` in the original video.
6. **Preview strip** — optionally saves a side-by-side strip of N evenly-spaced face crops for a quick visual sanity check.

---

## Output structure

```
data/processed_frames/
├── audio.wav               # 16 kHz mono audio
├── frames/
│   ├── frame_000000.png    # 224×224 grayscale face crop (or black if no face)
│   ├── frame_000002.png
│   └── ...
├── timestamps.json         # frame → {timestamp_sec, timestamp_ms, face_found, bbox}
├── video_info.json         # fps, resolution, duration of source video
├── processing_log.json     # counts: frames read, kept, faces found/missed
└── preview_strip.png       # optional visual check (10 frames side by side)
```

### `timestamps.json` schema

```json
{
  "frame_000000.png": {
    "frame_number": 0,
    "timestamp_sec": 0.0,
    "timestamp_ms": 0,
    "face_found": true,
    "bbox": [x1, y1, x2, y2]
  },
  ...
}
```

---

## Setup

### 1. Prerequisites

- Python >=3.11, <3.15 (as locked in `pyproject.toml`)
- `ffmpeg` installed and on your `PATH`
- **Note:** `mediapipe` depends on `opencv-contrib-python`. Do **not** also install `opencv-python` — the two packages conflict and will cause import errors.

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

### 2. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd <repo>
git checkout feature/video-processor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Edit the constants at the top of `video_processor.py`:

| Variable | Default | Description |
|---|---|---|
| `VIDEO_PATH` | *(set this)* | Path to your input `.mp4` file |
| `OUTPUT_DIR` | *(set this)* | Directory for all output files |
| `TARGET_FPS` | `15` | Target frame rate after downsampling |
| `FRAME_SIZE` | `(224, 224)` | Output crop size (width × height) |
| `FACE_PADDING` | `0.4` | Fractional padding around face bbox |
| `AUDIO_SAMPLE_RATE` | `16000` | WAV sample rate in Hz |

---

## Usage

```bash
python video_processor.py
```

After processing, a summary prints to stdout and all outputs land in `OUTPUT_DIR`.

To run just the preview strip on already-processed data, call `save_preview_strip()` directly in a Python session:

```python
from video_processor import save_preview_strip
save_preview_strip("/path/to/processed_frames", n_frames=10)
```

---

## Notes

- Frames where no face is detected are saved as **black 224×224 PNGs** so the frame sequence stays contiguous and alignment with audio is preserved.
- If multiple faces appear in a frame, the detection with the **highest confidence score** is used.
- The output directory is **wiped and recreated** on each run — don't store anything important there between runs.
- `ffmpeg` must be accessible on your system `PATH`; the script calls it via `subprocess`.

---

## Dependencies overview

| Package | Purpose |
|---|---|
| `opencv-python` | Video I/O, image processing, grayscale conversion |
| `mediapipe` | Real-time face detection |
| `numpy` | Array ops, blank frame generation |
| `ffmpeg` *(system)* | Audio extraction |

All other imports (`subprocess`, `os`, `json`, `pathlib`, `shutil`) are Python stdlib.
