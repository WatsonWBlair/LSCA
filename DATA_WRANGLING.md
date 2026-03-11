# Data Wrangling

This document covers data source selection and preprocessing for the LSCA project.

**Goal:** Produce audio/video pairs resembling webcam footage—mid-chest up, face-centered, appropriate aspect ratio.

## Quick Start

```bash
invoke wrangle-seamless --count 3
invoke wrangle-candor --count 1
```

Downloads and processes a small development dataset (3 Seamless Interaction pairs + 1 CANDOR part).

## Memory-Efficient Processing

All wrangling commands use **clean-as-you-go** processing to minimize disk usage:

| Dataset | Working Space | Without Clean-as-you-go |
|---------|---------------|-------------------------|
| Seamless Interaction | ~200-400MB | N × 200MB (scales with count) |
| CANDOR | ~5GB per part | 850GB+ (all zips) |

Each item is fully processed before the next is downloaded, keeping peak disk usage constant regardless of dataset size.

---

## Seamless Interaction

4,000+ hours of in-person face-to-face interaction footage from 4,000+ participants.

> Note: In-person interactions differ from video calls in turn-taking and gaze patterns ([Tian et al., 2024](./litrature/datawrangling/Seamless_Interaction.pdf)), but the dataset's goals align with ours: training agents with natural gestures, modeling turn-taking, and understanding multimodal social dynamics.

### Wrangling

```bash
invoke wrangle-seamless --count N
```

Downloads, crops to webcam framing, and cleans up one pair at a time.

**Options:**
- `--count N` — Number of interaction pairs (default: 1)
- `--style` — "naturalistic" or "improvised" (default: improvised)
- `--split` — "train", "dev", or "test" (default: dev)

### Output Structure

```
datasets/wrangled/
  S{session}/
    I{interaction}_P{participant}.mp4   # Cropped video (H.264)
    I{interaction}_P{participant}.wav   # Audio (16kHz mono)
    I{interaction}_P{participant}.json  # Transcript + VAD metadata
    I{interaction}_P{participant}.npz   # Pre-computed keypoints
```

Participants in the same interaction share session and interaction IDs, enabling programmatic conversation reconstruction.

---

## CANDOR Corpus

1,650 video chat conversations between strangers with rich pre/post-conversation survey metadata. Already captured via video chat—no cropping needed.

### Wrangling

```bash
invoke wrangle-candor
```

Two-step workflow: first download zips with `invoke download-candor`, then extract and process with `invoke wrangle-candor`. Both commands support resume via marker files.

**Options:**
- `--start N` — Part number to start from (default: 1)
- `--count N` — Number of parts to process (default: all 166)

**Dataset size:** 166 zip files (~5GB each, ~850GB total raw). After wrangling: ~280GB.

### Output Structure

```
datasets/candor/
  {conversation-uuid}/
    processed/
      {user-id}.mp4   # Per-participant video (cropped)
      {user-id}.wav   # Per-participant audio (16kHz mono)
    transcription/
    metadata.json
    survey.csv
    audio_video_features.csv
```

Storage: ~170MB per conversation.

### Utility Commands

For debugging or archival workflows:
- `invoke download-candor` — Download zips without processing
- `invoke download-candor --extract` — Download and extract zips
- `invoke extract-candor` — Extract audio from already-extracted MKVs
