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

Downloads, extracts, and wrangles parts directly into `datasets/wrangled/`. Supports resume via marker files.

**Options:**
- `--start N` — Part number to start from (default: 1)
- `--count N` — Number of parts to process (default: all 166)

**Dataset size:** 166 zip files (~5GB each, ~850GB total raw). After wrangling: ~280GB.

To backfill already-downloaded parts (e.g., from a prior `invoke download-candor` run):

```bash
invoke wrangle-candor-to-wrangled
```

### Output Structure

```
datasets/wrangled/
  C{part_num}/
    {uuid}_{user_id}.mp4    # Per-participant video
    {uuid}_{user_id}.wav    # Per-participant audio (16kHz mono)
    {uuid}_{user_id}.json   # Metadata (VAD, transcript, survey, audio/video features)
```

> **JSON schema:** `id` (conversation UUID), `metadata.vad`, `metadata.transcript`,
> `metadata.survey`, `metadata.audio_video_features`

Storage: ~170MB per conversation.

### Token Pre-generation

After wrangling, pregenerate backbone tokens for efficient training:

```bash
invoke generate-wrangled-tokens
```

Runs all wrangled files through the frozen encoders and saves outputs to `datasets/pregenerated/`.

### Utility Commands

For debugging or archival workflows:
- `invoke download-candor` — Download zips without processing
- `invoke download-candor --extract` — Download and extract zips
- `invoke extract-candor` — Extract audio from already-extracted MKVs
- `invoke wrangle-candor-to-wrangled` — Process already-downloaded parts into `datasets/wrangled/`
