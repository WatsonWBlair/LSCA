# Data Wrangling

This document covers data source selection and preprocessing for the LSCA project.

**Goal:** Produce audio/video pairs resembling webcam footage—mid-chest up, face-centered, appropriate aspect ratio.

## Quick Start

```bash
invoke wrangle-seamless-sessions --count 1
invoke wrangle-candor --count 1
```

Downloads and processes a small development dataset (1 Seamless session ~11 interactions + 1 CANDOR part).

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

### Setup: Local `seamless_interaction` Patch

The upstream library (`C:\Users\watso\Development\seamless_interaction`) calls `np.load()` without
`allow_pickle=True`. Some `movement_v4/pred_vertices` files contain object arrays and will crash
the multiprocessing pool with `ValueError: Object arrays cannot be loaded when allow_pickle=False`.

Apply this one-line patch before running any wrangling commands:

```python
# C:\Users\watso\Development\seamless_interaction\src\seamless_interaction\fs.py  ~line 969
# Change:
data = np.load(tmp_file_path)
# To:
data = np.load(tmp_file_path, allow_pickle=True)
```

This patch is local-only and does not need to be committed to this repo.

### Wrangling

**Session-based (preferred):**
```bash
invoke wrangle-seamless-sessions --count N
```

Downloads full sessions in chronological (archive_idx) order. Interactions within a session are processed sequentially, preserving recording order. Session metadata and conversation prompts are attached to each output JSON.

**Options:**
- `--count N` — Number of sessions (default: 28, all Improvised dev)
- `--style` — "naturalistic" or "improvised" (default: improvised)
- `--split` — "train", "dev", or "test" (default: dev)

**Random pairs (for quick sampling):**
```bash
invoke wrangle-seamless --count N
```
Downloads N random interaction pairs with no session ordering or prompt metadata.

### Output Structure

```
datasets/wrangled/
  S{session}/
    I{interaction}_P{participant}.mp4   # Cropped video (H.264)
    I{interaction}_P{participant}.wav   # Audio (16kHz mono)
    I{interaction}_P{participant}.json  # Transcript + VAD + session metadata
    I{interaction}_P{participant}.npz   # Pre-computed keypoints
```

Participants in the same interaction share session and interaction IDs, enabling programmatic conversation reconstruction.

**Session-wrangled JSON fields** (present when using `wrangle-seamless-sessions`):

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | str | Session key, e.g. `"V00_S0700"` |
| `session_interaction_idx` | int | 0-indexed position of this interaction within the session |
| `session_total_interactions` | int | Total interactions in the session |
| `prompt_a` | str | Prompt shown to participant A |
| `prompt_b` | str | Prompt shown to participant B |
| `ipc_a` | str | IPC code for participant A |
| `ipc_b` | str | IPC code for participant B |
| `interaction_type` | str | e.g. `"ipc_conversation"` |

These fields are propagated into each row of `chunks.jsonl` during token pre-generation.

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

### Storage Estimates: Pregenerated Tokens

Token files are written per participant-video into `datasets/pregenerated/{backbone_tag}/{session}/{stem}/`.
Each 2-second chunk (1-second stride) produces the following rows:

| File | Shape per chunk | Dtype | Size per chunk |
|------|----------------|-------|----------------|
| `v_raw.npy` | (768,) | float32 | 3 KB |
| `ph_raw.npy` | (50, 768) | float32 | **150 KB** ← dominates |
| `ph_labels.npy` | (50,) | int64 | ~0.4 KB |
| `ph_mask.npy` | (50,) | int64 | ~0.4 KB |
| `p_raw.npy` | (22,) | float32 | <1 KB |
| **Total** | | | **~154 KB/chunk** |

Chunk rate: **~3,600 chunks per hour** of footage.

**Full-dataset estimates:**

| Dataset | Scale | Estimated tokens |
|---------|-------|-----------------|
| CANDOR | 1,650 conversations × 2 participants × ~900 chunks (~15 min avg) | **~440 GB** |
| Seamless Interaction | ~540 MB per hour of footage processed | **~2 TB** (all 4,000+ hrs) |

> `ph_raw` accounts for ~97% of token storage. If memory is constrained, consider processing
> in per-session batches rather than generating all tokens upfront.

### Utility Commands

For debugging or archival workflows:

**Seamless Interaction:**
- `invoke wrangle-seamless-staged` — Backfill already-staged Seamless NPZs from `datasets/seamless_interaction/` into `datasets/wrangled/`

**CANDOR:**
- `invoke download-candor` — Download zips without processing
- `invoke download-candor --extract` — Download and extract zips
- `invoke extract-candor` — Extract audio from already-extracted MKVs
- `invoke wrangle-candor-to-wrangled` — Process already-downloaded parts into `datasets/wrangled/`

---

## Dataset Statistics

Generated by `invoke analyze-dataset --output stats.md`. Run after wrangling to refresh.

### Dataset Size Summary

| Source | Sessions | Participant Files |
|--------|----------|-------------------|
| CANDOR | — | — |
| Seamless | — | — |
| **Total** | **—** | **—** |

### Talk vs Silence Ratio

> Duration proxy: `max(vad.end)` per participant file — slightly underestimates if file ends in silence.

| Source | Speech | Duration (proxy) | Talk Ratio |
|--------|--------|------------------|------------|
| CANDOR | — | — | — |
| Seamless | — | — | — |
| **Total** | **—** | **—** | **—** |

### Gender Distribution (CANDOR only)

| Sex | Count | % |
|-----|-------|---|
| — | — | — |

### Age Distribution (CANDOR only)

| Bucket | Count | % |
|--------|-------|---|
| 18-25 | — | — |
| 26-35 | — | — |
| 36-45 | — | — |
| 46-55 | — | — |
| 55+ | — | — |

### Top-50 Most Common Words

| Rank | Word | Count |
|------|------|-------|
| — | — | — |
