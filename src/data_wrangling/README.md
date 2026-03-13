# src/data_wrangling/

Video/audio preprocessing pipeline that downloads and wrangles raw datasets into the
`datasets/wrangled/` format consumed by `invoke generate-wrangled-tokens`.

See `DATA_WRANGLING.md` in the repo root for full documentation.

## Quick Reference

| Dataset | Invoke Command | Output Location |
|---------|---------------|-----------------|
| Seamless Interaction (session-ordered) | `invoke wrangle-seamless-sessions --count N` | `datasets/wrangled/SI{session}/` |
| Seamless Interaction (random pairs) | `invoke wrangle-seamless --count N` | `datasets/wrangled/SI*/` |
| CANDOR Corpus | `invoke wrangle-candor` | `datasets/wrangled/C{part}/` |

## Wrangled Output Format

Each wrangled item produces per-participant files:

```
datasets/wrangled/{source}/{session_or_pair}/
    participant_A.mp4    # cropped/resized video (webcam-style)
    participant_A.wav    # 16 kHz mono audio
    participant_A.json   # metadata (prompt, session info, timestamps)
    participant_B.mp4
    participant_B.wav
    participant_B.json
```

## Module Structure

```
candor/
    wrangle.py     — download + extract CANDOR parts → datasets/wrangled/C{part}/
    download.py    — CANDOR-specific download helpers
seamless_interaction/
    wrangle.py     — session-based Seamless wrangling with prompt metadata
    download.py    — Seamless dataset download helpers
cmu_mosei/
    (in progress)
```

## Memory-Efficient Design

All commands use **clean-as-you-go** processing: each item is fully processed and
intermediate files deleted before the next download begins. Peak disk usage stays
constant regardless of how many items are processed.
