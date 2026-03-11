# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CAMELS (Conversational Agents in Multimodal Embedded Latent Space) is a research project building a real-time multimodal understanding pipeline for natural Human-AI interactions in video call contexts. It projects video, phoneme, and prosody modalities into a shared 768-D latent space using an Adapted Pretrained Encoders (APE) architecture (v8.1).

Key research areas: APE-based ablation studies, streaming architecture for real-time video, evidential embeddings with Dempster-Shafer Theory, and interpretability via white-box reasoning with conflict tracing.

Prior tooling lives at https://github.com/WatsonWBlair/cs627 (Multimodal Joint Representation Learning Via APE Swarm).

## Development Commands

All task automation uses [Invoke](https://www.pyinvoke.org/):

**Development:**
| Command | Description |
|---------|-------------|
| `invoke install` | Update conda environment from environment.yml |
| `invoke test` | Run tests with pytest |
| `invoke lint` | Run ruff linter |
| `invoke clean` | Remove build artifacts, caches, bytecode |
| `invoke freeze` | Export conda environment to environment.yml |

**Data Wrangling** (clean-as-you-go processing):
| Command | Description |
|---------|-------------|
| `invoke wrangle-dev` | Small dev dataset (3 Seamless + 1 CANDOR) |
| `invoke wrangle-seamless --count N` | Process N Seamless Interaction pairs |
| `invoke wrangle-candor` | Process CANDOR dataset iteratively |

Run tests: `pytest tests/ -v` or `pytest tests/test_config.py::test_default_config`

## Tech Stack

- **Python 3.11** with PyTorch, NumPy, OpenCV
- **Testing:** pytest (79 tests across 7 test files)
- **Linting:** ruff (via `invoke lint`), pylint (via Codacy)
- **Code analysis:** Codacy with semgrep + trivy (vulnerability scanning)

## Data Sources

Two primary datasets (see `DATA_WRANGLING.md` for details):
- **Seamless_Interaction:** 4,000+ hours of in-person face-to-face interaction footage
- **CANDOR Corpus:** 1,650 video chat conversations between strangers with rich metadata

## Code Structure (v8.1)

### `encoding/` — Main pipeline package

```
encoding/
├── config.py              # Centralized typed config (CAMELSConfig dataclass hierarchy)
├── inference.py            # infer_chunk(), infer_batch() for 3 modalities
├── export.py               # Shape-validated .npy export helpers
├── adapters/
│   ├── base.py             # MLP, AVAEAdapter, TemporalAttentionPool
│   ├── phoneme.py          # PhonemeAdapter (linear), PhonemeAttnPool, PhonemeProbeHead
│   ├── velocity.py         # VelocityNet (FM transport)
│   └── registry.py         # build_adapters(), save/load_adapters()
├── models/
│   └── loader.py           # load_marlin, load_wav2vec2_ctc, load_emformer
├── pipelines/
│   ├── video.py            # MARLIN + TemporalPool → (768,)
│   ├── phoneme.py          # wav2vec2-CTC segmentation + per-phoneme embedding
│   ├── prosody.py          # librosa 22-dim features
│   └── transcript.py       # EmformerASR (utility only, NOT a latent modality)
├── streaming/
│   ├── buffers.py          # AudioBuffer, FrameBuffer (thread-safe ring buffers)
│   ├── scheduler.py        # FixedStrideScheduler
│   └── dispatch.py         # run_all_pipelines(), handle_silent_chunk()
└── training/
    ├── losses.py           # InfoNCE, AVAE-cap, L_orth, L_var, L_cov, FM, phoneme probe
    ├── dataset.py          # MultimodalDataset (3 modalities)
    ├── evaluate.py         # 3-modality evaluation suite
    └── train.py            # 3-stage training loop (A → B → C)
```

### Architecture

| Modality | Frozen Encoder | Output | Adapter |
|----------|---------------|--------|---------|
| Video    | MARLIN ViT-Base | (768,) | AVAEAdapter |
| Phoneme  | wav2vec2-lv-60-espeak-cv-ft (CTC) | (MAX_PHONES, 768) | PhonemeAdapter (linear) + PhonemeAttnPool |
| Prosody  | librosa 22-dim features | (22,) | AVAEAdapter |

Shared latent dimension: **768** (configurable via `CAMELSConfig.latent.d_latent`).
Text/transcript is handled by EmformerASR as a utility (NOT a latent modality).

### Training (3-stage protocol)

- **Stage A:** InfoNCE + L_var + L_cov + L_aux (phoneme probe)
- **Stage B:** + AVAE (capacity-controlled KL) + L_orth
- **Stage C:** + bidirectional FM (video ↔ phoneme, separate optimizer)

### Entry Points

| File | Purpose |
|------|---------|
| `run_pipeline.py` | Live streaming pipeline |
| `scripts/preprocess_data.py` | Extract raw features for training |
| `scripts/train_adapters.py` | Run 3-stage training protocol |

### Other Directories

- `src/data_wrangling/` — Video preprocessing pipeline (Seamless + CANDOR)
- `tests/` — Full pytest test suite
- `pipeline/` — **DELETED** (old 4-modality code, replaced by `encoding/`)

## Critical Design Rules

- **Latent dimension is NEVER hardcoded** — always flows from `CAMELSConfig.latent.d_latent`
- Frames must be **RGB float32 + ImageNet normalized** (MARLIN requirement)
- Row N in every .npy == chunk N in chunks.json — never violate
- Silent chunks → append zero rows, never skip chunk_id
- FM loss: adapters **detached** — only VelocityNets receive gradients
- PhonemeAdapter is **linear** (not AVAE) — verification uses PhonemeProbeHead instead

## Python Environment

- **Always use**: `/Users/katyasha1/capstone/.venv/bin/python3.11`
- The venv symlink is `python3.11` (no `python3` alias)
