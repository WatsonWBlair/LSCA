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
| `invoke wrangle-seamless-sessions --count N` | Process N full Seamless sessions in chronological order with prompt metadata (default: 28) |
| `invoke wrangle-seamless --count N` | Process N random Seamless interaction pairs (no session ordering or prompt metadata) |
| `invoke wrangle-candor` | Download, extract, and wrangle CANDOR parts into datasets/wrangled/ |
| `invoke wrangle-candor-to-wrangled` | Backfill already-downloaded CANDOR parts into datasets/wrangled/ |
| `invoke generate-wrangled-tokens` | Pregenerate backbone tokens from datasets/wrangled/ into datasets/pregenerated/ |
| `invoke consolidate-pregenerated` | Consolidate per-stem .npy files into flat mmap files for training (run once per backbone tag) |
| `invoke run-ablations` | Run ablation sweep over d_latent, modality combos, MoCo, and phoneme adapter type |

Run tests: `pytest tests/ -v` or `pytest tests/test_config.py::test_default_config`

## Tech Stack

- **Python 3.11** with PyTorch, NumPy, OpenCV
- **Testing:** pytest (~94 tests across 11 test files)
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
    ├── dataset.py          # MultimodalDataset (mmap, 3 modalities), PregenDataset, DyadicPairDataset
    ├── evaluate.py         # 3-modality evaluation suite
    └── train.py            # 3-stage training loop (A → B → C)
```

### Architecture

| Modality | Frozen Encoder | Output | Adapter |
|----------|---------------|--------|---------|
| Video    | MARLIN ViT-Base | (768,) | AVAEAdapter |
| Phoneme  | wav2vec2-lv-60-espeak-cv-ft (CTC) | (MAX_PHONES, 1024) | PhonemeAdapter (linear) + PhonemeAttnPool **or** AVAEAdapter (configurable) |
| Prosody  | librosa 22-dim features | (22,) | AVAEAdapter |

`ModalityConfig.phoneme_adapter_type` selects between `"linear"` (default: sequence → linear proj → AttnPool) and `"avae"` (mean-pool raw features → AVAE, enables capacity-controlled KL in Stage B/C).

Shared latent dimension: **768** (configurable via `CAMELSConfig.latent.d_latent`).
Text/transcript is handled by EmformerASR as a utility (NOT a latent modality).

### Training (3-stage protocol)

MoCo (Momentum Contrast) is the default contrastive loss; falls back to InfoNCE when disabled. Configured via `CAMELSConfig.moco` (`MoCoConfig`: `enabled`, `momentum`, `queue_size`, `temperature`).

- **Stage A:** MoCo/InfoNCE + L_var + L_cov + L_aux (phoneme probe)
- **Stage B:** + AVAE (capacity-controlled KL) + L_orth
- **Stage C:** + bidirectional FM (video ↔ phoneme, separate optimizer)

### Entry Points

| File | Purpose |
|------|---------|
| `scripts/generate_wrangled_tokens.py` | Pregenerate backbone tokens from datasets/wrangled/ into datasets/pregenerated/ |
| `scripts/consolidate_pregenerated.py` | Streaming consolidation of per-stem .npy files into flat mmap files under datasets/consolidated/ |
| `scripts/train_adapters.py` | Run 3-stage training protocol (supports `--modalities`, `--phoneme-adapter-type`, `--no-moco`, `--num-workers`, loss weight flags) |
| `scripts/run_ablations.py` | Ablation harness — sweeps d_latent, modality combos, MoCo on/off, phoneme adapter type; use `--dry-run` to preview |
| `scripts/preprocess_data.py` | **Deprecated** — extract raw features directly from .mp4/.wav (use `invoke generate-wrangled-tokens` instead) |

Live streaming is handled via `encoding/streaming/` (see `encoding/streaming/dispatch.py`).

### Other Directories

- `src/data_wrangling/` — Video preprocessing pipeline (Seamless + CANDOR)
  - `candor/wrangle.py` — Extracts per-participant `.mp4`, `.wav`, and `.json` into `datasets/wrangled/C{part_num}/`
- `tests/` — Full pytest test suite

## Planned (Not Yet Implemented)

The following research directions are described in design docs but have no implementation yet:

- **Evidential embeddings** — Dempster-Shafer Theory for uncertainty-aware latent representations
- **Supervised contrastive disentanglement phase** — label-guided Stage D after Stage C
- **GAN discriminator** — adversarial alignment objective
- **HGNN context management** — Heterogeneous Graph Neural Network for multi-turn session context
- **Multi-agent latent space agents** — agent roles operating in shared 768-D space

## Critical Design Rules

- **Latent dimension is NEVER hardcoded** — always flows from `CAMELSConfig.latent.d_latent`
- Frames must be **RGB float32 + ImageNet normalized** (MARLIN requirement)
- Row N in every .npy == chunk N in chunks.jsonl — never violate
- Silent chunks → append zero rows, never skip chunk_id
- FM loss: adapters **detached** — only VelocityNets receive gradients
- `phoneme_adapter_type="linear"` (default): sequence path with `PhonemeAttnPool` + optional `PhonemeProbeHead`; `"avae"`: mean-pool → `AVAEAdapter`, no probe, AVAE loss active in Stage B/C via `c_max_phoneme`
- Ablation variants are isolated under `checkpoints/ablations/{name}/`; single-modality runs are expected to fail `validate()` and are recorded as errors without crashing the harness

