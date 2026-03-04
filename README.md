# LSCA - Conversational Agents in Multimodal Embedded Latent Space

A research project building a real-time multimodal understanding pipeline for natural Human-AI interactions in video call contexts.

## Quick Start

```bash
# Setup environment
conda env create -f environment.yml
conda activate lsca

# Download a small dev dataset
invoke wrangle-dev
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

## Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Environment setup and basic usage |
| [DATA_WRANGLING.md](DATA_WRANGLING.md) | Dataset sources and processing details |
# CAMELS

**Multimodal Understanding Pipeline for Real-Time Human-Computer Interaction**

CAMELS is a 4-modality live streaming pipeline that encodes video, audio, prosody, and text into a shared 1024-dimensional latent space. Each modality produces an independent embedding vector suitable for downstream graph-based reasoning (e.g., HyperGNN).

## Architecture

| Modality | Frozen Encoder | Raw Dim | Adapter |
|----------|---------------|---------|---------|
| Video | MARLIN ViT-Base | 768 | AVAE (hidden=256) |
| Audio | wav2vec2-base | 768 | AVAE (hidden=256) |
| Prosody | librosa (18 features) | 18 | AVAE (hidden=64) |
| Text | Emformer RNN-T + SONAR | 1024 | AVAE (hidden=256) |

All adapters project into a **shared 1024-D latent space** (SONAR-native). The pipeline outputs per-chunk embeddings (`z_v.npy`, `z_a.npy`, `z_p.npy`, `z_t.npy`) — modalities are **not fused**, preserving individual signals for downstream models.

## Training Protocol

Three-stage curriculum:

1. **Stage A** — InfoNCE contrastive alignment across all 6 modality pairs
2. **Stage B** — + AVAE reconstruction, KL divergence, z-consistency
3. **Stage C** — + Bidirectional flow matching (video <-> audio)

## Setup

**Requirements:** Python 3.11+, ffmpeg

```bash
# Clone the repo
git clone https://github.com/<your-username>/capstone.git
cd capstone

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

Pre-trained encoder weights (MARLIN, wav2vec2, SONAR) are downloaded automatically on first run.

## Data

You must supply your own dataset. The expected structure is:

```
data/
├── naturalistic/
│   └── dev/
│       └── <session>/<segment>/
│           ├── *.mp4    (video)
│           ├── *.wav    (audio)
│           ├── *.json   (transcript + annotations)
│           └── *.npz    (pose, optional)
└── improvised/
    └── test/
        └── ...
```

Each `.json` file should contain a `"metadata:transcript"` key with word-level timestamps.

## Usage

### 1. Preprocess (extract raw features)

```bash
python scripts/preprocess_data.py \
    --data-root data/ \
    --output-dir outputs/features \
    --device cpu
```

Outputs: `v_raw.npy`, `a_raw.npy`, `p_raw.npy`, `t_raw.npy`, and `prosody_stats.json`.

### 2. Train adapters

```bash
python scripts/train_adapters.py \
    --feature-dir outputs/features \
    --checkpoint-dir checkpoints/ \
    --device cpu \
    --batch-size 64
```

Checkpoints are saved as `checkpoints/stage_{a,b,c}_epoch{N}.pt`.

### 3. Live streaming

```bash
python run_pipeline.py \
    --device cpu \
    --output-dir run_output \
    --checkpoint checkpoints/stage_c_epoch020.pt \
    --camera 0
```

Streams from your webcam + microphone, processes 2-second overlapping windows (1s stride), and writes per-chunk embedding vectors to the output directory. Press `Ctrl+C` to stop.

## Project Structure

```
src/
  data_wrangling/
    seamless_interaction/   # Seamless Interaction dataset
      crop.py               # Face-centered video cropping
      download.py           # Download from S3
      types.py              # Type definitions
    candor/                 # CANDOR Corpus dataset
      download.py           # Download and wrangling
      extract.py            # Audio extraction from MKVs
tests/
  data_wrangling/           # Pipeline tests
tasks.py                    # Invoke task definitions
```

## Common Commands

```bash
# Data wrangling (memory-efficient, clean-as-you-go)
invoke wrangle-dev                   # Small dev dataset (3 Seamless + 1 CANDOR)
invoke wrangle-seamless --count 10   # Process 10 Seamless Interaction pairs
invoke wrangle-candor                # Process full CANDOR dataset

# Development
invoke test                          # Run tests
invoke lint                          # Run ruff linter
```

## Related Work

Prior multimodal research: [cs627 - APE Swarm](https://github.com/WatsonWBlair/cs627)
├── pipeline/
│   ├── config.py            # All constants and hyperparameters
│   ├── adapters.py          # AVAE adapters and VelocityNet
│   ├── buffers.py           # Audio and frame ring buffers
│   ├── scheduler.py         # Fixed-stride chunk scheduler
│   ├── dispatch.py          # Per-chunk pipeline orchestration
│   ├── video_pipeline.py    # MARLIN encoding
│   ├── audio_pipeline.py    # wav2vec2 encoding
│   ├── prosody_pipeline.py  # librosa feature extraction
│   ├── text_pipeline.py     # Emformer ASR + SONAR encoding
│   └── inference.py         # Batch/chunk inference utilities
├── training/
│   ├── train.py             # 3-stage training loop
│   ├── losses.py            # InfoNCE, AVAE, flow matching losses
│   ├── dataset.py           # Dataset and dataloader construction
│   └── evaluate.py          # Alignment, retrieval, and MSE metrics
├── models/
│   └── model_loader.py      # Frozen encoder loading
├── scripts/
│   ├── preprocess_data.py   # Offline feature extraction
│   └── train_adapters.py    # Training entry point
├── run_pipeline.py          # Live streaming entry point
└── pyproject.toml           # Project metadata and dependencies
```

## License

All rights reserved.
