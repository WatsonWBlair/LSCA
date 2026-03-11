# CAMELS — Conversational Agents in Multimodal Embedded Latent Space

A research project building a real-time multimodal understanding pipeline for natural Human-AI interactions in video call contexts. CAMELS projects video, phoneme, and prosody modalities into a shared 768-D latent space using an Adapted Pretrained Encoders (APE) architecture.

## Architecture

| Modality | Frozen Encoder | Output | Adapter |
|----------|---------------|--------|---------|
| Video    | MARLIN ViT-Base | (768,) | AVAEAdapter |
| Phoneme  | wav2vec2-lv-60-espeak-cv-ft (CTC) | (MAX_PHONES, 768) | PhonemeAdapter + PhonemeAttnPool |
| Prosody  | librosa 22-dim features | (22,) | AVAEAdapter |

Shared latent dimension: **768** (configurable via `CAMELSConfig.latent.d_latent`). Text/transcript is handled by EmformerASR as a utility and is not a latent modality.

## Setup

```bash
# Clone the repository
git clone https://github.com/WatsonWBlair/LSCA.git
cd LSCA

# Create and activate conda environment
conda env create -f environment.yml
conda activate lsca
```

See [QUICKSTART.md](QUICKSTART.md) for full setup instructions including the `seamless_interaction` library.

## Data Wrangling

```bash
# Seamless Interaction dataset
invoke wrangle-seamless --count N

# CANDOR Corpus (download first, then process)
invoke download-candor --count N
invoke wrangle-candor --count N
```

See [DATA_WRANGLING.md](DATA_WRANGLING.md) for dataset details and output structures.

## Project Structure

```
encoding/                   # Main pipeline package (v8.1)
  config.py                 # CAMELSConfig dataclass hierarchy
  inference.py              # infer_chunk(), infer_batch()
  export.py                 # Shape-validated .npy export
  adapters/                 # AVAEAdapter, PhonemeAdapter, VelocityNet
  models/                   # Frozen encoder loading
  pipelines/                # Per-modality encoding pipelines
  streaming/                # Ring buffers, scheduler, dispatch
  training/                 # Losses, dataset, evaluation, train loop
src/data_wrangling/         # Dataset preprocessing
  seamless_interaction/     # Crop + download utilities
  candor/                   # Download + extract + wrangle utilities
scripts/
  preprocess_data.py        # Offline feature extraction
  train_adapters.py         # 3-stage training entry point
run_pipeline.py             # Live streaming entry point
tests/                      # Full pytest test suite
tasks.py                    # Invoke task definitions
```

## Common Commands

```bash
# Development
invoke test                          # Run pytest
invoke lint                          # Run ruff linter
invoke install                       # Update conda environment

# Training
python scripts/preprocess_data.py --data-root datasets/ --output-dir outputs/features
python scripts/train_adapters.py --feature-dir outputs/features --checkpoint-dir checkpoints/

# Live streaming
python run_pipeline.py --checkpoint checkpoints/stage_c_epoch020.pt --camera 0
```

## Related Work

Prior multimodal research: [cs627 — APE Swarm](https://github.com/WatsonWBlair/cs627)

## License

All rights reserved.
