# scripts/

Entry point scripts for training, token generation, and analysis.

## Script Selection Guide

| Task | Script / Command |
|------|-----------------|
| Pregenerate backbone tokens for training | `invoke generate-wrangled-tokens` |
| Train adapter networks | `python scripts/train_adapters.py` |
| Run ablation sweep | `invoke run-ablations` or `python scripts/run_ablations.py` |
| Analyze dataset statistics | `invoke analyze-dataset --output stats.md` |
| Extract MERBench features | `python scripts/extract_merbench_features.py` |

## Deprecated

**`scripts/preprocess_data.py`** — Operated on raw `.mp4`/`.wav` files directly, predating the wrangled-dataset pipeline. Do not use for new work. Use `invoke generate-wrangled-tokens` instead, which reads from `datasets/wrangled/` (output of `invoke wrangle-*` commands).

## Token Generation Pipeline

```
invoke wrangle-seamless-sessions   →  datasets/wrangled/SI{session}/
invoke wrangle-candor              →  datasets/wrangled/C{part}/
invoke generate-wrangled-tokens    →  datasets/pregenerated/
python scripts/train_adapters.py   →  checkpoints/
```

See `DATA_WRANGLING.md` for wrangling details and `encoding/README.md` for the model architecture.

## train_adapters.py — Key Flags

```
--modalities video phoneme prosody   # subset to ablate (default: all 3)
--phoneme-adapter-type linear|avae   # default: linear
--no-moco                            # fall back to InfoNCE
--moco-momentum 0.999
--moco-queue-size 4096
--temperature 0.07
--lambda-var/cov/orth/aux            # geometric loss weights
--c-max-video/prosody/phoneme        # AVAE capacity targets
```

## run_ablations.py — Ablation Harness

Sweeps 13 variants across four axes: `d_latent` (256/512/768), modality combos (6 pairs + full + 3 single-modality), MoCo on/off, and phoneme adapter type (linear/avae). Single-modality variants are expected to fail `cfg.validate()` and are recorded as errors without crashing the sweep.

```bash
# Preview all commands
python scripts/run_ablations.py --dry-run --feature-dir datasets/pregenerated

# Run specific variants
python scripts/run_ablations.py --feature-dir datasets/pregenerated \
    --variants phoneme_avae no_moco d_latent_512

# Results aggregated to:
checkpoints/ablations/ablation_results.json
checkpoints/ablations/{name}/training_history.json
```
