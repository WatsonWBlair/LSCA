# scripts/

Entry point scripts for training, token generation, and analysis.

## Script Selection Guide

| Task | Script / Command |
|------|-----------------|
| Pregenerate backbone tokens for training | `invoke generate-wrangled-tokens` |
| Train adapter networks | `python scripts/train_adapters.py` |
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
