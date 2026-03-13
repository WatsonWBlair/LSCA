# encoding/

Main pipeline package for CAMELS v8.1. Implements the APE (Adapted Pretrained Encoders) architecture projecting video, phoneme, and prosody into a shared 768-D latent space.

## Module Overview

```
config.py        → CAMELSConfig (single source of truth for all dimensions and hyperparams)
inference.py     → infer_chunk(), infer_batch() — high-level 3-modality inference
export.py        → shape-validated .npy export helpers
```

### adapters/

Trainable modules that project frozen encoder outputs into the shared latent space.

```
base.py          → MLP, AVAEAdapter (AVAE forward/embed/encode/decode), TemporalAttentionPool
phoneme.py       → PhonemeAdapter (linear), PhonemeAttnPool, PhonemeProbeHead
velocity.py      → VelocityNet (OT flow matching transport)
registry.py      → build_adapters(cfg), save_adapters(), load_adapters()
```

### models/

Frozen pretrained encoders (never trained, weights not updated).

```
loader.py        → load_marlin(), load_wav2vec2_ctc(), load_emformer()
```

### pipelines/

Per-modality feature extraction from raw audio/video.

```
video.py         → MARLIN ViT-Base + TemporalPool → (768,)
phoneme.py       → wav2vec2-CTC segmentation + per-phoneme hidden states → (MAX_PHONES, 768)
prosody.py       → librosa 22-dim features → (22,)
transcript.py    → EmformerASR — utility only, NOT a latent modality
```

### streaming/

Thread-safe ring buffers and scheduling for live inference.

```
buffers.py       → AudioBuffer, FrameBuffer
scheduler.py     → FixedStrideScheduler
dispatch.py      → run_all_pipelines(), handle_silent_chunk()
```

### training/

3-stage training curriculum.

```
losses.py        → MoCo, InfoNCE, AVAE-cap, L_orth, L_var, L_cov, FM, phoneme probe
momentum.py      → MoCoQueue, MomentumEncoderManager (EMA key encoder + queue)
dataset.py       → MultimodalDataset
evaluate.py      → 3-modality evaluation suite (10+ metrics)
train.py         → train_stage_a/b/c(), train_all_stages()
```

## Typical Import Patterns

```python
from encoding.config import CAMELSConfig
from encoding.adapters.registry import build_adapters, load_adapters
from encoding.inference import infer_batch
```

## Data Flow

```
raw input
   ↓ pipelines/ (frozen encoders)
raw features  (v_raw, ph_raw, p_raw)
   ↓ adapters/ (trainable)
latent embeddings  (z_v, z_ph, z_p)  — all (B, 768)
   ↓ training/losses.py
MoCo / InfoNCE + AVAE + geometric losses
```
