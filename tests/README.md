# tests/

Pytest test suite for CAMELS v8.1.

## Running Tests

```bash
invoke test                          # full suite
pytest tests/ -v                     # verbose
pytest tests/test_losses.py -v       # single file
pytest tests/test_config.py::test_default_config  # single test
```

## Test Organization

| File | What It Covers |
|------|---------------|
| `test_config.py` | CAMELSConfig defaults, validation, enabled_modalities() |
| `test_adapters.py` | AVAEAdapter forward/embed, PhonemeAdapter, VelocityNet shapes |
| `test_losses.py` | All loss functions: MoCo, InfoNCE, AVAE, geometric, FM, phoneme probe |
| `test_training.py` | 3-stage training smoke tests, MoCo integration, gradient isolation |
| `test_dataset.py` | MultimodalDataset loading and batching |
| `test_export.py` | .npy shape validation and export helpers |
| `conftest.py` | Shared fixtures (synthetic tensors, mini configs) |
| `data_wrangling/` | Wrangling pipeline tests (CANDOR, Seamless) |

## Known Gaps

- **Streaming modules** (`encoding/streaming/`) — `AudioBuffer`, `FrameBuffer`, `FixedStrideScheduler` are not covered
- **Model loading** (`encoding/models/loader.py`) — tests would require downloading MARLIN/wav2vec2 weights; excluded from CI
- **Full pipeline integration** — `infer_chunk()` / `infer_batch()` end-to-end tests require real model weights

## Fixtures

`conftest.py` provides small synthetic configs (`d_latent=16`, `max_phones=8`) and fake
data loaders to keep all unit tests fast and dependency-free.
