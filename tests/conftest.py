# tests/conftest.py
# Shared fixtures for CAMELS v8.1 test suite.

import os
import tempfile

import numpy as np
import pytest
import torch

from encoding.config import CAMELSConfig, LatentConfig, TrainingConfig


@pytest.fixture
def cfg():
    """Default test config with small dimensions for speed."""
    return CAMELSConfig(
        latent=LatentConfig(
            d_latent=32,
            d_video=32,
            d_phoneme=32,
            d_prosody=22,
            max_phones=10,
            num_phoneme_classes=40,
        ),
        training=TrainingConfig(
            stage_a_epochs=2,
            stage_b_epochs=2,
            stage_c_epochs=2,
            batch_size=4,
            eval_every=1,
        ),
    )


@pytest.fixture
def batch(cfg):
    """Random batch matching the 3-modality format."""
    B = cfg.training.batch_size
    d_v = cfg.latent.d_video
    d_ph = cfg.latent.d_phoneme
    d_p = cfg.latent.d_prosody
    max_ph = cfg.latent.max_phones

    v_raw = torch.randn(B, d_v)
    ph_raw = torch.randn(B, max_ph, d_ph)
    ph_labels = torch.randint(0, cfg.latent.num_phoneme_classes, (B, max_ph))
    # Each sample has 3-7 real phonemes
    ph_mask = torch.zeros(B, max_ph)
    for i in range(B):
        n_real = torch.randint(3, min(8, max_ph + 1), (1,)).item()
        ph_mask[i, :n_real] = 1.0
    p_raw = torch.randn(B, d_p)

    return v_raw, ph_raw, ph_labels, ph_mask, p_raw


@pytest.fixture
def adapters(cfg):
    """Build adapters from config."""
    from encoding.adapters.registry import build_adapters
    return build_adapters(cfg)


@pytest.fixture
def feature_dir(cfg):
    """Create a temporary directory with fake .npy feature files."""
    N = 20
    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "v_raw.npy"),
                np.random.randn(N, cfg.latent.d_video).astype(np.float32))
        np.save(os.path.join(tmpdir, "ph_raw.npy"),
                np.random.randn(N, cfg.latent.max_phones, cfg.latent.d_phoneme).astype(np.float32))
        np.save(os.path.join(tmpdir, "ph_labels.npy"),
                np.random.randint(0, cfg.latent.num_phoneme_classes, (N, cfg.latent.max_phones)).astype(np.int64))
        mask = np.zeros((N, cfg.latent.max_phones), dtype=np.float32)
        for i in range(N):
            n_real = np.random.randint(3, min(8, cfg.latent.max_phones + 1))
            mask[i, :n_real] = 1.0
        np.save(os.path.join(tmpdir, "ph_mask.npy"), mask)
        np.save(os.path.join(tmpdir, "p_raw.npy"),
                np.random.randn(N, cfg.latent.d_prosody).astype(np.float32))
        yield tmpdir
