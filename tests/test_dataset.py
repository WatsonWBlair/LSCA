# tests/test_dataset.py
# Dataset loading, __getitem__ shapes, make_dataloaders.

import pytest
import torch

from encoding.training.dataset import MultimodalDataset, make_dataloaders


class TestMultimodalDataset:
    def test_load(self, feature_dir, cfg):
        ds = MultimodalDataset(feature_dir, cfg)
        assert len(ds) == 20

    def test_getitem_shapes(self, feature_dir, cfg):
        ds = MultimodalDataset(feature_dir, cfg)
        v, ph, labels, mask, p = ds[0]

        assert v.shape == (cfg.latent.d_video,)
        assert ph.shape == (cfg.latent.max_phones, cfg.latent.d_phoneme)
        assert labels.shape == (cfg.latent.max_phones,)
        assert mask.shape == (cfg.latent.max_phones,)
        assert p.shape == (cfg.latent.d_prosody,)

    def test_mask_is_bool(self, feature_dir, cfg):
        ds = MultimodalDataset(feature_dir, cfg)
        _, _, _, mask, _ = ds[0]
        assert mask.dtype == torch.bool

    def test_missing_file_raises(self, cfg, tmp_path):
        with pytest.raises(FileNotFoundError):
            MultimodalDataset(str(tmp_path), cfg)


class TestMakeDataloaders:
    def test_three_loaders(self, feature_dir, cfg):
        train_dl, val_dl, test_dl = make_dataloaders(feature_dir, cfg)
        assert len(train_dl.dataset) > 0
        assert len(val_dl.dataset) > 0
        assert len(test_dl.dataset) > 0

    def test_batch_shape(self, feature_dir, cfg):
        train_dl, _, _ = make_dataloaders(feature_dir, cfg)
        batch = next(iter(train_dl))
        v, ph, labels, mask, p = batch
        B = v.shape[0]
        assert B <= cfg.training.batch_size
        assert ph.shape == (B, cfg.latent.max_phones, cfg.latent.d_phoneme)
        assert labels.shape == (B, cfg.latent.max_phones)
        assert p.shape == (B, cfg.latent.d_prosody)
