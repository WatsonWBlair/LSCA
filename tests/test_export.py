# tests/test_export.py
# Export validation: shapes, row sync, dimension checks.

import os

import numpy as np
import pytest
import torch

from encoding.config import CAMELSConfig, LatentConfig
from encoding.export import validate_embedding_shape, validate_row_sync, validate_export_dimensions


@pytest.fixture
def export_cfg():
    return CAMELSConfig(latent=LatentConfig(d_latent=32, max_phones=10))


class TestValidateEmbeddingShape:
    def test_correct_shape(self):
        t = torch.randn(4, 32)
        assert validate_embedding_shape(t, (4, 32), "test") is True

    def test_wrong_shape(self):
        t = torch.randn(4, 32)
        assert validate_embedding_shape(t, (4, 64), "test") is False


class TestValidateRowSync:
    def test_sync_ok(self, tmp_path, export_cfg):
        d = export_cfg.latent.d_latent
        max_ph = export_cfg.latent.max_phones
        np.save(str(tmp_path / "z_v.npy"), np.zeros((5, d)))
        np.save(str(tmp_path / "z_p.npy"), np.zeros((5, d)))
        np.save(str(tmp_path / "z_ph.npy"), np.zeros((5, max_ph, d)))
        assert validate_row_sync(str(tmp_path), 4, export_cfg) is True

    def test_sync_mismatch(self, tmp_path, export_cfg):
        d = export_cfg.latent.d_latent
        max_ph = export_cfg.latent.max_phones
        np.save(str(tmp_path / "z_v.npy"), np.zeros((5, d)))
        np.save(str(tmp_path / "z_p.npy"), np.zeros((3, d)))  # Wrong count
        np.save(str(tmp_path / "z_ph.npy"), np.zeros((5, max_ph, d)))
        assert validate_row_sync(str(tmp_path), 4, export_cfg) is False

    def test_missing_file(self, tmp_path, export_cfg):
        # No files exist
        assert validate_row_sync(str(tmp_path), 0, export_cfg) is False


class TestValidateExportDimensions:
    def test_correct_dims(self, tmp_path, export_cfg):
        d = export_cfg.latent.d_latent
        max_ph = export_cfg.latent.max_phones
        np.save(str(tmp_path / "z_v.npy"), np.zeros((5, d)))
        np.save(str(tmp_path / "z_p.npy"), np.zeros((5, d)))
        np.save(str(tmp_path / "z_ph.npy"), np.zeros((5, max_ph, d)))
        assert validate_export_dimensions(str(tmp_path), export_cfg) is True

    def test_wrong_latent_dim(self, tmp_path, export_cfg):
        d = export_cfg.latent.d_latent
        max_ph = export_cfg.latent.max_phones
        np.save(str(tmp_path / "z_v.npy"), np.zeros((5, d + 1)))  # Wrong dim
        np.save(str(tmp_path / "z_p.npy"), np.zeros((5, d)))
        np.save(str(tmp_path / "z_ph.npy"), np.zeros((5, max_ph, d)))
        assert validate_export_dimensions(str(tmp_path), export_cfg) is False
