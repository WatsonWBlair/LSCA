# tests/test_pipelines.py
# Pipeline output shape tests with mock encoders.
# These tests verify pipeline logic WITHOUT requiring real models.

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from encoding.config import CAMELSConfig, LatentConfig
from encoding.adapters.base import TemporalAttentionPool
from encoding.pipelines.prosody import extract_prosody_raw


@pytest.fixture
def pipeline_cfg():
    return CAMELSConfig(latent=LatentConfig(d_prosody=22))


class TestProsodyPipeline:
    def test_extract_prosody_raw_shape(self, pipeline_cfg):
        sr = pipeline_cfg.streaming.sample_rate
        duration_sec = 2.0
        waveform = np.random.randn(int(sr * duration_sec)).astype(np.float32)
        raw = extract_prosody_raw(waveform, sr=sr, d_prosody=22)
        assert raw.shape == (22,)

    def test_extract_prosody_raw_finite(self, pipeline_cfg):
        sr = pipeline_cfg.streaming.sample_rate
        waveform = np.random.randn(sr * 2).astype(np.float32)
        raw = extract_prosody_raw(waveform, sr=sr, d_prosody=22)
        assert np.all(np.isfinite(raw))

    def test_extract_prosody_silence(self, pipeline_cfg):
        sr = pipeline_cfg.streaming.sample_rate
        waveform = np.zeros(sr * 2, dtype=np.float32)
        raw = extract_prosody_raw(waveform, sr=sr, d_prosody=22)
        assert raw.shape == (22,)
        assert np.all(np.isfinite(raw))


class TestVideoPipelineHelpers:
    def test_temporal_pool_shape(self, pipeline_cfg):
        d = pipeline_cfg.latent.d_video
        pool = TemporalAttentionPool(d=d)
        # Simulate 4 MARLIN windows
        H = torch.randn(4, d)
        out = pool(H)
        assert out.shape == (d,)

    def test_temporal_pool_single_window(self, pipeline_cfg):
        d = pipeline_cfg.latent.d_video
        pool = TemporalAttentionPool(d=d)
        H = torch.randn(1, d)
        out = pool(H)
        assert out.shape == (d,)


class TestPhonemePipelineHelpers:
    def test_pad_phonemes(self):
        from encoding.pipelines.phoneme import pad_phonemes

        max_phones = 10
        d = 32
        # 5 real phonemes
        embs = torch.randn(5, d)
        labels = torch.randint(0, 40, (5,))
        mask = torch.ones(5, dtype=torch.bool)
        padded_embs, padded_labels, padded_mask = pad_phonemes(embs, labels, mask, max_phones, d)

        assert padded_embs.shape == (max_phones, d)
        assert padded_labels.shape == (max_phones,)
        assert padded_mask.shape == (max_phones,)
        assert padded_mask[:5].sum() == 5
        assert padded_mask[5:].sum() == 0

    def test_pad_phonemes_truncation(self):
        from encoding.pipelines.phoneme import pad_phonemes

        max_phones = 5
        d = 32
        # 10 phonemes — should be truncated
        embs = torch.randn(10, d)
        labels = torch.randint(0, 40, (10,))
        mask = torch.ones(10, dtype=torch.bool)
        padded_embs, padded_labels, padded_mask = pad_phonemes(embs, labels, mask, max_phones, d)

        assert padded_embs.shape == (max_phones, d)
        assert padded_mask.sum() == max_phones

    def test_pad_phonemes_empty(self):
        from encoding.pipelines.phoneme import pad_phonemes

        max_phones = 10
        d = 32
        embs = torch.randn(0, d)
        labels = torch.randint(0, 40, (0,))
        mask = torch.zeros(0, dtype=torch.bool)
        padded_embs, padded_labels, padded_mask = pad_phonemes(embs, labels, mask, max_phones, d)

        assert padded_embs.shape == (max_phones, d)
        assert padded_mask.sum() == 0


class TestPhonemeDecoder:
    def _make_decoder(self, vocab):
        from encoding.models.loader import _PhonemeDecoder

        fe = MagicMock()
        return _PhonemeDecoder(fe, vocab), fe

    def test_decode_returns_ipa(self):
        decoder, _ = self._make_decoder({4: "n", 5: "s"})
        assert decoder.decode([4, 5]) == "n s"

    def test_decode_unknown_id_fallback(self):
        decoder, _ = self._make_decoder({})
        assert decoder.decode([999]) == "999"

    def test_call_delegates(self):
        mock_audio = np.zeros(16000, dtype=np.float32)
        sentinel = object()
        decoder, fe = self._make_decoder({})
        fe.return_value = sentinel
        result = decoder(mock_audio, sampling_rate=16000, return_tensors="pt")
        fe.assert_called_once_with(mock_audio, sampling_rate=16000, return_tensors="pt")
        assert result is sentinel
