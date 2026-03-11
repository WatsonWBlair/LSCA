# tests/test_adapters.py
# Shape tests for all adapter types and build_adapters().

import torch
import pytest

from encoding.adapters.base import MLP, AVAEAdapter, TemporalAttentionPool
from encoding.adapters.phoneme import PhonemeAdapter, PhonemeAttnPool, PhonemeProbeHead
from encoding.adapters.velocity import VelocityNet
from encoding.adapters.registry import build_adapters, save_adapters, trainable_params
from encoding.config import CAMELSConfig, LatentConfig, ModalityConfig


class TestMLP:
    def test_output_shape(self):
        mlp = MLP([64, 32, 16])
        x = torch.randn(4, 64)
        assert mlp(x).shape == (4, 16)

    def test_with_layernorm(self):
        mlp = MLP([64, 32, 16], norm="layernorm")
        x = torch.randn(4, 64)
        assert mlp(x).shape == (4, 16)


class TestAVAEAdapter:
    def test_forward_shapes(self, cfg):
        adapter = AVAEAdapter(d_in=cfg.latent.d_video, d_latent=cfg.latent.d_latent, hidden=64)
        x = torch.randn(4, cfg.latent.d_video)
        mu, logvar, z, x_hat, z_prime = adapter(x)

        assert mu.shape == (4, cfg.latent.d_latent)
        assert logvar.shape == (4, cfg.latent.d_latent)
        assert z.shape == (4, cfg.latent.d_latent)
        assert x_hat.shape == (4, cfg.latent.d_video)
        assert z_prime.shape == (4, cfg.latent.d_latent)

    def test_embed(self, cfg):
        adapter = AVAEAdapter(d_in=cfg.latent.d_video, d_latent=cfg.latent.d_latent, hidden=64)
        x = torch.randn(4, cfg.latent.d_video)
        z = adapter.embed(x)
        assert z.shape == (4, cfg.latent.d_latent)

    def test_decode(self, cfg):
        adapter = AVAEAdapter(d_in=cfg.latent.d_video, d_latent=cfg.latent.d_latent, hidden=64)
        z = torch.randn(4, cfg.latent.d_latent)
        x_hat = adapter.decode(z)
        assert x_hat.shape == (4, cfg.latent.d_video)


class TestTemporalAttentionPool:
    def test_output_shape(self, cfg):
        pool = TemporalAttentionPool(d=cfg.latent.d_video)
        H = torch.randn(8, cfg.latent.d_video)  # 8 windows
        out = pool(H)
        assert out.shape == (cfg.latent.d_video,)


class TestPhonemeAdapter:
    def test_forward_shape(self, cfg):
        adapter = PhonemeAdapter(d_in=cfg.latent.d_phoneme, d_latent=cfg.latent.d_latent)
        x = torch.randn(4, cfg.latent.max_phones, cfg.latent.d_phoneme)
        out = adapter(x)
        assert out.shape == (4, cfg.latent.max_phones, cfg.latent.d_latent)

    def test_embed_alias(self, cfg):
        adapter = PhonemeAdapter(d_in=cfg.latent.d_phoneme, d_latent=cfg.latent.d_latent)
        x = torch.randn(4, cfg.latent.max_phones, cfg.latent.d_phoneme)
        assert torch.equal(adapter(x), adapter.embed(x))


class TestPhonemeAttnPool:
    def test_output_shape(self, cfg):
        pool = PhonemeAttnPool(d=cfg.latent.d_latent)
        H = torch.randn(4, cfg.latent.max_phones, cfg.latent.d_latent)
        mask = torch.ones(4, cfg.latent.max_phones)
        mask[:, 5:] = 0  # 5 real phonemes
        out = pool(H, mask)
        assert out.shape == (4, cfg.latent.d_latent)

    def test_mask_effect(self, cfg):
        pool = PhonemeAttnPool(d=cfg.latent.d_latent)
        H = torch.randn(4, cfg.latent.max_phones, cfg.latent.d_latent)
        mask_full = torch.ones(4, cfg.latent.max_phones)
        mask_half = torch.ones(4, cfg.latent.max_phones)
        mask_half[:, cfg.latent.max_phones // 2:] = 0
        out_full = pool(H, mask_full)
        out_half = pool(H, mask_half)
        # Different masks should produce different outputs
        assert not torch.allclose(out_full, out_half)


class TestPhonemeProbeHead:
    def test_output_shape(self, cfg):
        probe = PhonemeProbeHead(d=cfg.latent.d_latent, n_classes=cfg.latent.num_phoneme_classes)
        z = torch.randn(4, cfg.latent.max_phones, cfg.latent.d_latent)
        logits = probe(z)
        assert logits.shape == (4, cfg.latent.max_phones, cfg.latent.num_phoneme_classes)


class TestVelocityNet:
    def test_output_shape(self, cfg):
        net = VelocityNet(d=cfg.latent.d_latent)
        z_t = torch.randn(4, cfg.latent.d_latent)
        t = torch.rand(4, 1)
        out = net(z_t, t)
        assert out.shape == (4, cfg.latent.d_latent)


class TestBuildAdapters:
    def test_all_modalities(self, cfg):
        adapters = build_adapters(cfg)
        assert "video_adapter" in adapters
        assert "temporal_pool" in adapters
        assert "phoneme_adapter" in adapters
        assert "phoneme_attn_pool" in adapters
        assert "phoneme_probe" in adapters
        assert "prosody_adapter" in adapters
        assert "velocity_vph" in adapters
        assert "velocity_phv" in adapters

    def test_drop_video(self, cfg):
        cfg.modality.video_enabled = False
        adapters = build_adapters(cfg)
        assert "video_adapter" not in adapters
        assert "temporal_pool" not in adapters
        assert "velocity_vph" not in adapters  # needs both video and phoneme
        assert "phoneme_adapter" in adapters

    def test_drop_prosody(self, cfg):
        cfg.modality.prosody_enabled = False
        adapters = build_adapters(cfg)
        assert "prosody_adapter" not in adapters
        assert "video_adapter" in adapters

    def test_no_probe_without_classes(self, cfg):
        cfg.latent.num_phoneme_classes = 0
        adapters = build_adapters(cfg)
        assert "phoneme_probe" not in adapters

    def test_save_and_reload(self, cfg, tmp_path):
        adapters = build_adapters(cfg)
        path = str(tmp_path / "test.pt")
        save_adapters(adapters, path)

        from encoding.adapters.registry import load_adapters
        loaded = load_adapters(path, cfg)
        assert set(loaded.keys()) == set(adapters.keys())

    def test_trainable_params_exclude(self, cfg):
        adapters = build_adapters(cfg)
        all_params = trainable_params(adapters)
        no_vel = trainable_params(adapters, exclude={"velocity_vph", "velocity_phv"})
        assert len(no_vel) < len(all_params)
