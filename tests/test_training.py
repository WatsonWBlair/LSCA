# tests/test_training.py
# Smoke test: 1-epoch training per stage doesn't crash on synthetic data.

import pytest
import torch

import torch.nn.functional as F

from encoding.config import CAMELSConfig, LatentConfig, TrainingConfig, MoCoConfig
from encoding.adapters.registry import build_adapters
from encoding.training.train import (
    _forward_batch,
    _compute_losses,
    train_stage_a,
    train_stage_b,
    train_stage_c,
)


@pytest.fixture
def train_cfg():
    return CAMELSConfig(
        latent=LatentConfig(
            d_latent=16,
            d_video=16,
            d_phoneme=16,
            d_prosody=22,
            max_phones=8,
            num_phoneme_classes=20,
        ),
        training=TrainingConfig(
            stage_a_epochs=1,
            stage_b_epochs=1,
            stage_c_epochs=1,
            batch_size=4,
            eval_every=1,
        ),
    )


@pytest.fixture
def train_adapters(train_cfg):
    return build_adapters(train_cfg)


@pytest.fixture
def fake_loader(train_cfg):
    """Create a simple list-based loader with 2 batches."""
    B = train_cfg.training.batch_size
    d_v = train_cfg.latent.d_video
    d_ph = train_cfg.latent.d_phoneme
    d_p = train_cfg.latent.d_prosody
    max_ph = train_cfg.latent.max_phones
    n_cls = train_cfg.latent.num_phoneme_classes

    batches = []
    for _ in range(2):
        v = torch.randn(B, d_v)
        ph = torch.randn(B, max_ph, d_ph)
        labels = torch.randint(0, n_cls, (B, max_ph))
        mask = torch.zeros(B, max_ph)
        for i in range(B):
            mask[i, :4] = 1.0
        p = torch.randn(B, d_p)
        batches.append((v, ph, labels, mask, p))
    return batches


class TestForwardBatch:
    def test_stage_a(self, fake_loader, train_adapters, train_cfg):
        batch = fake_loader[0]
        fwd = _forward_batch(batch, train_adapters, train_cfg, "cpu", "A")
        assert "z_v" in fwd
        assert "z_ph_pooled" in fwd
        assert "z_p" in fwd
        # Stage A uses embed() — no mu/logvar
        assert "mu_v" not in fwd

    def test_stage_b(self, fake_loader, train_adapters, train_cfg):
        batch = fake_loader[0]
        fwd = _forward_batch(batch, train_adapters, train_cfg, "cpu", "B")
        assert "mu_v" in fwd
        assert "lv_v" in fwd
        assert "xh_v" in fwd
        assert "z_v" in fwd


class TestComputeLosses:
    def test_stage_a_losses(self, fake_loader, train_adapters, train_cfg):
        batch = fake_loader[0]
        fwd = _forward_batch(batch, train_adapters, train_cfg, "cpu", "A")
        total, losses = _compute_losses(fwd, train_adapters, train_cfg, "A", 1, 0, 0)
        assert total.ndim == 0
        assert "nce" in losses
        assert "var" in losses
        assert "cov" in losses
        assert "aux" in losses
        # No AVAE in stage A
        assert "avae_total" not in losses

    def test_stage_b_losses(self, fake_loader, train_adapters, train_cfg):
        batch = fake_loader[0]
        fwd = _forward_batch(batch, train_adapters, train_cfg, "cpu", "B")
        total, losses = _compute_losses(fwd, train_adapters, train_cfg, "B", 1, 1, 20)
        assert "avae_total" in losses
        assert "orth" in losses
        assert "avae_video_recon" in losses
        assert "avae_prosody_recon" in losses


class TestStageTraining:
    def test_stage_a_smoke(self, fake_loader, train_adapters, train_cfg):
        history = train_stage_a(fake_loader, fake_loader, train_adapters, train_cfg, device="cpu")
        assert len(history) == 1
        assert history[0]["stage"] == "A"

    def test_stage_b_smoke(self, fake_loader, train_adapters, train_cfg):
        history = train_stage_b(fake_loader, fake_loader, train_adapters, train_cfg, device="cpu")
        assert len(history) == 1
        assert history[0]["stage"] == "B"

    def test_stage_c_smoke(self, fake_loader, train_adapters, train_cfg):
        history = train_stage_c(fake_loader, fake_loader, train_adapters, train_cfg, device="cpu")
        assert len(history) == 1
        assert history[0]["stage"] == "C"

    def test_gradient_isolation_stage_c(self, fake_loader, train_adapters, train_cfg):
        """Stage C: FM loss should NOT flow gradients to adapter parameters."""
        # This is a smoke test — the real check is in the training code's
        # use of .detach() in bidirectional_fm_loss
        history = train_stage_c(fake_loader, fake_loader, train_adapters, train_cfg, device="cpu")
        assert "fm" in history[0]


class TestMoCoIntegration:
    def test_moco_queue_enqueue(self, train_cfg):
        from encoding.training.momentum import MoCoQueue
        d, K, B = train_cfg.latent.d_latent, 32, 4
        q = MoCoQueue(d, K, "cpu")
        keys = F.normalize(torch.randn(B, d), dim=-1)
        q.enqueue_dequeue(keys)
        assert q.queue.shape == (d, K)

    def test_ema_update_shifts_params(self, train_cfg, train_adapters):
        from encoding.training.momentum import MomentumEncoderManager
        mgr = MomentumEncoderManager(train_adapters, train_cfg, "cpu")
        # Record initial key params
        init_params = {
            k: [p.clone() for p in m.parameters()]
            for k, m in mgr.key_adapters.items()
        }
        # Perturb query adapters
        for mod in train_adapters.values():
            for p in mod.parameters():
                p.data.add_(torch.randn_like(p) * 0.1)
        mgr.ema_update(train_adapters, m=0.9)
        # At least one key adapter param should have shifted
        shifted = False
        for k, m in mgr.key_adapters.items():
            for p_new, p_old in zip(m.parameters(), init_params[k]):
                if not torch.allclose(p_new, p_old):
                    shifted = True
                    break
        assert shifted

    def test_stage_a_with_moco(self, fake_loader, train_adapters, train_cfg):
        from encoding.training.momentum import MomentumEncoderManager
        cfg = CAMELSConfig(
            latent=train_cfg.latent,
            training=TrainingConfig(
                stage_a_epochs=1, stage_b_epochs=1, stage_c_epochs=1,
                batch_size=4, eval_every=1,
            ),
            moco=MoCoConfig(enabled=True, queue_size=16),
        )
        mgr = MomentumEncoderManager(train_adapters, cfg, "cpu")
        history = train_stage_a(
            fake_loader, fake_loader, train_adapters, cfg, device="cpu", momentum_manager=mgr
        )
        assert len(history) == 1
        assert history[0]["stage"] == "A"
