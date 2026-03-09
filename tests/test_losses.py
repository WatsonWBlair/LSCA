# tests/test_losses.py
# Loss function correctness: shapes, gradient flow, known values, capacity ramp.

import torch
import pytest

from encoding.training.losses import (
    info_nce_loss,
    all_pairs_nce,
    avae_loss,
    get_capacity,
    cross_modal_orth_loss,
    variance_loss,
    covariance_loss,
    flow_matching_loss,
    bidirectional_fm_loss,
    phoneme_probe_loss,
    monitor_nce_pairs,
)
from encoding.adapters.velocity import VelocityNet
from encoding.adapters.phoneme import PhonemeProbeHead


class TestInfoNCE:
    def test_scalar_output(self):
        z_a = torch.randn(8, 32)
        z_b = torch.randn(8, 32)
        loss = info_nce_loss(z_a, z_b)
        assert loss.ndim == 0

    def test_gradient_flows(self):
        z_a = torch.randn(8, 32, requires_grad=True)
        z_b = torch.randn(8, 32, requires_grad=True)
        loss = info_nce_loss(z_a, z_b)
        loss.backward()
        assert z_a.grad is not None
        assert z_b.grad is not None

    def test_identical_inputs_low_loss(self):
        z = torch.randn(8, 32)
        loss = info_nce_loss(z, z, temperature=0.07)
        # Perfect alignment should produce low loss
        assert loss.item() < 1.0

    def test_temperature_effect(self):
        z_a = torch.randn(8, 32)
        z_b = torch.randn(8, 32)
        low_t = info_nce_loss(z_a, z_b, temperature=0.01)
        high_t = info_nce_loss(z_a, z_b, temperature=1.0)
        # Lower temperature amplifies logits, generally produces higher loss on random data
        assert low_t.item() != high_t.item()


class TestAllPairsNCE:
    def test_three_modalities(self):
        z_dict = {
            "video": torch.randn(8, 32),
            "phoneme": torch.randn(8, 32),
            "prosody": torch.randn(8, 32),
        }
        total, per_pair = all_pairs_nce(z_dict)
        assert total.ndim == 0
        assert len(per_pair) == 3  # C(3,2) = 3 pairs

    def test_two_modalities(self):
        z_dict = {
            "video": torch.randn(8, 32),
            "phoneme": torch.randn(8, 32),
        }
        total, per_pair = all_pairs_nce(z_dict)
        assert len(per_pair) == 1


class TestAVAELoss:
    def test_returns_all_terms(self):
        d_in, d_latent = 32, 16
        x = torch.randn(4, d_in)
        x_hat = torch.randn(4, d_in)
        mu = torch.randn(4, d_latent)
        logvar = torch.randn(4, d_latent)
        z = torch.randn(4, d_latent)
        z_prime = torch.randn(4, d_latent)

        result = avae_loss(x, x_hat, mu, logvar, z, z_prime, capacity=5.0, beta_cap=1.0)
        assert "recon" in result
        assert "kl" in result
        assert "kl_cap" in result
        assert "consist" in result
        assert "total" in result
        for v in result.values():
            assert v.ndim == 0

    def test_capacity_control(self):
        x = torch.randn(4, 32)
        mu = torch.zeros(4, 16)
        logvar = torch.zeros(4, 16)
        z = torch.randn(4, 16)

        r0 = avae_loss(x, x, mu, logvar, z, z, capacity=0.0, beta_cap=1.0)
        r5 = avae_loss(x, x, mu, logvar, z, z, capacity=5.0, beta_cap=1.0)
        # Different capacity targets should produce different kl_cap
        assert r0["kl_cap"].item() != r5["kl_cap"].item()

    def test_gradient_flows(self):
        x = torch.randn(4, 32, requires_grad=True)
        x_hat = torch.randn(4, 32, requires_grad=True)
        mu = torch.randn(4, 16, requires_grad=True)
        logvar = torch.randn(4, 16, requires_grad=True)
        z = torch.randn(4, 16, requires_grad=True)
        z_prime = torch.randn(4, 16, requires_grad=True)

        result = avae_loss(x, x_hat, mu, logvar, z, z_prime)
        result["total"].backward()
        assert x_hat.grad is not None
        assert mu.grad is not None


class TestCapacityRamp:
    def test_zero_at_start(self):
        assert get_capacity(0, 1, 20, 25.0) == 0.0

    def test_max_at_end(self):
        assert get_capacity(20, 1, 20, 25.0) == pytest.approx(25.0, rel=0.01)

    def test_linear_midpoint(self):
        c = get_capacity(10, 1, 20, 20.0)
        # At epoch 10 out of 1-20, progress = 9/19 ≈ 0.47
        assert 5.0 < c < 15.0

    def test_clamps_past_end(self):
        c = get_capacity(100, 1, 20, 25.0)
        assert c == pytest.approx(25.0, rel=0.01)


class TestGeometricLosses:
    def test_orth_loss_scalar(self):
        z_list = [torch.randn(8, 32) for _ in range(3)]
        loss = cross_modal_orth_loss(z_list)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_variance_loss_scalar(self):
        z_list = [torch.randn(8, 32) for _ in range(3)]
        loss = variance_loss(z_list)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_variance_loss_collapsed_embedding(self):
        # All same value → high variance loss (low std < gamma)
        z_list = [torch.ones(8, 32)]
        loss = variance_loss(z_list, gamma=1.0)
        assert loss.item() > 0

    def test_covariance_loss_scalar(self):
        z_list = [torch.randn(8, 32) for _ in range(3)]
        loss = covariance_loss(z_list)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_covariance_loss_identity_low(self):
        # Random data with high variance should have some covariance
        z_list = [torch.randn(100, 32)]
        loss = covariance_loss(z_list)
        # Should be finite and reasonable
        assert not torch.isnan(loss)


class TestFlowMatching:
    def test_fm_loss_scalar(self):
        d = 32
        net = VelocityNet(d=d)
        z_src = torch.randn(4, d)
        z_tgt = torch.randn(4, d)
        loss = flow_matching_loss(z_src, z_tgt, net)
        assert loss.ndim == 0

    def test_fm_gradient_to_net(self):
        d = 32
        net = VelocityNet(d=d)
        z_src = torch.randn(4, d)
        z_tgt = torch.randn(4, d)
        loss = flow_matching_loss(z_src, z_tgt, net)
        loss.backward()
        # VelocityNet params should have gradients
        for p in net.parameters():
            assert p.grad is not None

    def test_bidirectional_fm(self):
        d = 32
        vel_vph = VelocityNet(d=d)
        vel_phv = VelocityNet(d=d)
        z_v = torch.randn(4, d)
        z_ph = torch.randn(4, d)
        loss = bidirectional_fm_loss(z_v, z_ph, vel_vph, vel_phv)
        assert loss.ndim == 0
        loss.backward()


class TestPhonemeProbe:
    def test_loss_scalar(self):
        d = 32
        n_classes = 40
        probe = PhonemeProbeHead(d=d, n_classes=n_classes)
        z_ph = torch.randn(4, 10, d)
        labels = torch.randint(0, n_classes, (4, 10))
        mask = torch.ones(4, 10)
        mask[:, 7:] = 0

        loss = phoneme_probe_loss(z_ph, labels, mask, probe)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_empty_mask_zero_loss(self):
        d = 32
        n_classes = 40
        probe = PhonemeProbeHead(d=d, n_classes=n_classes)
        z_ph = torch.randn(4, 10, d)
        labels = torch.randint(0, n_classes, (4, 10))
        mask = torch.zeros(4, 10)

        loss = phoneme_probe_loss(z_ph, labels, mask, probe)
        assert loss.item() == 0.0


class TestMonitorNCE:
    def test_returns_dict(self):
        z_dict = {
            "video": torch.randn(8, 32),
            "phoneme": torch.randn(8, 32),
            "prosody": torch.randn(8, 32),
        }
        result = monitor_nce_pairs(z_dict)
        assert len(result) == 3
        for v in result.values():
            assert isinstance(v, float)
