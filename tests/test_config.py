# tests/test_config.py
# Config validation, dimension consistency, modality toggle.

from encoding.config import CAMELSConfig, LatentConfig, ModalityConfig


def test_default_config():
    cfg = CAMELSConfig()
    assert cfg.latent.d_latent == 768
    assert cfg.latent.d_prosody == 22
    assert cfg.latent.max_phones == 50
    cfg.validate()


def test_custom_d_latent():
    cfg = CAMELSConfig(latent=LatentConfig(d_latent=512))
    assert cfg.latent.d_latent == 512
    cfg.validate()


def test_enabled_modalities_all():
    cfg = CAMELSConfig()
    assert cfg.enabled_modalities() == ["video", "phoneme", "prosody"]


def test_enabled_modalities_drop_one():
    cfg = CAMELSConfig(modality=ModalityConfig(prosody_enabled=False))
    assert cfg.enabled_modalities() == ["video", "phoneme"]
    cfg.validate()


def test_validate_fails_single_modality():
    cfg = CAMELSConfig(modality=ModalityConfig(
        video_enabled=True, phoneme_enabled=False, prosody_enabled=False,
    ))
    try:
        cfg.validate()
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_validate_fails_bad_adapter_type():
    cfg = CAMELSConfig(modality=ModalityConfig(video_adapter_type="transformer"))
    try:
        cfg.validate()
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_streaming_seg_total():
    cfg = CAMELSConfig()
    expected = (cfg.streaming.seg_hops + cfg.streaming.rc_hops) * cfg.streaming.hop
    assert cfg.streaming.seg_total == expected


def test_dimension_consistency(cfg):
    """All dimension references should be consistent within config."""
    assert cfg.latent.d_latent > 0
    assert cfg.latent.d_video > 0
    assert cfg.latent.d_phoneme > 0
    assert cfg.latent.d_prosody > 0
    assert cfg.latent.max_phones > 0
