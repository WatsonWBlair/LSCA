# encoding/config.py
# Centralized typed configuration for the CAMELS pipeline.
# Every dimension, loss weight, and toggle flows from here.
# No module should hardcode dimensions — always read from config.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Modality(IntEnum):
    """Integer-tagged modality identifiers for the shared latent space."""
    VIDEO = 0
    PHONEME = 1
    PROSODY = 2


@dataclass
class LatentConfig:
    """Dimensions of the shared latent space and encoder outputs."""
    d_latent: int = 768
    d_video: int = 768             # MARLIN ViT-Base output
    d_phoneme: int = 1024          # wav2vec2-lv-60-espeak-cv-ft hidden size (1024, not 768)
    d_prosody: int = 22            # librosa features (expanded)
    max_phones: int = 50           # 99th-percentile phoneme count per 2s chunk
    num_phoneme_classes: int = 0   # 0 = auto-detect from CTC model vocab at load time


@dataclass
class ModalityConfig:
    """Which modalities are active and what adapter type each uses."""
    video_enabled: bool = True
    phoneme_enabled: bool = True
    prosody_enabled: bool = True
    video_adapter_type: str = "avae"
    phoneme_adapter_type: str = "linear"
    prosody_adapter_type: str = "avae"


@dataclass
class AdapterConfig:
    """Hidden-layer sizes for adapter networks."""
    hidden_high: int = 256         # Video AVAE hidden
    hidden_prosody: int = 64       # Prosody AVAE hidden (small input → small hidden)
    hidden_probe: int = 256        # Phoneme probe MLP hidden


@dataclass
class TrainingConfig:
    """Hyperparameters for the 3-stage training curriculum."""
    lr: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    sigma_min: float = 1e-4
    # Geometric loss weights
    lambda_orth: float = 0.01
    lambda_var: float = 0.04
    lambda_cov: float = 0.04
    gamma_var: float = 1.0
    # Phoneme probe weight
    lambda_aux: float = 0.1
    # Capacity-controlled KL
    c_max_video: float = 25.0
    c_max_prosody: float = 10.0
    beta_cap: float = 1.0
    # Stage epochs
    stage_a_epochs: int = 20
    stage_b_epochs: int = 20
    stage_c_epochs: int = 20
    eval_every: int = 5
    batch_size: int = 64
    checkpoint_dir: str = "checkpoints"


@dataclass
class StreamingConfig:
    """Parameters for the live-streaming scheduler and ring buffers."""
    window_sec: float = 2.0
    stride_sec: float = 1.0
    rms_silence: float = 0.01
    sample_rate: int = 16000
    audio_buffer_sec: int = 8
    frame_buffer_sec: int = 8
    target_fps: int = 30
    # MARLIN video encoder
    marlin_frames: int = 16
    marlin_size: int = 224
    imagenet_mean: tuple = (0.485, 0.456, 0.406)
    imagenet_std: tuple = (0.229, 0.224, 0.225)
    face_detect_conf: float = 0.3
    # Emformer ASR
    hop: int = 160
    seg_hops: int = 16
    rc_hops: int = 4
    asr_beam_width: int = 10
    reset_silence_sec: float = 30.0
    # Model identifiers
    marlin_model_name: str = "marlin_vit_base_ytf"
    wav2vec2_ctc_model: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    default_lang: str = "eng_Latn"

    @property
    def seg_total(self) -> int:
        return (self.seg_hops + self.rc_hops) * self.hop


@dataclass
class ExportConfig:
    """Output file names for .npy embeddings and metadata."""
    zv_file: str = "z_v.npy"
    zph_file: str = "z_ph.npy"
    zp_file: str = "z_p.npy"
    chunk_file: str = "chunks.json"
    transcript_file: str = "transcript.json"
    phonemes_file: str = "phonemes.json"
    prosody_stats_file: str = "prosody_stats.json"


@dataclass
class CAMELSConfig:
    """Top-level configuration aggregating all sub-configs."""
    latent: LatentConfig = field(default_factory=LatentConfig)
    modality: ModalityConfig = field(default_factory=ModalityConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def enabled_modalities(self) -> list[str]:
        """Return list of enabled modality names."""
        mods = []
        if self.modality.video_enabled:
            mods.append("video")
        if self.modality.phoneme_enabled:
            mods.append("phoneme")
        if self.modality.prosody_enabled:
            mods.append("prosody")
        return mods

    def validate(self):
        """Runtime validation of config consistency."""
        assert self.latent.d_latent > 0, "d_latent must be positive"
        assert self.latent.max_phones > 0, "max_phones must be positive"
        assert len(self.enabled_modalities()) >= 2, "At least 2 modalities required for contrastive learning"
        for name in ("video", "phoneme", "prosody"):
            atype = getattr(self.modality, f"{name}_adapter_type")
            assert atype in ("avae", "linear"), f"Unknown adapter type for {name}: {atype}"
