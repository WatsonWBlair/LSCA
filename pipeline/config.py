# CAMELS Pipeline Configuration
# All constants are defined here. Import from this module everywhere.

# ── Scheduler ──────────────────────────────────────────────────────────────
WINDOW_SEC   = 2.0    # seconds of audio/video in each chunk
STRIDE_SEC   = 1.0    # seconds between scheduler ticks (50% overlap)
RMS_SILENCE  = 0.01   # audio.std() threshold — below this → silent chunk

# ── Audio ───────────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000          # Hz — required by wav2vec2 and Emformer
AUDIO_BUFFER_SEC   = 8              # ring buffer keeps last N seconds of audio

# ── Emformer RNN-T ─────────────────────────────────────────────────────────
HOP            = 160                # samples per hop (10 ms at 16 kHz)
SEG_HOPS       = 16                 # hops per segment → 2560 samples = 160 ms
RC_HOPS        = 4                  # right-context hops → 640 samples = 40 ms
SEG_TOTAL      = (SEG_HOPS + RC_HOPS) * HOP   # 3200 samples per Emformer call
ASR_BEAM_WIDTH = 10
# Reset ASR state after this many seconds of silence (prevents unbounded memory)
RESET_SILENCE_SEC = 30.0

# ── Video ────────────────────────────────────────────────────────────────────
FRAME_BUFFER_SEC  = 8              # ring buffer keeps last N seconds of frames
TARGET_FPS        = 30             # expected camera fps (used for buffer sizing)
MARLIN_FRAMES     = 16             # exactly 16 frames per MARLIN window
MARLIN_SIZE       = 224            # spatial resolution
IMAGENET_MEAN     = [0.485, 0.456, 0.406]
IMAGENET_STD      = [0.229, 0.224, 0.225]
FACE_DETECT_CONF  = 0.3            # MediaPipe min_detection_confidence

# ── MARLIN model ────────────────────────────────────────────────────────────
MARLIN_MODEL_NAME = "marlin_vit_base_ytf"   # 768-D embeddings
D_VIDEO           = 768                      # MARLIN base output dim

# ── wav2vec2 ────────────────────────────────────────────────────────────────
WAV2VEC2_MODEL = "facebook/wav2vec2-base"   # 768-D; use -large for 1024-D
D_AUDIO        = 768                         # matches wav2vec2-base hidden size

# ── SONAR ────────────────────────────────────────────────────────────────────
SONAR_ENCODER   = "text_sonar_basic_encoder"
SONAR_TOKENIZER = "text_sonar_basic_encoder"
DEFAULT_LANG    = "eng_Latn"                 # BCP-47 + script code for SONAR

# ── Latent space ─────────────────────────────────────────────────────────────
D_LATENT  = 1024    # shared latent dimension — matches SONAR natively
D_PROSODY = 18      # actual librosa feature count: 2 pitch + 1 rate + 1 rms + 1 centroid + 1 rolloff + 12 MFCCs

# ── AVAE adapter hidden sizes ────────────────────────────────────────────────
HIDDEN_HIGH = 256   # video / audio / text adapters
HIDDEN_PROS = 64    # prosody adapter — small input, prevent posterior collapse

# ── Output files ─────────────────────────────────────────────────────────────
CHUNK_FILE      = "chunks.json"
TRANSCRIPT_FILE = "transcript.json"
ZV_FILE         = "z_v.npy"
ZA_FILE         = "z_a.npy"
ZP_FILE         = "z_p.npy"
ZT_FILE         = "z_t.npy"
PROSODY_STATS   = "prosody_stats.json"
