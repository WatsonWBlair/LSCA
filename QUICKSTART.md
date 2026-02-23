# LSCA Quickstart Guide

Get the project running from scratch.

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git

## 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/WatsonWBlair/LSCA.git
cd LSCA

# Create and activate conda environment
conda env create -f environment.yml
conda activate lsca
```

## 2. Install seamless_interaction Library

The dataset download requires the `seamless_interaction` library with its asset files:

```bash
# Clone the library (if not already present)
git clone https://github.com/facebookresearch/seamless_interaction.git ../seamless_interaction

# Install in development mode
pip install -e ../seamless_interaction
```

## 3. Verify Installation

```bash
# Check FFmpeg is available
ffmpeg -version

# Check Python dependencies
python -c "import torch; import ffmpeg; import seamless_interaction; print('All imports OK')"

# Run tests
invoke test
```

## 4. Download Dataset

Download interaction pairs from HuggingFace:

```bash
# Download 1 random interaction pair (2 participants)
invoke wrangle-download

# Download multiple pairs
invoke wrangle-download --num-pairs 5
```

Files are saved to `datasets/seamless_interaction/`. Each interaction includes:
- `.mp4` - Video file
- `.wav` - Audio file
- `.json` - Transcript + VAD metadata
- `.npz` - Pre-computed keypoints

## 5. Process Videos

Crop all downloaded videos to webcam-style framing:

```bash
invoke wrangle
```

Output cropped videos are written to `datasets/wrangled/`.

## Common Commands

| Command | Description |
|---------|-------------|
| `invoke wrangle-download` | Download interaction pairs |
| `invoke wrangle` | Crop videos to webcam framing |
| `invoke test` | Run pytest tests |
| `invoke lint` | Run ruff linter |

## Troubleshooting

**"Could not load filelist" error:**
The `seamless_interaction` library must be installed from source (not pip) to include the filelist.csv asset. See step 2.

**FFmpeg not found:**
Ensure the `lsca` conda environment is activated. FFmpeg is installed via conda.

**Video processing fails:**
Check that both `.mp4` and `.npz` files exist for each interaction.
