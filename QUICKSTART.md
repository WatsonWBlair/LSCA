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

The Seamless Interaction dataset requires the `seamless_interaction` library with its asset files:

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

## 4. Download and Process Data

All wrangling commands use memory-efficient processing—each item is fully processed before the next is downloaded.

### Seamless Interaction

```bash
# Process N interaction pairs
invoke wrangle-seamless --count 5
```

Each pair is downloaded, cropped to webcam framing, and cleaned up before the next.

### CANDOR Corpus

```bash
# Download, extract, and wrangle in one step
invoke wrangle-candor --count 5

# Or, if zips are already downloaded, backfill directly into wrangled output
invoke wrangle-candor-to-wrangled --count 5
```

## 5. Pre-generate Tokens

After wrangling, run the frozen encoders once over all wrangled data to produce cached tokens for fast training:

```bash
invoke generate-wrangled-tokens
```

Outputs are saved to `datasets/pregenerated/`.

## Output Structure

```
datasets/wrangled/
  S{session}/                         # Seamless Interaction output
    I{interaction}_P{participant}.mp4
    I{interaction}_P{participant}.wav
    I{interaction}_P{participant}.json
    I{interaction}_P{participant}.npz
  C{part_num}/                        # CANDOR output
    {uuid}_{user_id}.mp4
    {uuid}_{user_id}.wav
    {uuid}_{user_id}.json             # VAD, transcript, survey, audio/video features
```

## Common Commands

| Command | Description |
|---------|-------------|
| `invoke wrangle-seamless --count N` | Process N Seamless Interaction pairs |
| `invoke download-candor` | Download CANDOR zips (requires file_urls.txt) |
| `invoke wrangle-candor` | Download, extract, and wrangle CANDOR parts |
| `invoke wrangle-candor-to-wrangled` | Backfill already-downloaded CANDOR parts into wrangled output |
| `invoke generate-wrangled-tokens` | Pregenerate backbone tokens from datasets/wrangled/ |
| `invoke test` | Run pytest tests |
| `invoke lint` | Run ruff linter |

## Troubleshooting

**"Could not load filelist" error:**
The `seamless_interaction` library must be installed from source (not pip) to include the filelist.csv asset. See step 2.

**FFmpeg not found:**
Ensure the `lsca` conda environment is activated. FFmpeg is installed via conda.

**Video processing fails (Seamless Interaction):**
Check that both `.mp4` and `.npz` files exist for each interaction. CANDOR wrangled output uses `.mp4` + `.json` (no `.npz`).
