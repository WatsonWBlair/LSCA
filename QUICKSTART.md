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

### Quick Dev Dataset

```bash
invoke wrangle-dev
```

Downloads and processes 3 Seamless Interaction pairs + 1 CANDOR part (~500MB working space).

### Seamless Interaction

```bash
# Process N interaction pairs
invoke wrangle-seamless --count 5
```

Each pair is downloaded, cropped to webcam framing, and cleaned up before the next.

### CANDOR Corpus

```bash
# Process all 166 parts (or use --count to limit)
invoke wrangle-candor --count 5
```

Requires pre-signed URLs in `src/data_wrangling/candor/file_urls.txt`.

## Output Structure

```
datasets/wrangled/              # Seamless Interaction output
  S{session}/
    I{interaction}_P{participant}.mp4
    I{interaction}_P{participant}.wav
    I{interaction}_P{participant}.json
    I{interaction}_P{participant}.npz

datasets/candor/                # CANDOR output
  {conversation-uuid}/
    processed/
      {user-id}.mp4
      {user-id}.wav
    metadata.json
    ...
```

## Common Commands

| Command | Description |
|---------|-------------|
| `invoke wrangle-dev` | Small dev dataset (3 Seamless + 1 CANDOR) |
| `invoke wrangle-seamless --count N` | Process N Seamless Interaction pairs |
| `invoke wrangle-candor` | Process CANDOR dataset iteratively |
| `invoke test` | Run pytest tests |
| `invoke lint` | Run ruff linter |

## Troubleshooting

**"Could not load filelist" error:**
The `seamless_interaction` library must be installed from source (not pip) to include the filelist.csv asset. See step 2.

**FFmpeg not found:**
Ensure the `lsca` conda environment is activated. FFmpeg is installed via conda.

**Video processing fails:**
Check that both `.mp4` and `.npz` files exist for each interaction.
