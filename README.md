# LSCA - Conversational Agents in Multimodal Embedded Latent Space

A research project building a real-time multimodal understanding pipeline for natural Human-AI interactions in video call contexts.

## Quick Start

```bash
# Setup environment
conda env create -f environment.yml
conda activate lsca

# Download a small dev dataset
invoke wrangle-dev
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

## Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Environment setup and basic usage |
| [DATA_WRANGLING.md](DATA_WRANGLING.md) | Dataset sources and processing details |

## Project Structure

```
src/
  data_wrangling/
    seamless_interaction/   # Seamless Interaction dataset
      crop.py               # Face-centered video cropping
      download.py           # Download from S3
      types.py              # Type definitions
    candor/                 # CANDOR Corpus dataset
      download.py           # Download and wrangling
      extract.py            # Audio extraction from MKVs
tests/
  data_wrangling/           # Pipeline tests
tasks.py                    # Invoke task definitions
```

## Common Commands

```bash
# Data wrangling (memory-efficient, clean-as-you-go)
invoke wrangle-dev                   # Small dev dataset (3 Seamless + 1 CANDOR)
invoke wrangle-seamless --count 10   # Process 10 Seamless Interaction pairs
invoke wrangle-candor                # Process full CANDOR dataset

# Development
invoke test                          # Run tests
invoke lint                          # Run ruff linter
```

## Related Work

Prior multimodal research: [cs627 - APE Swarm](https://github.com/WatsonWBlair/cs627)
