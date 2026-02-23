# LSCA - Conversational Agents in Multimodal Embedded Latent Space

A research project building a real-time multimodal understanding pipeline for natural Human-AI interactions in video call contexts.

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for setup instructions.

## Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Environment setup and basic usage |
| [DATA_WRANGLING.md](DATA_WRANGLING.md) | Dataset processing details |

## Project Structure

```
src/
  data_wrangling/       # Video preprocessing pipeline
    crop.py             # Face-centered video cropping
    download.py         # Dataset download from S3
    types.py            # Shared type definitions
tests/
  data_wrangling/       # Pipeline tests
tasks.py                # Invoke task definitions
```

## Common Commands

```bash
invoke download --count 5  # Download interaction pairs from S3
invoke crop                # Crop videos and copy companion files
invoke cleanup             # Remove source files after processing
invoke test                # Run tests
invoke lint                # Run ruff linter
```

## Related Work

Prior multimodal research: [cs627 - APE Swarm](https://github.com/WatsonWBlair/cs627)
