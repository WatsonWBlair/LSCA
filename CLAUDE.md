# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LSCA (Conversational Agents in Multimodal Embedded Latent Space) is a research project building a real-time multimodal understanding pipeline for natural Human-AI interactions in video call contexts. It integrates audio, video, and text modalities for voice-to-voice conversations using an Adapted Pretrained Encoders (APE) architecture.

Key research areas: APE-based ablation studies, streaming architecture for real-time video, evidential embeddings with Dempster-Shafer Theory, and interpretability via white-box reasoning with conflict tracing.

Prior tooling lives at https://github.com/WatsonWBlair/cs627 (Multimodal Joint Representation Learning Via APE Swarm).

## Development Commands

All task automation uses [Invoke](https://www.pyinvoke.org/):

| Command | Description |
|---------|-------------|
| `invoke install` | Update conda environment from environment.yml |
| `invoke test` | Run tests with pytest |
| `invoke lint` | Run ruff linter |
| `invoke clean` | Remove build artifacts, caches, bytecode |
| `invoke freeze` | Export conda environment to environment.yml |
| `invoke download --count N` | Download N interaction pairs from S3 |
| `invoke crop` | Crop videos and copy companion files |
| `invoke cleanup` | Remove source files after processing |

Run a single test file: `pytest path/to/test_file.py`
Run a single test: `pytest path/to/test_file.py::test_function_name`

## Tech Stack

- **Python 3.11** with PyTorch, NumPy, OpenCV
- **Testing:** pytest
- **Linting:** ruff (via `invoke lint`), pylint (via Codacy)
- **Code analysis:** Codacy with semgrep + trivy (vulnerability scanning)

## Data Sources

Two primary datasets (see `DATA_WRANGLING.md` for details):
- **Seamless_Interaction:** 4,000+ hours of in-person face-to-face interaction footage, requiring cropping to video-call framing (mid-chest up)
- **CANDOR Corpus:** 1,650 video chat conversations between strangers with rich metadata

All wrangled data should resemble raw webcam footage: mid-chest and up, centered on the speaker's face, cropped to appropriate aspect ratio.

## Code Structure

- `src/data_wrangling/` — Video preprocessing pipeline
  - `crop.py` — Face-centered video cropping using NPZ keypoints
  - `download.py` — Download interaction pairs from S3
  - `types.py` — Type definitions (`CropRegion`)
- `tests/data_wrangling/` — Pipeline tests

## Project Status

Data wrangling pipeline is functional. Downloads interaction pairs and crops videos to webcam-style framing. See `litrature/project_overview_document.md` for full objectives.
