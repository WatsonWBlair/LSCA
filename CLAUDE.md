# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LSCA (Conversational Agents in Multimodal Embedded Latent Space) is a research project building a real-time multimodal understanding pipeline for natural Human-AI interactions in video call contexts. It integrates audio, video, and text modalities for voice-to-voice conversations using an Adapted Pretrained Encoders (APE) architecture.

Key research areas: APE-based ablation studies, streaming architecture for real-time video, evidential embeddings with Dempster-Shafer Theory, and interpretability via white-box reasoning with conflict tracing.

Prior tooling lives at https://github.com/WatsonWBlair/cs627 (Multimodal Joint Representation Learning Via APE Swarm).

## Development Commands

All task automation uses [Invoke](https://www.pyinvoke.org/):

**Development:**
| Command | Description |
|---------|-------------|
| `invoke install` | Update conda environment from environment.yml |
| `invoke test` | Run tests with pytest |
| `invoke lint` | Run ruff linter |
| `invoke clean` | Remove build artifacts, caches, bytecode |
| `invoke freeze` | Export conda environment to environment.yml |

**Data Wrangling** (clean-as-you-go processing):
| Command | Description |
|---------|-------------|
| `invoke wrangle-dev` | Small dev dataset (3 Seamless + 1 CANDOR) |
| `invoke wrangle-seamless --count N` | Process N Seamless Interaction pairs |
| `invoke wrangle-candor` | Process CANDOR dataset iteratively |

**CANDOR Utilities** (debugging/archival):
| Command | Description |
|---------|-------------|
| `invoke download-candor` | Download zip files only |
| `invoke extract-candor` | Extract audio from raw MKVs |

Run tests: `pytest path/to/test.py` or `pytest path/to/test.py::test_name`

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
  - `seamless_interaction/` — Seamless Interaction dataset utilities
    - `crop.py` — Face-centered video cropping using NPZ keypoints
    - `download.py` — Download interaction pairs from S3
    - `types.py` — Type definitions (`CropRegion`)
  - `candor/` — CANDOR Corpus dataset utilities
    - `download.py` — Download dataset zips from pre-signed S3 URLs
    - `extract.py` — Extract per-participant audio from raw MKVs
- `tests/data_wrangling/` — Pipeline tests

## Project Status

Data wrangling pipeline is functional. Downloads interaction pairs and crops videos to webcam-style framing. See `litrature/project_overview_document.md` for full objectives.
