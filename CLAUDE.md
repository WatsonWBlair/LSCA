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
| `invoke install` | Install dependencies from requirements.txt |
| `invoke test` | Run tests with pytest |
| `invoke lint` | Run ruff linter |
| `invoke clean` | Remove build artifacts, caches, bytecode |
| `invoke freeze` | Freeze current dependencies to requirements.txt |
| `invoke wrangle` | Run Seamless Interaction data wrangling pipeline |
| `invoke wrangle-download` | Download/extract dataset archives from HuggingFace |
| `invoke wrangle-manifest` | Regenerate master JSON manifest from wrangled data |

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

- `src/data_wrangling/` — Data wrangling pipeline for Seamless Interaction dataset (stubs)
  - `types.py` — Shared dataclasses (`FileIdentifier`, `SpeakerFileSet`, `InteractionPair`, etc.)
  - `naming.py` — Filename parsing (`V{vendor}_S{session}_I{interaction}_P{participant}`)
  - `download.py` — HuggingFace download/extraction/streaming
  - `crop.py` — Video cropping using NPZ bounding boxes/keypoints
  - `repackage.py` — Reorganize raw files into interaction-centric layout
  - `manifest.py` — Master JSON manifest with VAD-based stitching order
  - `seamless_interaction.py` — Top-level pipeline orchestrator
- `tests/data_wrangling/` — Tests for the wrangling pipeline

## Project Status

Early stage. Data wrangling stubs are in place. See `litrature/project_overview_document.md` for full objectives and timeline.
