"""Download utilities for the Seamless Interaction dataset.

HuggingFace repo: https://huggingface.co/datasets/facebook/seamless-interaction
Official tools: https://github.com/facebookresearch/seamless-interaction
"""

from __future__ import annotations

from pathlib import Path

from seamless_interaction.fs import DatasetConfig, SeamlessInteractionFS

DEFAULT_LOCAL_DIR: Path = Path("datasets/seamless_interaction")


def download_interaction(
    style: str,
    split: str,
    *,
    local_dir: Path = DEFAULT_LOCAL_DIR,
    num_pairs: int = 1,
) -> list[Path]:
    """Download interaction pairs from S3 via seamless_interaction library.

    Auto-samples interaction pairs from preferred vendors and downloads
    all four file types (.mp4, .wav, .json, .npz) for each participant.

    Args:
        style: "naturalistic" or "improvised".
        split: "train", "dev", "test", or "private".
        local_dir: Local directory to save downloaded files.
        num_pairs: Number of interaction pairs to download.

    Returns:
        List of Paths to downloaded file stems (without extension).

    Raises:
        FileNotFoundError: If no interaction pairs are found.
    """
    config = DatasetConfig(label=style, split=split, preferred_vendors_only=True)
    fs = SeamlessInteractionFS(config=config)

    print(f"Getting {num_pairs} random interaction pair(s)...")
    pairs = fs.get_interaction_pairs(
        num_pairs=num_pairs,
        split=split,
        label=style,
        preferred_vendors_only=True,
    )

    if not pairs or not pairs[0]:
        raise FileNotFoundError("No interaction pairs found")

    file_ids = pairs[0]
    print(f"Auto-sampled interaction pair: {file_ids}")

    print(f"Downloading to {local_dir}...")
    fs.download_batch_from_s3(file_ids, local_dir=str(local_dir))
    print(f"Downloaded interaction pair: {file_ids}")

    # Find downloaded files by searching for matching stems
    result = []
    for file_id in file_ids:
        matches = list(local_dir.rglob(f"{file_id}.mp4"))
        if matches:
            result.append(matches[0].with_suffix(""))
    return result