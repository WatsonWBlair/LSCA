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
) -> None:
    """Download interaction pairs from S3 via seamless_interaction library.

    Auto-samples interaction pairs from preferred vendors and downloads
    all four file types (.mp4, .wav, .json, .npz) for each participant.

    Args:
        style: "naturalistic" or "improvised".
        split: "train", "dev", "test", or "private".
        local_dir: Local directory to save downloaded files.
        num_pairs: Number of interaction pairs to download.

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

    if not pairs:
        raise FileNotFoundError("No interaction pairs found")

    # Flatten all pairs into a single list of file IDs
    file_ids = [fid for pair in pairs for fid in pair]
    print(f"Sampled {len(pairs)} pair(s), {len(file_ids)} file(s)")

    fs.download_batch_from_s3(file_ids, local_dir=str(local_dir))