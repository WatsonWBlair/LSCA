"""Download utilities for the Seamless Interaction dataset.

HuggingFace repo: https://huggingface.co/datasets/facebook/seamless-interaction
Official tools: https://github.com/facebookresearch/seamless-interaction
"""

from __future__ import annotations

import re
from collections.abc import Generator
from pathlib import Path

from seamless_interaction.fs import DatasetConfig, SeamlessInteractionFS

_SESSION_PATTERN = re.compile(r"^(V\d+_S\d+)_(I(\d+))_P\w+$")

DEFAULT_LOCAL_DIR: Path = Path("datasets/seamless_interaction")



def download_pairs_iter(
    style: str,
    split: str,
    *,
    local_dir: Path = DEFAULT_LOCAL_DIR,
    num_pairs: int = 1,
) -> Generator[list[str], None, None]:
    """Download interaction pairs one at a time, yielding file IDs after each.

    Downloads each pair individually for clean-as-you-go processing.
    Each pair contains 2 participants with 4 files each (.mp4, .wav, .json, .npz).

    Args:
        style: "naturalistic" or "improvised".
        split: "train", "dev", "test", or "private".
        local_dir: Local directory to save downloaded files.
        num_pairs: Number of interaction pairs to download.

    Yields:
        List of file IDs (2 per pair) that were just downloaded.
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

    print(f"Sampled {len(pairs)} pair(s)")

    for i, pair in enumerate(pairs, start=1):
        print(f"Downloading pair {i}/{len(pairs)} ({len(pair)} files)...")
        fs.download_batch_from_s3(pair, local_dir=str(local_dir))
        yield pair


def download_sessions_iter(
    style: str,
    split: str,
    num_sessions: int = 28,
    *,
    local_dir: Path = DEFAULT_LOCAL_DIR,
    skip_sessions: set[str] | None = None,
) -> Generator[tuple[str, str, list[str], int], None, None]:
    """Download session interactions one at a time in chronological (archive_idx) order.

    Completes all interactions of one session before moving to the next.
    Each interaction's files are downloaded together, then yielded.

    Args:
        style: "naturalistic" or "improvised".
        split: "train", "dev", "test", or "private".
        num_sessions: Number of sessions to download.
        local_dir: Local directory to save downloaded files.

    Yields:
        (session_key, interaction_key, file_ids, session_total_interactions) tuples.
        session_key: e.g. "V00_S0700"
        interaction_key: e.g. "I00000131"
        file_ids: list of file IDs for both participants
        session_total_interactions: total interactions in this session
    """
    config = DatasetConfig(label=style, split=split, preferred_vendors_only=True)
    fs = SeamlessInteractionFS(config=config)

    print(f"Getting {num_sessions} session(s)...")
    session_groups = fs.get_session_groups(
        num_sessions=num_sessions,
        interactions_per_session=0,
        label=style,
        split=split,
    )

    if not session_groups:
        raise FileNotFoundError("No sessions found")

    print(f"Sampled {len(session_groups)} session(s)")

    for session_file_ids in session_groups:
        if not session_file_ids:
            continue

        # Extract session_key from first file_id
        m = _SESSION_PATTERN.match(session_file_ids[0])
        session_key = m.group(1) if m else "UNKNOWN"

        # Skip already-wrangled sessions
        s_part = session_key.split("_", 1)[1]  # e.g. "S0700"
        if skip_sessions and s_part in skip_sessions:
            print(f"Skipping {s_part} (already wrangled)")
            continue

        # Group file_ids by interaction key, preserving archive_idx order
        seen: dict[str, list[str]] = {}
        for fid in session_file_ids:
            fm = _SESSION_PATTERN.match(fid)
            if not fm:
                continue
            i_key = fm.group(2)  # e.g. "I00000131"
            if i_key not in seen:
                seen[i_key] = []
            seen[i_key].append(fid)

        session_total = len(seen)
        for i_key, file_ids in seen.items():
            print(
                f"Downloading {session_key} / {i_key} ({len(file_ids)} files)..."
            )
            fs.download_batch_from_s3(file_ids, local_dir=str(local_dir))
            yield session_key, i_key, file_ids, session_total