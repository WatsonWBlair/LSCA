import logging
import re
import shutil
from pathlib import Path

from invoke import task

from src.data_wrangling.seamless_interaction.crop import process_interaction

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Pattern: V{vendor}_S{session}_I{interaction}_P{participant}
FILENAME_PATTERN = re.compile(r"^V(\d+)_S(\d+)_I(\d+)_P(\w+)$")


@task
def install(c):
    """Update conda environment from environment.yml."""
    c.run("conda env update -f environment.yml --prune")


@task
def test(c):
    """Run tests with pytest."""
    c.run("pytest")


@task
def clean(c):
    """Remove build artifacts, caches, and bytecode."""
    c.run("find . -type d -name __pycache__ -exec rm -rf {} +", warn=True)
    c.run("find . -type d -name .pytest_cache -exec rm -rf {} +", warn=True)
    c.run("rm -rf dist build *.egg-info", warn=True)


@task
def lint(c):
    """Run ruff linter."""
    c.run("ruff check .")


@task
def freeze(c):
    """Export current conda environment to environment.yml."""
    c.run("conda env export --from-history > environment.yml")


def wrangle_seamless_impl(count: int, style: str = "improvised", split: str = "dev"):
    """Core implementation for wrangling Seamless Interaction pairs.

    Downloads, crops, and cleans up one pair at a time to minimize disk usage.

    Args:
        count: Number of interaction pairs to process.
        style: "naturalistic" or "improvised".
        split: "train", "dev", "test".
    """
    from src.data_wrangling.seamless_interaction.download import download_pairs_iter

    source_dir = Path('datasets/seamless_interaction')
    output_dir = Path('datasets/wrangled')

    print(f"Processing {count} interaction pair(s)...")

    for pair_file_ids in download_pairs_iter(style, split, num_pairs=count):
        for file_id in pair_file_ids:
            npz_path = source_dir / style / split / f"{file_id}.npz"

            if not npz_path.exists() or not npz_path.with_suffix('.mp4').exists():
                print(f"Skipping {file_id} (missing files)")
                continue

            match = FILENAME_PATTERN.match(file_id)
            if not match:
                print(f"Skipping {file_id} (doesn't match pattern)")
                continue

            vendor, session, interaction, participant = match.groups()
            short_name = f"I{interaction}_P{participant}"
            session_dir = output_dir / f"S{session}"

            # Process if not already done
            if not (session_dir / f"{short_name}.mp4").exists():
                print(f"Cropping: {file_id} -> S{session}/{short_name}")
                process_interaction(npz_path, session_dir, short_name)

                # Copy companion files
                for ext in ['.wav', '.json', '.npz']:
                    src = npz_path.with_suffix(ext)
                    if src.exists():
                        shutil.copy2(src, session_dir / f"{short_name}{ext}")
            else:
                print(f"Skipping {file_id} (already processed)")

            # Delete source files immediately
            for ext in ['.mp4', '.wav', '.json', '.npz']:
                src = npz_path.with_suffix(ext)
                if src.exists():
                    src.unlink()
            print(f"Cleaned up: {file_id}")

    # Cleanup empty directories
    for path in sorted(source_dir.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()

    print(f"Wrangled {count} pair(s)")


@task(name="wrangle-dev")
def wrangle_dev(c):
    """Download and process a small dev dataset with minimal disk usage.

    Processes:
    - 3 Seamless Interaction pairs (crop + cleanup per-file)
    - 1 CANDOR zip part (download → extract audio → cleanup raw/)
    """
    from src.data_wrangling.candor.download import wrangle_candor

    # --- Seamless Interaction (3 pairs) ---
    print("=== Seamless Interaction ===")
    wrangle_seamless_impl(3, "improvised", "dev")

    # --- CANDOR (1 part) ---
    print("\n=== CANDOR ===")
    urls_file = Path('src/data_wrangling/candor/file_urls.txt')
    candor_dir = Path('datasets/candor')
    wrangle_candor(urls_file, candor_dir, start=1, count=1)

    print("\n=== Dev dataset ready ===")


@task(name="wrangle-seamless")
def wrangle_seamless(c, count=1, style="improvised", split="dev"):
    """Download, crop, and cleanup Seamless Interaction pairs one at a time.

    Processes each pair individually to minimize disk usage:
    1. Download pair (2 participants)
    2. Crop each participant's video
    3. Copy companion files
    4. Delete source files

    Args:
        count: Number of interaction pairs to process (default: 1)
        style: "naturalistic" or "improvised" (default: improvised)
        split: "train", "dev", "test" (default: dev)
    """
    wrangle_seamless_impl(int(count), style, split)


@task(name="download-candor")
def download_candor(c, start=1, count=None, extract=False):
    """Download CANDOR dataset parts from S3.

    Args:
        start: Part number to start from (default: 1)
        count: Number of parts to download (default: all 166)
        extract: Extract zip files after download (default: False)
    """
    from src.data_wrangling.candor.download import download_candor as dl_candor

    urls_file = Path('src/data_wrangling/candor/file_urls.txt')
    output_dir = Path('datasets/candor')

    start = int(start)
    count = int(count) if count is not None else None

    print(f"Downloading CANDOR parts starting from {start}...")
    dl_candor(urls_file, output_dir, start=start, count=count, extract=extract)


@task(name="extract-candor")
def extract_candor(c):
    """Extract per-participant audio from raw CANDOR MKV files."""
    from src.data_wrangling.candor.extract import extract_all_audio

    candor_dir = Path('datasets/candor')

    if not candor_dir.exists():
        print("CANDOR directory doesn't exist")
        return

    print("Extracting audio from CANDOR conversations...")
    count = extract_all_audio(candor_dir)
    print(f"Extracted audio from {count} conversation(s)")


@task(name="wrangle-candor")
def wrangle_candor_task(c, start=1, count=None):
    """Download, extract audio, and cleanup CANDOR parts iteratively.

    Processes each part one at a time to minimize disk usage:
    1. Download zip
    2. Extract zip
    3. Extract per-participant audio to WAV
    4. Remove raw/ directories
    5. Move to next part

    Args:
        start: Part number to start from (default: 1)
        count: Number of parts to process (default: all 166)
    """
    from src.data_wrangling.candor.download import wrangle_candor

    urls_file = Path('src/data_wrangling/candor/file_urls.txt')
    output_dir = Path('datasets/candor')

    start = int(start)
    count = int(count) if count is not None else None

    print(f"Wrangling CANDOR parts starting from {start}...")
    processed = wrangle_candor(urls_file, output_dir, start=start, count=count)
    print(f"Wrangled {processed} part(s)")
