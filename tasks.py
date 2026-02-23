import logging
import re
import shutil
from pathlib import Path

from invoke import task

from src.data_wrangling.crop import process_interaction

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


@task
def download(c, count=1):
    """Download Seamless Interaction pairs from S3.

    Args:
        num_pairs: Number of interaction pairs to download (default: 1)
    """
    from src.data_wrangling.download import download_interaction

    print(f"Downloading {count} interaction pair(s)...")
    download_interaction("improvised", "dev", num_pairs=int(count))


@task
def crop(c):
    """Crop all downloaded videos and copy companion files."""
    source_dir = Path('datasets/seamless_interaction')
    output_dir = Path('datasets/wrangled')

    npz_files = sorted(source_dir.rglob("*.npz"))
    print(f"Found {len(npz_files)} NPZ files")

    for npz_path in npz_files:
        if not npz_path.with_suffix('.mp4').exists():
            print(f"Skipping: {npz_path.stem} (no matching .mp4)")
            continue

        match = FILENAME_PATTERN.match(npz_path.stem)
        if not match:
            print(f"Skipping: {npz_path.stem} (doesn't match naming pattern)")
            continue

        vendor, session, interaction, participant = match.groups()
        short_name = f"I{interaction}_P{participant}"
        session_dir = output_dir / f"S{session}"

        # Skip if already processed
        if (session_dir / f"{short_name}.mp4").exists():
            print(f"Skipping: {npz_path.stem} (already cropped)")
            continue

        print(f"Cropping: {npz_path.stem} -> S{session}/{short_name}")
        process_interaction(npz_path, session_dir, short_name)

        # Copy companion files
        for ext in ['.wav', '.json', '.npz']:
            src = npz_path.with_suffix(ext)
            if src.exists():
                shutil.copy2(src, session_dir / f"{short_name}{ext}")


@task
def cleanup(c):
    """Remove all files from the seamless_interaction source directory."""
    source_dir = Path('datasets/seamless_interaction')

    if not source_dir.exists():
        print("Source directory doesn't exist")
        return

    count = 0
    for path in source_dir.rglob("*"):
        if path.is_file():
            path.unlink()
            count += 1

    # Remove empty directories
    for path in sorted(source_dir.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()

    print(f"Cleaned up {count} file(s)")

