from pathlib import Path

from invoke import task

from src.data_wrangling.crop import (
    load_keypoints,
    compute_crop_region,
    crop_video,
    get_video_dimensions,
)


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
def wrangle(c):
    """Run crop preview on all interaction files in local dataset."""
    local_dir = Path('datasets/seamless_interaction')
    output_dir = Path('datasets/wrangled')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all .npz files
    npz_files = sorted(local_dir.rglob("*.npz"))
    print(f"Found {len(npz_files)} NPZ files")

    for npz_path in npz_files:
        video_path = npz_path.with_suffix('.mp4')

        if not video_path.exists():
            print(f"Skipping: {npz_path.stem} (no matching .mp4)")
            continue

        print(f"Processing: {video_path.name}")

        keypoints, validity = load_keypoints(npz_path)
        print(f"  Loaded {len(keypoints)} frames, {validity.sum()} valid")

        frame_width, frame_height = get_video_dimensions(video_path)
        print(f"  Video dimensions: {frame_width}x{frame_height}")

        crop_region = compute_crop_region(keypoints, validity, frame_width, frame_height)
        print(f"  Crop region: x={crop_region.x}, y={crop_region.y}, "
              f"w={crop_region.width}, h={crop_region.height}")

        output_path = output_dir / f"{video_path.stem}_cropped.mp4"
        crop_video(video_path, output_path, crop_region)
        print(f"  Output: {output_path}")


@task(name="wrangle-download")
def wrangle_download(c, style="improvised", split="dev", num_pairs=1):
    """Download interaction pairs from HuggingFace.

    Args:
        style: "naturalistic" or "improvised" (default: improvised)
        split: "train", "dev", "test", or "private" (default: dev)
        num_pairs: Number of interaction pairs to download (default: 1)
    """
    from src.data_wrangling.download import download_interaction

    print(f"Downloading {num_pairs} interaction pair(s) from {style}/{split}...")
    paths = download_interaction(style, split, num_pairs=int(num_pairs))
    print(f"Downloaded {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")


