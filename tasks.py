"""Invoke task definitions for LSCA project."""

import logging
from pathlib import Path

from invoke import task

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Paths
CANDOR_URLS = Path('src/data_wrangling/candor/file_urls.txt')
CANDOR_DIR = Path('datasets/candor')


# =============================================================================
# Development
# =============================================================================

@task
def install(c):
    """Update conda environment from environment.yml."""
    c.run("conda env update -f environment.yml --prune")


@task
def test(c):
    """Run tests with pytest."""
    c.run("pytest")


@task
def lint(c):
    """Run ruff linter."""
    c.run("ruff check .")


@task
def clean(c):
    """Remove build artifacts and caches."""
    c.run("find . -type d -name __pycache__ -exec rm -rf {} +", warn=True)
    c.run("find . -type d -name .pytest_cache -exec rm -rf {} +", warn=True)
    c.run("rm -rf dist build *.egg-info", warn=True)


@task
def freeze(c):
    """Export conda environment to environment.yml."""
    c.run("conda env export --from-history > environment.yml")


# =============================================================================
# Data Wrangling
# =============================================================================

@task(name="wrangle-dev")
def wrangle_dev(c):
    """Process small dev dataset (3 Seamless + 1 CANDOR)."""
    from src.data_wrangling.seamless_interaction.wrangle import wrangle_seamless
    from src.data_wrangling.candor.download import wrangle_candor

    print("=== Seamless Interaction ===")
    wrangle_seamless(3)

    print("\n=== CANDOR ===")
    wrangle_candor(CANDOR_URLS, CANDOR_DIR, count=1)

    print("\n=== Dev dataset ready ===")


@task(name="wrangle-seamless")
def wrangle_seamless_task(c, count=1, style="improvised", split="dev"):
    """Process N Seamless Interaction pairs (clean-as-you-go)."""
    from src.data_wrangling.seamless_interaction.wrangle import wrangle_seamless
    wrangle_seamless(int(count), style, split)


@task(name="wrangle-candor")
def wrangle_candor_task(c, start=1, count=None):
    """Process CANDOR parts iteratively (clean-as-you-go)."""
    from src.data_wrangling.candor.download import wrangle_candor

    start = int(start)
    count = int(count) if count else None
    processed = wrangle_candor(CANDOR_URLS, CANDOR_DIR, start=start, count=count)
    print(f"Wrangled {processed} part(s)")


# =============================================================================
# CANDOR Utilities
# =============================================================================

@task(name="download-candor")
def download_candor_task(c, start=1, count=None, extract=False):
    """Download CANDOR zips without processing."""
    from src.data_wrangling.candor.download import download_candor

    start = int(start)
    count = int(count) if count else None
    download_candor(CANDOR_URLS, CANDOR_DIR, start=start, count=count, extract=extract)


@task(name="extract-candor")
def extract_candor_task(c):
    """Extract audio from raw CANDOR MKVs."""
    from src.data_wrangling.candor.extract import extract_all_audio

    if not CANDOR_DIR.exists():
        print("CANDOR directory doesn't exist")
        return
    count = extract_all_audio(CANDOR_DIR)
    print(f"Extracted audio from {count} conversation(s)")
