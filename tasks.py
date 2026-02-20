from invoke import task


@task
def install(c):
    """Install dependencies from requirements.txt."""
    c.run("pip install -r requirements.txt")


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
    """Freeze current dependencies to requirements.txt."""
    c.run("pip freeze > requirements.txt")


@task
def wrangle(c, style=None, split=None, no_crop=False):
    """Run the Seamless Interaction data wrangling pipeline.

    Usage:
        invoke wrangle                      # wrangle all available data
        invoke wrangle --style improvised   # only improvised sessions
        invoke wrangle --split dev          # only the dev split
        invoke wrangle --no-crop            # skip video cropping
    """
    raise NotImplementedError("Wrangling pipeline not yet implemented")


@task(name="wrangle-download")
def wrangle_download(c, style="improvised", split="dev", batch=None, archive=None):
    """Download and extract Seamless Interaction dataset archives from HuggingFace.

    Usage:
        invoke wrangle-download --batch 0001 --archive 0004
        invoke wrangle-download --style naturalistic --split train
    """
    raise NotImplementedError("Download pipeline not yet implemented")


@task(name="wrangle-manifest")
def wrangle_manifest(c):
    """Regenerate the master JSON manifest from already-wrangled data."""
    raise NotImplementedError("Manifest generation not yet implemented")
