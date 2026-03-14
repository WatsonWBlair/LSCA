#!/usr/bin/env python3
# scripts/backfill_source_tag.py
# Walk datasets/wrangled/*/ and patch missing "source" / "id" fields in .json files.
#
# Source is inferred from session dir prefix:
#   S  -> "seamless_interaction"
#   C  -> "candor"
#   MS -> "cmu_mosei"
#
# Usage:
#   python scripts/backfill_source_tag.py [--wrangled-root datasets/wrangled] [--dry-run]

import argparse
import json
from pathlib import Path


_PREFIX_TO_SOURCE = {
    "MS": "cmu_mosei",
    "S": "seamless_interaction",
    "C": "candor",
}


def infer_source(session_dir_name: str) -> str | None:
    """Infer dataset source tag from session directory name prefix."""
    for prefix, source in _PREFIX_TO_SOURCE.items():
        if session_dir_name.startswith(prefix):
            return source
    return None


def backfill(wrangled_root: str, dry_run: bool) -> None:
    root = Path(wrangled_root)
    patched = 0
    visited = 0

    for session_dir in sorted(root.iterdir()):
        if not session_dir.is_dir():
            continue

        source = infer_source(session_dir.name)
        if source is None:
            print(f"  [skip] Unknown prefix: {session_dir.name}")
            continue

        for json_path in sorted(session_dir.glob("*.json")):
            visited += 1
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            changed = False

            if "source" not in data:
                data["source"] = source
                changed = True

            if "id" not in data:
                data["id"] = f"{source}_{json_path.stem}"
                changed = True

            if changed:
                if not dry_run:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                patched += 1
                print(f"  {'[dry-run] ' if dry_run else ''}patched: {json_path}")

    print(f"\nDone. Visited {visited} JSON file(s), patched {patched}.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Backfill missing source/id tags in wrangled JSON files"
    )
    p.add_argument("--wrangled-root", default="datasets/wrangled",
                   help="Root of wrangled data (default: datasets/wrangled)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be changed without writing")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    backfill(args.wrangled_root, args.dry_run)
