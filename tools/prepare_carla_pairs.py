"""Create stem-aligned training pairs from a CARLA capture manifest."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", default="output_images")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output", default="data/carla_development")
    args = parser.parse_args()

    capture_root = Path(args.capture_root).resolve()
    output = Path(args.output).resolve()
    clean_dir = output / "clean"
    hazy_dir = output / "hazy"
    clean_dir.mkdir(parents=True, exist_ok=True)
    hazy_dir.mkdir(parents=True, exist_ok=True)

    manifest = (
        Path(args.manifest).resolve()
        if args.manifest
        else capture_root / "metadata.csv"
    )
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("CARLA metadata contains no samples")

    for row in rows:
        stem = row.get("sample_id") or row["id"]
        clear_value = row.get("clear_path") or row["clear_source"]
        hazy_value = row.get("hazy_path") or row["hazy_source"]
        clear_source = Path(clear_value)
        hazy_source = Path(hazy_value)
        if not clear_source.is_absolute():
            clear_source = Path.cwd() / clear_source
        if not hazy_source.is_absolute():
            hazy_source = Path.cwd() / hazy_source
        if not clear_source.is_file() or not hazy_source.is_file():
            raise FileNotFoundError(f"Missing pair for {stem}")
        link_or_copy(clear_source, clean_dir / f"{stem}.png")
        link_or_copy(hazy_source, hazy_dir / f"{stem}.png")

    print(f"Prepared {len(rows)} CARLA pairs in {output}")


if __name__ == "__main__":
    main()
