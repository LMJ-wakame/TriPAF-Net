"""Validate publication pairs, metadata, split isolation, and YOLO boxes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.protocol import validate_yolo_row

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.root / "metadata.csv").open(encoding="utf-8")))
    seen = set()
    split_ids = {"train": set(), "val": set(), "test": set()}
    for row in rows:
        identifier = row["id"]
        if identifier in seen:
            raise ValueError(f"duplicate group {identifier}")
        seen.add(identifier)
        split = row.get("split", "")
        if split not in split_ids:
            raise ValueError(f"invalid split for {identifier}: {split!r}")
        split_ids[split].add(identifier)
        for variant in ("clear", "hazy"):
            if not (args.root / "images" / variant / f"{identifier}.png").is_file():
                raise ValueError(f"missing {variant} member for {identifier}")
        for line in (
            (args.root / "labels" / f"{identifier}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        ):
            class_id, x, y, width, height = line.split()
            validate_yolo_row(
                int(class_id), float(x), float(y), float(width), float(height)
            )
    if any(not values for values in split_ids.values()):
        raise ValueError("train, validation, and test splits must all be nonempty")
    if (
        split_ids["train"] & split_ids["val"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["val"] & split_ids["test"]
    ):
        raise ValueError("publication splits overlap")
    label_metadata = json.loads(
        (args.root / "labels" / "label_metadata.json").read_text(encoding="utf-8")
    )
    if label_metadata.get("annotation_type") != "carla_native_ground_truth":
        raise ValueError("formal labels are not marked as CARLA native ground truth")
    print(f"validated {len(rows)} complete paired groups")
