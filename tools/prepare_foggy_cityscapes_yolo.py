"""Convert Foggy Cityscapes gtFine polygons into a compact YOLO dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CLASSES = ["car", "person", "rider", "truck", "bus", "train", "motorcycle", "bicycle"]
CLASS_IDS = {name: index for index, name in enumerate(CLASSES)}
SPLITS = ("train", "val")
FOGGY_MARKER = "_leftImg8bit_foggy_beta_"
POLYGON_SUFFIX = "_gtFine_polygons"


@dataclass(frozen=True)
class FoggyImage:
    path: Path
    split: str
    key: str


def parse_beta(value: str) -> float:
    beta = float(value)
    if beta not in {0.005, 0.01, 0.02}:
        raise argparse.ArgumentTypeError("--beta must be one of: 0.005, 0.01, 0.02")
    return beta


def split_for_path(path: Path) -> str | None:
    for parent in (path.parent, *path.parents):
        if parent.name in SPLITS:
            return parent.name
    return None


def foggy_key(path: Path) -> str | None:
    if FOGGY_MARKER not in path.stem:
        return None
    return path.stem.split(FOGGY_MARKER, maxsplit=1)[0]


def matches_beta(path: Path, beta: float) -> bool:
    if FOGGY_MARKER not in path.stem:
        return False
    try:
        image_beta = float(path.stem.rsplit(FOGGY_MARKER, maxsplit=1)[1])
    except ValueError:
        return False
    return abs(image_beta - beta) < 1e-9


def discover_foggy_images(raw_dir: Path, beta: float) -> list[FoggyImage]:
    images: list[FoggyImage] = []
    for path in raw_dir.rglob("*_leftImg8bit_foggy_beta_*.png"):
        split = split_for_path(path)
        key = foggy_key(path)
        if split is not None and key is not None and matches_beta(path, beta):
            images.append(FoggyImage(path=path, split=split, key=key))
    return sorted(images, key=lambda item: (item.split, item.path.as_posix()))


def discover_polygons(raw_dir: Path) -> dict[str, Path]:
    polygons: dict[str, Path] = {}
    for path in raw_dir.rglob("*_gtFine_polygons.json"):
        if not path.stem.endswith(POLYGON_SUFFIX):
            continue
        key = path.stem.removesuffix(POLYGON_SUFFIX)
        polygons.setdefault(key, path)
    return polygons


def polygon_boxes(annotation_path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    boxes: list[tuple[int, float, float, float, float]] = []
    for obj in payload.get("objects", []):
        if obj.get("deleted"):
            continue
        class_id = CLASS_IDS.get(obj.get("label"))
        polygon = obj.get("polygon")
        if class_id is None or not isinstance(polygon, list) or len(polygon) < 3:
            continue
        points = [point for point in polygon if isinstance(point, list) and len(point) >= 2]
        if len(points) < 3:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
        y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
        box_width, box_height = x2 - x1, y2 - y1
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        boxes.append(
            (
                class_id,
                ((x1 + x2) * 0.5) / width,
                ((y1 + y2) * 0.5) / height,
                box_width / width,
                box_height / height,
            )
        )
    return boxes


def write_dataset_yaml(output_dir: Path) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    payload = "\n".join(
        (
            f"path: {output_dir.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            names,
            "",
        )
    )
    (output_dir / "data.yaml").write_text(
        payload,
        encoding="utf-8",
    )
    # Keep the historical filename as a compatibility alias for older commands.
    (output_dir / "foggy.yaml").write_text(
        payload,
        encoding="utf-8",
    )


def materialize_image(source: Path, target: Path, mode: str) -> str:
    """Create a space-efficient dataset image while remaining portable."""

    if target.is_file() and target.stat().st_size == source.stat().st_size:
        return "existing"
    if target.exists():
        target.unlink()
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, target)
    return "copy"


def prepare(raw_dir: Path, output_dir: Path, beta: float, link_mode: str = "auto") -> int:
    if not raw_dir.is_dir():
        print(f"Please put Foggy Cityscapes dataset into {raw_dir}")
        return 0

    from PIL import Image
    from tqdm import tqdm

    images = discover_foggy_images(raw_dir, beta)
    polygons = discover_polygons(raw_dir)
    print(f"Found images: {len(images)} (beta={beta:g})")
    print(f"Found polygon annotations: {len(polygons)}")
    if not images:
        print("No matching foggy images were found. Expected *_leftImg8bit_foggy_beta_<beta>.png")
        return 1

    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    copied = Counter()
    converted = Counter()
    missing_annotations = Counter()
    materialization = Counter()
    for item in tqdm(images, desc="Preparing Foggy Cityscapes", unit="image"):
        image_out = output_dir / "images" / item.split / item.path.name
        label_out = output_dir / "labels" / item.split / f"{item.path.stem}.txt"
        materialization[materialize_image(item.path, image_out, link_mode)] += 1
        copied[item.split] += 1

        annotation_path = polygons.get(item.key)
        if annotation_path is None:
            missing_annotations[item.split] += 1
            continue
        with Image.open(item.path) as image:
            width, height = image.size
        boxes = polygon_boxes(annotation_path, width, height)
        label_out.write_text(
            "".join(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"
                for class_id, x_center, y_center, box_width, box_height in boxes
            ),
            encoding="utf-8",
        )
        converted[item.split] += 1

    write_dataset_yaml(output_dir)
    label_metadata = {
        "annotation_type": "cityscapes_gtfine_polygons",
        "classes": CLASSES,
        "source": str(raw_dir.resolve()),
        "beta": beta,
    }
    (output_dir / "labels" / "label_metadata.json").write_text(
        json.dumps(label_metadata, indent=2), encoding="utf-8"
    )

    print(f"Converted labels: {sum(converted.values())}")
    print(f"Image materialization: {dict(materialization)}")
    for split in SPLITS:
        print(
            f"{split.title()}: {copied[split]} images, {converted[split]} labels, "
            f"{missing_annotations[split]} without gtFine"
        )
    print(f"Wrote: {output_dir / 'data.yaml'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/foggy_cityscapes"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/foggy_cityscapes_yolo"))
    parser.add_argument("--beta", type=parse_beta, default=0.02)
    parser.add_argument(
        "--link-mode", choices=("auto", "hardlink", "copy"), default="auto",
        help="Use hard links when possible to avoid duplicating the 6+ GB source images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(prepare(args.raw_dir, args.output_dir, args.beta, args.link_mode))


if __name__ == "__main__":
    main()
