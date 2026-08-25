"""Validate, repair, summarize, and visualize Foggy Cityscapes YOLO labels."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASSES = ["car", "person", "rider", "truck", "bus", "train", "motorcycle", "bicycle"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_and_repair(path: Path) -> tuple[list[tuple[int, float, float, float, float]], int, int]:
    valid: list[tuple[int, float, float, float, float]] = []
    repaired = dropped = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if len(fields) != 5:
            dropped += 1
            continue
        try:
            class_id = int(fields[0])
            xc, yc, width, height = (float(value) for value in fields[1:])
        except ValueError:
            dropped += 1
            continue
        values = (xc, yc, width, height)
        if class_id not in range(len(CLASSES)) or not all(math.isfinite(value) for value in values):
            dropped += 1
            continue
        # YOLO validates the four normalized components.  A box touching the
        # image boundary can reconstruct to +/-5e-7 after six-decimal output;
        # that harmless quantization must not trigger a non-idempotent repair.
        if 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0:
            valid.append((class_id, xc, yc, width, height))
            continue
        x1, x2 = xc - width / 2.0, xc + width / 2.0
        y1, y2 = yc - height / 2.0, yc + height / 2.0
        nx1, nx2 = max(0.0, x1), min(1.0, x2)
        ny1, ny2 = max(0.0, y1), min(1.0, y2)
        if nx2 <= nx1 or ny2 <= ny1:
            dropped += 1
            continue
        fixed = ((nx1 + nx2) / 2.0, (ny1 + ny2) / 2.0, nx2 - nx1, ny2 - ny1)
        if any(abs(left - right) > 5e-7 for left, right in zip(values, fixed)):
            repaired += 1
        valid.append((class_id, *fixed))
    if repaired or dropped:
        path.write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in valid),
            encoding="utf-8",
        )
    return valid, repaired, dropped


def draw_labels(image_path: Path, labels: list[tuple[int, float, float, float, float]], output: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    line_width = max(2, round(min(width, height) / 350))
    for class_id, xc, yc, bw, bh in labels:
        x1, y1 = (xc - bw / 2.0) * width, (yc - bh / 2.0) * height
        x2, y2 = (xc + bw / 2.0) * width, (yc + bh / 2.0) * height
        color = COLORS[class_id]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = CLASSES[class_id]
        box = draw.textbbox((x1, y1), label, font=font, stroke_width=1)
        draw.rectangle(box, fill=color)
        draw.text((x1, y1), label, fill="white", font=font, stroke_width=1, stroke_fill=color)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=92)


def check(dataset: Path, output_dir: Path, samples: int, seed: int) -> dict:
    all_pairs: list[tuple[Path, Path, list[tuple[int, float, float, float, float]]]] = []
    split_stats = {}
    class_counts: Counter[int] = Counter()
    areas: list[float] = []
    total_repaired = total_dropped = 0
    for split in ("train", "val"):
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        image_stems = {path.stem for path in images}
        missing = [path for path in images if path.stem not in labels]
        for image in missing:
            target = label_dir / f"{image.stem}.txt"
            target.write_text("", encoding="utf-8")
            labels[image.stem] = target
        extra = sorted(set(labels) - image_stems)
        empty = boxes = repaired = dropped = 0
        for image in images:
            parsed, fixed, removed = parse_and_repair(labels[image.stem])
            empty += int(not parsed)
            boxes += len(parsed)
            repaired += fixed
            dropped += removed
            all_pairs.append((image, labels[image.stem], parsed))
            for class_id, _, _, width, height in parsed:
                class_counts[class_id] += 1
                areas.append(width * height)
        total_repaired += repaired
        total_dropped += dropped
        split_stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels_created": len(missing),
            "extra_labels": len(extra),
            "empty_labels": empty,
            "empty_label_ratio": empty / len(images) if images else 0.0,
            "boxes": boxes,
            "repaired_boxes": repaired,
            "dropped_invalid_lines": dropped,
        }
    rng = random.Random(seed)
    chosen = rng.sample(all_pairs, min(samples, len(all_pairs)))
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (image, _, parsed) in enumerate(chosen, 1):
        draw_labels(image, parsed, output_dir / f"{index:02d}_{image.stem}.jpg")
    area_array = np.asarray(areas, dtype=np.float64)
    summary = {
        "dataset": str(dataset.resolve()),
        "classes": CLASSES,
        "splits": split_stats,
        "class_distribution": {CLASSES[index]: class_counts[index] for index in range(len(CLASSES))},
        "bbox_area_normalized": {
            "count": int(area_array.size),
            "min": float(area_array.min()) if area_array.size else None,
            "mean": float(area_array.mean()) if area_array.size else None,
            "median": float(np.median(area_array)) if area_array.size else None,
            "p95": float(np.percentile(area_array, 95)) if area_array.size else None,
            "max": float(area_array.max()) if area_array.size else None,
        },
        "auto_repair": {"repaired_boxes": total_repaired, "dropped_invalid_lines": total_dropped},
        "visualizations": len(chosen),
    }
    (output_dir / "label_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/foggy_cityscapes_yolo"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/foggy_label_visualization"))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check(args.dataset, args.output_dir, args.samples, args.seed)
