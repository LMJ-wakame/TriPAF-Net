"""Evaluate aligned Foggy Cityscapes variants with one frozen YOLOv8m detector."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

WORKSPACE_CONFIG = Path("artifacts/ultralytics_config").resolve()
WORKSPACE_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE_CONFIG))

from ultralytics import YOLO  # noqa: E402

CITYSCAPES_NAMES = ["car", "person", "rider", "truck", "bus", "train", "motorcycle", "bicycle"]
CITYSCAPES_TO_COCO = {0: 2, 1: 0, 3: 7, 4: 5, 5: 6, 6: 3, 7: 1}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_paths(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def hardlink_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def count_native_instances(labels: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for path in labels.glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 5:
                counts[int(fields[0])] += 1
    return counts


def make_dataset(root: Path, images: Path, labels: Path, coco_names: dict[int, str]) -> tuple[Path, int]:
    image_list = image_paths(images)
    if not image_list:
        raise ValueError(f"No images found: {images}")
    image_out = root / "images" / "val"
    label_out = root / "labels" / "val"
    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)
    missing = []
    for image in image_list:
        source_label = labels / f"{image.stem}.txt"
        if not source_label.is_file():
            missing.append(image.name)
            continue
        hardlink_or_copy(image, image_out / image.name)
        converted = []
        for line in source_label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            mapped = CITYSCAPES_TO_COCO.get(int(fields[0]))
            if mapped is not None:
                converted.append(" ".join((str(mapped), *fields[1:])))
        (label_out / source_label.name).write_text(
            "".join(f"{line}\n" for line in converted), encoding="utf-8"
        )
    if missing:
        raise ValueError(f"{len(missing)} images have no matching labels in {labels}")
    dataset = {
        "path": str(root.resolve()),
        "train": "images/val",
        "val": "images/val",
        "names": coco_names,
    }
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    return yaml_path, len(image_list)


def class_rows(method: str, metrics, native_counts: Counter[int], model_names: dict[int, str]) -> list[dict]:
    present_ids = [int(value) for value in np.asarray(metrics.ap_class_index).tolist()]
    by_coco = {class_id: index for index, class_id in enumerate(present_ids)}
    precision = np.asarray(metrics.p, dtype=float)
    recall = np.asarray(metrics.r, dtype=float)
    ap50 = np.asarray(metrics.ap50, dtype=float)
    ap = np.asarray(metrics.ap, dtype=float)
    rows = []
    for native_id, name in enumerate(CITYSCAPES_NAMES):
        coco_id = CITYSCAPES_TO_COCO.get(native_id)
        index = by_coco.get(coco_id) if coco_id is not None else None
        rows.append(
            {
                "method": method,
                "class": name,
                "instances": native_counts[native_id],
                "evaluated": bool(index is not None),
                "precision": float(precision[index]) if index is not None else "",
                "recall": float(recall[index]) if index is not None else "",
                "AP50": float(ap50[index]) if index is not None else "",
                "AP50_95": float(ap[index].mean()) if index is not None else "",
                "detector_class": model_names[coco_id] if coco_id is not None else "not available in COCO",
            }
        )
    return rows


def evaluate(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels = args.labels.resolve()
    variants = {
        "Foggy": args.foggy.resolve(),
        "DCP_BCP": args.dcp_bcp.resolve(),
        "TriPAF": args.tripaf.resolve(),
    }
    detector = YOLO(str(args.weights.resolve()))
    model_names = detector.names if isinstance(detector.names, dict) else dict(enumerate(detector.names))
    expected = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck"}
    if any(model_names.get(index) != name for index, name in expected.items()):
        raise ValueError("Selected detector does not use the expected COCO class space")
    native_counts = count_native_instances(labels)
    summary_rows = []
    per_class_rows = []
    speed_rows = []
    work_parent = output / "tmp"
    work_parent.mkdir(exist_ok=True)
    for method, image_dir in variants.items():
        with tempfile.TemporaryDirectory(prefix=f"{method.lower()}_", dir=work_parent) as temporary:
            dataset_yaml, count = make_dataset(Path(temporary), image_dir, labels, model_names)
            result = detector.val(
                data=str(dataset_yaml),
                split="val",
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                batch=args.batch,
                workers=0,
                plots=False,
                save_json=False,
                half=args.half,
                seed=42,
                deterministic=True,
                project=str(output / "ultralytics_runs"),
                name=method,
                exist_ok=True,
                verbose=False,
            )
        box = result.box
        row = {
            "method": method,
            "images": count,
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
        }
        summary_rows.append(row)
        per_class_rows.extend(class_rows(method, box, native_counts, model_names))
        speed_rows.append({"method": method, **{f"{key}_ms": float(value) for key, value in result.speed.items()}})
        print(json.dumps(row), flush=True)
    csv_path = output / "foggy_detection_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    class_csv = output / "foggy_detection_classwise_ap.csv"
    with class_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)
    speed_csv = output / "foggy_detection_speed.csv"
    with speed_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(speed_rows[0]))
        writer.writeheader()
        writer.writerows(speed_rows)
    metadata = {
        "detector": str(args.weights.resolve()),
        "detector_pretraining": "COCO pretrained YOLOv8m; no detector training was performed",
        "dataset": "Foggy Cityscapes beta=0.02 validation split",
        "images": summary_rows[0]["images"],
        "label_source": "Cityscapes gtFine polygons converted to YOLO boxes",
        "class_policy": {
            "native_classes": CITYSCAPES_NAMES,
            "evaluated_classes": [name for index, name in enumerate(CITYSCAPES_NAMES) if index in CITYSCAPES_TO_COCO],
            "excluded_from_detector_metrics": ["rider"],
            "reason": "COCO YOLOv8m has no rider category; mapping rider to person would be scientifically invalid",
        },
        "domain_gap": "The detector is COCO-pretrained, not adapted to Foggy Cityscapes; absolute AP is therefore a cross-domain result.",
        "settings": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "batch": args.batch, "half": args.half},
        "results": summary_rows,
    }
    (output / "foggy_detection_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {csv_path}")
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("results/foggy_detection/inputs")
    parser.add_argument("--foggy", type=Path, default=base / "Foggy")
    parser.add_argument("--dcp-bcp", type=Path, default=base / "DCP_BCP")
    parser.add_argument("--tripaf", type=Path, default=base / "TriPAF")
    parser.add_argument("--labels", type=Path, default=Path("data/foggy_cityscapes_yolo/labels/val"))
    parser.add_argument("--weights", type=Path, default=Path("yolov8/yolov8m.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/foggy_detection"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
