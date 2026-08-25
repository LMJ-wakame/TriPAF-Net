"""Evaluate YOLO on identically named clear/hazy/dehazed image variants."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

import yaml

WORKSPACE_CONFIG = Path("artifacts/ultralytics_config").resolve()
WORKSPACE_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE_CONFIG))

from ultralytics import YOLO  # noqa: E402

COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
FOGGY_CITYSCAPES_NAMES = [
    "car",
    "person",
    "rider",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]
CITYSCAPES_TO_COCO = {0: 2, 1: 0, 3: 7, 4: 5, 5: 6, 6: 3, 7: 1}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def image_stems(directory: Path) -> set[str]:
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def label_stems(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.txt")}


def validate_variant(name: str, image_dir: Path, labels_dir: Path) -> None:
    if not image_dir.is_dir():
        raise ValueError(
            f"Variant {name!r}: image directory does not exist: {image_dir}"
        )
    if not labels_dir.is_dir():
        raise ValueError(f"Label directory does not exist: {labels_dir}")
    images = image_stems(image_dir)
    labels = label_stems(labels_dir)
    missing = images - labels
    extra = labels - images
    if missing:
        raise ValueError(
            f"Variant {name!r}: {len(missing)} images have no matching label stem"
        )
    if not images:
        raise ValueError(f"Variant {name!r}: no images found in {image_dir}")
    if extra:
        print(
            f"Warning: ignoring {len(extra)} labels with no image in variant {name!r}"
        )


def make_variant_dataset(
    root: Path,
    image_dir: Path,
    labels_dir: Path,
    name: str,
    class_names: list[str],
    label_remap: dict[int, int] | None = None,
) -> Path:
    validate_variant(name, image_dir, labels_dir)
    variant_root = root / name
    images_out = variant_root / "images" / "val"
    labels_out = variant_root / "labels" / "val"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    for image in sorted(image_dir.iterdir()):
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
            shutil.copy2(image, images_out / image.name)
            source_label = labels_dir / f"{image.stem}.txt"
            target_label = labels_out / source_label.name
            if label_remap is None:
                shutil.copy2(source_label, target_label)
            else:
                converted = []
                for line in source_label.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) != 5:
                        continue
                    mapped_class = label_remap.get(int(fields[0]))
                    if mapped_class is not None:
                        converted.append(" ".join((str(mapped_class), *fields[1:])))
                target_label.write_text(
                    "".join(f"{line}\n" for line in converted), encoding="utf-8"
                )
    # Ultralytics validates the presence of both keys even when split="val".
    # Reusing the validation directory here does not train or tune the detector.
    dataset = {
        "path": str(variant_root.resolve()),
        "train": "images/val",
        "val": "images/val",
        "names": class_names,
    }
    yaml_path = variant_root / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    return yaml_path


def parse_variants(args: argparse.Namespace) -> dict[str, Path]:
    variants = {}
    for value in args.variant:
        if "=" not in value:
            raise ValueError("--variant must use NAME=IMAGE_DIRECTORY")
        name, directory = value.split("=", 1)
        variants[name.strip()] = Path(directory.strip())
    if not variants:
        raise ValueError("Provide one or more --variant NAME=IMAGE_DIRECTORY arguments")
    return variants


def resolve_dataset_type(args: argparse.Namespace, reference_type: str) -> str:
    if args.dataset_type != "auto":
        return args.dataset_type
    if reference_type == "cityscapes_gtfine_polygons":
        return "foggy_cityscapes"
    return "carla"


def resolve_label_space(
    args: argparse.Namespace, dataset_type: str, model: YOLO
) -> str:
    if dataset_type != "foggy_cityscapes":
        return "native"
    if args.label_space != "auto":
        return args.label_space
    names = model.names.values() if isinstance(model.names, dict) else model.names
    return "coco" if len(names) == len(COCO_NAMES) else "cityscapes"


def evaluate(args: argparse.Namespace) -> Path:
    variants = parse_variants(args)
    labels_dir = Path(args.label_dir)

    label_metadata_path = labels_dir / "label_metadata.json"
    if label_metadata_path.is_file():
        label_metadata = json.loads(label_metadata_path.read_text(encoding="utf-8"))
        reference_type = label_metadata.get("annotation_type", "unknown")
    else:
        label_metadata = {}
        reference_type = "ground_truth_or_unspecified"
    dataset_type = resolve_dataset_type(args, reference_type)
    formal_reference_types = {"carla_native_ground_truth", "cityscapes_gtfine_polygons"}
    if (
        reference_type not in formal_reference_types
        and not args.allow_non_native_labels
    ):
        raise ValueError(
            "Formal detection evaluation requires CARLA native or Cityscapes gtFine ground truth. "
            "Use --allow-non-native-labels only for explicitly named pseudo-label analysis."
        )
    metadata_classes = label_metadata.get("classes")
    native_class_names = (
        metadata_classes
        if isinstance(metadata_classes, list)
        and metadata_classes
        and all(isinstance(value, str) for value in metadata_classes)
        else (
            FOGGY_CITYSCAPES_NAMES if dataset_type == "foggy_cityscapes" else COCO_NAMES
        )
    )

    model = YOLO(args.weights)
    label_space = resolve_label_space(args, dataset_type, model)
    if dataset_type == "foggy_cityscapes" and label_space == "coco":
        class_names = COCO_NAMES
        label_remap: dict[int, int] | None = CITYSCAPES_TO_COCO
        print(
            "Foggy Cityscapes labels are remapped to COCO IDs; Cityscapes 'rider' is excluded."
        )
    else:
        class_names = native_class_names
        label_remap = None

    csv_default = (
        "artifacts/foggy_detection_results.csv"
        if dataset_type == "foggy_cityscapes"
        else "artifacts/evaluation/yolo_metrics.csv"
    )
    output_csv = Path(args.csv_out or csv_default)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    run_root = output_csv.parent / "ultralytics_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    rows = []
    work_parent = output_csv.parent / "tmp"
    work_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="yolo_compare_", dir=work_parent
    ) as temporary:
        temporary_root = Path(temporary)
        for name, image_dir in variants.items():
            dataset_yaml = make_variant_dataset(
                temporary_root,
                image_dir,
                labels_dir,
                name,
                class_names,
                label_remap,
            )
            result = model.val(
                data=str(dataset_yaml),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                split="val",
                device=args.device,
                batch=args.batch,
                workers=0,
                plots=False,
                save_json=False,
                project=str(run_root),
                name=name,
                exist_ok=True,
                verbose=False,
            )
            metrics = result.box
            if dataset_type == "foggy_cityscapes":
                row = {
                    "method": name,
                    "mAP50": float(metrics.map50),
                    "mAP50-95": float(metrics.map),
                    "P": float(metrics.mp),
                    "R": float(metrics.mr),
                }
            else:
                row = {
                    "variant": name,
                    "reference_type": reference_type,
                    "images": len(image_stems(image_dir)),
                    "precision": float(metrics.mp),
                    "recall": float(metrics.mr),
                    "map50": float(metrics.map50),
                    "map50_95": float(metrics.map),
                    "preprocess_ms": float(result.speed.get("preprocess", 0.0)),
                    "inference_ms": float(result.speed.get("inference", 0.0)),
                    "postprocess_ms": float(result.speed.get("postprocess", 0.0)),
                }
            rows.append(row)
            print(json.dumps(row))

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "warning": label_metadata.get("warning"),
        "weights": str(Path(args.weights).resolve()),
        "labels": str(labels_dir.resolve()),
        "dataset_type": dataset_type,
        "label_space": label_space,
        "results": rows,
    }
    output_csv.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Saved: {output_csv}")
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="NAME=IMAGE_DIRECTORY; repeat as needed",
    )
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--weights", default="yolov8/yolov8m.pt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--csv-out", default="")
    parser.add_argument(
        "--dataset-type", choices=("auto", "carla", "foggy_cityscapes"), default="auto"
    )
    parser.add_argument(
        "--label-space", choices=("auto", "cityscapes", "coco"), default="auto"
    )
    parser.add_argument(
        "--allow-non-native-labels",
        action="store_true",
        help="Permit pseudo/unspecified labels for separately named exploratory analysis.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
