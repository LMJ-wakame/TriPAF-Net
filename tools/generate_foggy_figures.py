"""Generate IEEE-style Foggy Cityscapes detection figures in PDF and PNG."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

WORKSPACE_CONFIG = Path("artifacts/ultralytics_config").resolve()
WORKSPACE_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE_CONFIG))

from ultralytics import YOLO  # noqa: E402

METHODS = ["Foggy", "DCP_BCP", "TriPAF"]
LABELS = {"Foggy": "Foggy", "DCP_BCP": "DCP/BCP", "TriPAF": "TriPAF-Net v2"}
COLORS = {"Foggy": "#7A7A7A", "DCP_BCP": "#4C78A8", "TriPAF": "#D55E00"}
CITYSCAPES_TO_COCO = {0: 2, 1: 0, 3: 7, 4: 5, 5: 6, 6: 3, 7: 1}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.png", dpi=400)
    plt.close(fig)


def annotate_bars(axis, bars, digits: int = 3) -> None:
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{bar.get_height():.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def metric_figures(summary: list[dict], classwise: list[dict], output: Path) -> None:
    by_method = {row["method"]: row for row in summary}
    x = np.arange(len(METHODS))
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    width = 0.36
    bars1 = ax.bar(x - width / 2, [float(by_method[m]["mAP50"]) for m in METHODS], width, label="mAP@0.50", color=[COLORS[m] for m in METHODS], edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, [float(by_method[m]["mAP50_95"]) for m in METHODS], width, label="mAP@0.50:0.95", color=[COLORS[m] for m in METHODS], alpha=0.48, edgecolor="black", linewidth=0.5)
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    ax.set_xticks(x, [LABELS[m] for m in METHODS])
    ax.set_ylabel("Average precision")
    ax.set_ylim(0, max(bar.get_height() for bar in [*bars1, *bars2]) * 1.25 + 0.02)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("Foggy Cityscapes Detection Performance")
    save(fig, output, "detection_map_comparison")

    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    bars1 = ax.bar(x - width / 2, [float(by_method[m]["precision"]) for m in METHODS], width, label="Precision", color=[COLORS[m] for m in METHODS], edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, [float(by_method[m]["recall"]) for m in METHODS], width, label="Recall", color=[COLORS[m] for m in METHODS], alpha=0.48, edgecolor="black", linewidth=0.5)
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    ax.set_xticks(x, [LABELS[m] for m in METHODS])
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(bar.get_height() for bar in [*bars1, *bars2]) * 1.25 + 0.02)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("Precision and Recall Comparison")
    save(fig, output, "precision_recall_comparison")

    valid = [row for row in classwise if row["evaluated"].lower() == "true"]
    classes = []
    for row in valid:
        if row["class"] not in classes:
            classes.append(row["class"])
    fig, ax = plt.subplots(figsize=(7.16, 2.75))
    width = 0.25
    x = np.arange(len(classes))
    for index, method in enumerate(METHODS):
        lookup = {row["class"]: float(row["AP50"]) for row in valid if row["method"] == method}
        ax.bar(x + (index - 1) * width, [lookup[name] for name in classes], width, label=LABELS[method], color=COLORS[method], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x, [name.capitalize() for name in classes])
    ax.set_ylabel("AP@0.50")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_title("Class-wise AP on Foggy Cityscapes (COCO-compatible Classes)")
    ax.text(0.0, -0.28, "Note: Cityscapes 'rider' is not evaluated because COCO YOLOv8m has no rider class.", transform=ax.transAxes, fontsize=7)
    save(fig, output, "classwise_ap")


def load_gt(label_path: Path) -> list[tuple[int, np.ndarray]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        native = int(fields[0])
        coco = CITYSCAPES_TO_COCO.get(native)
        if coco is None:
            continue
        xc, yc, width, height = map(float, fields[1:])
        boxes.append((coco, np.array([xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2])))
    return boxes


def matched_tp(result, gt: list[tuple[int, np.ndarray]], iou_threshold: float = 0.5) -> int:
    if result.boxes is None:
        return 0
    predictions = [
        (int(class_id), box.astype(float), float(conf))
        for class_id, box, conf in zip(result.boxes.cls.cpu().numpy(), result.boxes.xyxyn.cpu().numpy(), result.boxes.conf.cpu().numpy())
    ]
    predictions.sort(key=lambda item: item[2], reverse=True)
    used: set[int] = set()
    matches = 0
    for class_id, box, _ in predictions:
        best_index = None
        best_iou = 0.0
        for index, (gt_class, gt_box) in enumerate(gt):
            if index in used or gt_class != class_id:
                continue
            inter1 = np.maximum(box[:2], gt_box[:2])
            inter2 = np.minimum(box[2:], gt_box[2:])
            intersection = np.maximum(0.0, inter2 - inter1).prod()
            union = np.maximum(0.0, box[2:] - box[:2]).prod() + np.maximum(0.0, gt_box[2:] - gt_box[:2]).prod() - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_index is not None and best_iou >= iou_threshold:
            used.add(best_index)
            matches += 1
    return matches


def qualitative(args: argparse.Namespace, output: Path) -> None:
    roots = {method: args.inputs / method for method in METHODS}
    names = sorted(path.name for path in roots["Foggy"].glob("*.png"))
    candidate_indices = np.linspace(0, len(names) - 1, min(args.candidates, len(names)), dtype=int)
    candidates = [names[index] for index in candidate_indices]
    model = YOLO(str(args.weights))
    predictions = {}
    for method in METHODS:
        predictions[method] = model.predict(
            [str(roots[method] / name) for name in candidates],
            imgsz=args.imgsz,
            conf=0.25,
            iou=0.7,
            device=args.device,
            batch=2,
            half=True,
            verbose=False,
        )
    ranked = []
    for index, name in enumerate(candidates):
        gt = load_gt(args.labels / f"{Path(name).stem}.txt")
        scores = {method: matched_tp(predictions[method][index], gt) for method in METHODS}
        ranked.append((scores["TriPAF"] - scores["Foggy"], scores["TriPAF"] - scores["DCP_BCP"], len(gt), index, name, scores))
    ranked.sort(reverse=True)
    chosen = ranked[: min(4, len(ranked))]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(7.16, 1.52 * len(chosen)), squeeze=False)
    for row, (_, _, gt_count, index, name, scores) in enumerate(chosen):
        for column, method in enumerate(METHODS):
            rendered = predictions[method][index].plot(labels=True, conf=True, line_width=2)
            axes[row, column].imshow(rendered[..., ::-1])
            axes[row, column].axis("off")
            axes[row, column].set_title(f"{LABELS[method]} | matched TP: {scores[method]}/{gt_count}", fontsize=8)
        axes[row, 0].set_ylabel(Path(name).stem.split("_leftImg8bit")[0], fontsize=7)
    fig.suptitle("Qualitative Detection Examples (Deterministically Selected Success Cases)", fontsize=10, y=0.995)
    fig.text(0.5, 0.002, "Boxes: frozen COCO-pretrained YOLOv8m predictions; matched TP uses class-aware IoU >= 0.50.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.018, 1, 0.98), pad=0.4)
    save(fig, output, "qualitative_examples")


def pipeline(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.16, 1.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.33, 0.16, 0.34, "Foggy image\n(beta = 0.02)"),
        (0.23, 0.33, 0.18, 0.34, "DCP/BCP priors\ndark, bright, sky"),
        (0.46, 0.33, 0.18, 0.34, "TriPAF-Net v2\nadaptive fusion"),
        (0.69, 0.33, 0.13, 0.34, "Frozen\nYOLOv8m"),
        (0.87, 0.33, 0.11, 0.34, "Detection\nprediction"),
    ]
    for index, (x, y, width, height, label) in enumerate(boxes):
        color = "#F3F3F3" if index != 2 else "#FCE8DE"
        edge = "#333333" if index != 2 else COLORS["TriPAF"]
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012", facecolor=color, edgecolor=edge, linewidth=1.1))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=8)
        if index < len(boxes) - 1:
            next_x = boxes[index + 1][0]
            ax.add_patch(FancyArrowPatch((x + width + 0.008, 0.50), (next_x - 0.008, 0.50), arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="#333333"))
    ax.text(0.5, 0.87, "Real-fog Detection Evaluation Pipeline", ha="center", fontsize=10, fontweight="bold")
    ax.text(0.5, 0.13, "All three image variants are evaluated by the same detector; no detector or dehazing retraining is performed.", ha="center", fontsize=7.5)
    save(fig, output, "pipeline_diagram")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/foggy_detection"))
    parser.add_argument("--inputs", type=Path, default=Path("results/foggy_detection/inputs"))
    parser.add_argument("--labels", type=Path, default=Path("data/foggy_cityscapes_yolo/labels/val"))
    parser.add_argument("--weights", type=Path, default=Path("yolov8/yolov8m.pt"))
    parser.add_argument("--output", type=Path, default=Path("figures/foggy_cityscapes"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--candidates", type=int, default=36)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    style()
    metric_figures(read_csv(args.results / "foggy_detection_results.csv"), read_csv(args.results / "foggy_detection_classwise_ap.csv"), args.output)
    qualitative(args, args.output)
    pipeline(args.output)
