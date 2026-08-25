"""Generate Foggy, DCP/BCP, and TriPAF inputs without training any model."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

from utils.classical import dcp_bcp_dehaze
from utils.inference_v1 import tensor_to_pil
from utils.inference_v2 import load_v2_checkpoint, predict_pil_v2_outputs

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def materialize(source: Path, target: Path) -> None:
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        return
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def run(args: argparse.Namespace) -> Path:
    source = args.source.resolve()
    output = args.output_dir.resolve()
    directories = {name: output / "inputs" / name for name in ("Foggy", "DCP_BCP", "TriPAF")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise ValueError(f"No images found in {source}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, payload = load_v2_checkpoint(args.checkpoint, device, debug=True)
    if not payload.get("adaptive_fusion", False):
        raise ValueError("The formal TriPAF experiment requires adaptive_fusion=True")
    rows = []
    for index, path in enumerate(images, 1):
        foggy_target = directories["Foggy"] / path.name
        dcp_target = directories["DCP_BCP"] / path.name
        tripaf_target = directories["TriPAF"] / path.name
        materialize(path, foggy_target)
        if not args.force and dcp_target.is_file() and tripaf_target.is_file():
            rows.append({"image": path.name, "dcp_bcp_ms": 0.0, "tripaf_ms": 0.0, "status": "existing"})
            continue
        image = Image.open(path).convert("RGB")
        start = time.perf_counter()
        classical, _, _ = dcp_bcp_dehaze(np.asarray(image))
        dcp_ms = (time.perf_counter() - start) * 1000.0
        Image.fromarray((np.clip(classical, 0.0, 1.0) * 255.0).round().astype(np.uint8)).save(dcp_target)
        start = time.perf_counter()
        outputs = predict_pil_v2_outputs(
            model,
            image,
            device,
            amp=not args.no_amp,
            classical_guidance=classical,
            use_detail_guidance=not args.no_detail_guidance,
        )
        tripaf_ms = (time.perf_counter() - start) * 1000.0
        tensor_to_pil(outputs["stable_image"]).save(tripaf_target)
        rows.append({"image": path.name, "dcp_bcp_ms": dcp_ms, "tripaf_ms": tripaf_ms, "status": "generated"})
        print(f"[{index}/{len(images)}] {path.name}: DCP/BCP {dcp_ms:.1f} ms, TriPAF {tripaf_ms:.1f} ms", flush=True)
    manifest = output / "restoration_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} aligned image triplets and {manifest}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/foggy_cityscapes_yolo/images/val"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tripaf_v2/seed_42/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/foggy_detection"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-detail-guidance", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
