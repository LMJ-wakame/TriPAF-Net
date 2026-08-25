"""Evaluate locked TriPAF variants with their stable inference outputs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_project_runtime() -> None:
    if all(importlib.util.find_spec(name) is not None for name in ("cv2", "torch")):
        return
    configured = os.environ.get("TRIPAF_PYTHON")
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        Path.home() / root / "envs" / "dehaze_env" / "python.exe"
        for root in ("miniconda3", "anaconda3")
    )
    current = Path(sys.executable).resolve()
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() != current:
            os.execv(
                str(candidate),
                [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]],
            )
    raise RuntimeError(
        "Evaluation requires the project PyTorch environment. Activate dehaze_env "
        "or set TRIPAF_PYTHON to its Python executable."
    )


_ensure_project_runtime()

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils.classical import dcp_bcp_dehaze
from utils.inference_v2 import load_v2_checkpoint, predict_pil_v2_outputs

# The formal capture uses exactly four-digit group identifiers.  Earlier
# two-image smoke captures (group_01/group_02) are intentionally excluded so
# they cannot collide with group_0001/group_0002 in normalized outputs.
CLEAR_PATTERN = re.compile(r"^group_(\d{4})_clear$", re.IGNORECASE)
FOG_PATTERN = re.compile(r"^group_(\d{4})_fog_(\d+)$", re.IGNORECASE)


def discover_output_image_pairs(
    directory: str | Path, split: str = "test"
) -> list[dict]:
    directory = Path(directory)
    metadata_path = directory / "metadata.csv"
    if metadata_path.is_file():
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        pairs = [
            {
                "id": row["id"],
                "clear": directory / "images" / "clear" / f"{row['id']}.png",
                "hazy": directory / "images" / "hazy" / f"{row['id']}.png",
                "fog_density": float(row["fog_density"]),
            }
            for row in rows
        ]
        if not pairs or any(
            not pair["clear"].is_file() or not pair["hazy"].is_file() for pair in pairs
        ):
            raise ValueError(
                f"Incomplete publication pairs for split {split!r} in {directory}"
            )
        return pairs
    clears: dict[str, Path] = {}
    fogs: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in directory.iterdir():
        if not path.is_file():
            continue
        clear_match = CLEAR_PATTERN.match(path.stem)
        fog_match = FOG_PATTERN.match(path.stem)
        if clear_match:
            clears[clear_match.group(1)] = path
        elif fog_match:
            fogs[fog_match.group(1)].append((int(fog_match.group(2)), path))
    pairs = []
    for group_id, clear_path in sorted(clears.items(), key=lambda item: int(item[0])):
        candidates = fogs.get(group_id, [])
        if not candidates:
            continue
        # Several early groups were captured twice.  The fog frame closest in
        # modification time to the current clear frame belongs to the same run.
        fog_density, fog_path = min(
            candidates,
            key=lambda item: (
                abs(item[1].stat().st_mtime - clear_path.stat().st_mtime),
                item[1].name,
            ),
        )
        pairs.append(
            {
                "id": f"group_{int(group_id):04d}",
                "clear": clear_path,
                "hazy": fog_path,
                "fog_density": fog_density,
            }
        )
    if not pairs:
        raise ValueError(f"No group_*_clear/group_*_fog pairs found in {directory}")
    return pairs


def image_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    reference = np.clip(reference.astype(np.float32), 0, 1)
    prediction = np.clip(prediction.astype(np.float32), 0, 1)
    psnr = peak_signal_noise_ratio(reference, prediction, data_range=1.0)
    ssim = structural_similarity(reference, prediction, data_range=1.0, channel_axis=2)
    mae = np.abs(reference - prediction).mean()
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    pred_gray = cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY)
    ref_edge = cv2.Laplacian(ref_gray, cv2.CV_32F)
    pred_edge = cv2.Laplacian(pred_gray, cv2.CV_32F)
    edge_mae = np.abs(ref_edge - pred_edge).mean()
    ref_small = cv2.resize(reference, (256, 256), interpolation=cv2.INTER_AREA)
    pred_small = cv2.resize(prediction, (256, 256), interpolation=cv2.INTER_AREA)
    delta_e = float(deltaE_ciede2000(rgb2lab(ref_small), rgb2lab(pred_small)).mean())
    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "mae": float(mae),
        "edge_mae": float(edge_mae),
        "delta_e00": delta_e,
    }


def bootstrap_summary(
    rows: list[dict], seed: int = 42, samples: int = 2000
) -> list[dict]:
    rng = np.random.default_rng(seed)
    metrics = ("psnr", "ssim", "mae", "edge_mae", "delta_e00", "runtime_ms")
    methods = sorted({row["method"] for row in rows})
    summary = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        record = {"method": method, "images": len(method_rows)}
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in method_rows], dtype=np.float64
            )
            means = np.empty(samples, dtype=np.float64)
            for index in range(samples):
                means[index] = rng.choice(values, size=len(values), replace=True).mean()
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            if samples > 0:
                record[f"{metric}_ci95_low"] = float(np.percentile(means, 2.5))
                record[f"{metric}_ci95_high"] = float(np.percentile(means, 97.5))
            else:
                record[f"{metric}_ci95_low"] = float(values.mean())
                record[f"{metric}_ci95_high"] = float(values.mean())
        summary.append(record)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_dir)
    image_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    pairs = discover_output_image_pairs(args.pairs_dir, split=args.split)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    allowed = {
        "hazy",
        "dcp_bcp",
        "tripaf_v2_fixed",
        "tripaf_v2",
    }
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    v2_models = {}
    for variant, checkpoint in (
        ("tripaf_v2_fixed", args.tripaf_v2_fixed_checkpoint),
        ("tripaf_v2", args.tripaf_v2_checkpoint),
    ):
        if variant in methods:
            model, payload = load_v2_checkpoint(checkpoint, device, debug=args.debug)
            if variant == "tripaf_v2" and not payload["adaptive_fusion"]:
                raise ValueError(
                    "Official tripaf_v2 evaluation requires adaptive_fusion=True"
                )
            if variant == "tripaf_v2_fixed" and payload["adaptive_fusion"]:
                raise ValueError(
                    "TriPAF-v2-Fixed evaluation requires adaptive_fusion=False"
                )
            v2_models[variant] = model

    rows = []
    manifest = []
    for method in ["clear", *methods]:
        (image_root / method).mkdir(parents=True, exist_ok=True)

    for pair_index, pair in enumerate(pairs, start=1):
        clear_image = Image.open(pair["clear"]).convert("RGB")
        hazy_image = Image.open(pair["hazy"]).convert("RGB")
        clear = np.asarray(clear_image, dtype=np.float32) / 255.0
        hazy = np.asarray(hazy_image, dtype=np.float32) / 255.0
        normalized_name = f"{pair['id']}.png"
        shutil.copy2(pair["clear"], image_root / "clear" / normalized_name)
        manifest.append(
            {
                "id": pair["id"],
                "fog_density": pair["fog_density"],
                "clear_source": str(pair["clear"]),
                "hazy_source": str(pair["hazy"]),
            }
        )

        v2_output_cache: dict[str, tuple[dict[str, torch.Tensor], float]] = {}
        classical_prediction = None
        classical_runtime_ms = 0.0
        needs_detail_guidance = (
            any(
                method.startswith("v2_") or method.startswith("tripaf_v2")
                for method in methods
            )
            and not args.no_detail_guidance
        )
        if "dcp_bcp" in methods or needs_detail_guidance:
            classical_started = time.perf_counter()
            classical_prediction, _, _ = dcp_bcp_dehaze(np.asarray(hazy_image))
            classical_runtime_ms = (time.perf_counter() - classical_started) * 1000.0

        for method in methods:
            if method == "hazy":
                started = time.perf_counter()
                prediction = hazy
            elif method == "dcp_bcp":
                started = time.perf_counter()
                prediction = classical_prediction
            else:
                if method not in v2_output_cache:
                    started = time.perf_counter()
                    cached_outputs = predict_pil_v2_outputs(
                        v2_models[method],
                        hazy_image,
                        device,
                        amp=not args.no_amp,
                        debug=args.debug,
                        classical_guidance=classical_prediction,
                        use_detail_guidance=not args.no_detail_guidance,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    cached_runtime_ms = (time.perf_counter() - started) * 1000.0
                    if not args.no_detail_guidance:
                        cached_runtime_ms += classical_runtime_ms
                    v2_output_cache[method] = (cached_outputs, cached_runtime_ms)
                cached_outputs, runtime_ms = v2_output_cache[method]
                tensor = cached_outputs["stable_image"]
                prediction = tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            if method == "dcp_bcp":
                runtime_ms = classical_runtime_ms
            elif method == "hazy":
                if device.type == "cuda":
                    torch.cuda.synchronize()
                runtime_ms = (time.perf_counter() - started) * 1000.0
            metrics = image_metrics(clear, prediction)
            row = {
                "id": pair["id"],
                "fog_density": pair["fog_density"],
                "method": method,
                **metrics,
                "runtime_ms": runtime_ms,
            }
            rows.append(row)
            Image.fromarray(
                np.round(prediction * 255).astype(np.uint8), mode="RGB"
            ).save(
                image_root / method / normalized_name,
                compress_level=1,
            )
        print(f"Evaluated {pair_index:04d}/{len(pairs):04d}: {pair['id']}")

    summary = bootstrap_summary(rows, seed=args.seed, samples=args.bootstrap_samples)
    write_csv(output_root / "per_image_metrics.csv", rows)
    write_csv(output_root / "summary_metrics.csv", summary)
    write_csv(output_root / "pair_manifest.csv", manifest)
    (output_root / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return output_root / "summary_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-dir", default="data/carla_tripaf_1024")
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--methods", default="hazy,dcp_bcp,tripaf_v2_fixed,tripaf_v2")
    parser.add_argument(
        "--tripaf-v2-fixed-checkpoint",
        default="checkpoints/tripaf_v2_fixed/seed_42/best.pt",
    )
    parser.add_argument(
        "--tripaf-v2-checkpoint",
        default="checkpoints/tripaf_v2/seed_42/best.pt",
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-detail-guidance", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
