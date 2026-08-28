"""Single-image inference for the official TriPAF-Net v2 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from utils.image_io import tensor_to_pil
from utils.inference_v2 import load_v2_checkpoint, predict_pil_v2

OUTPUT_KEYS = (
    "stable_image",
    "model_stable_image",
    "image",
    "restoration",
    "direct",
    "physical",
    "prior_reconstruction",
)


def run(
    input_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path,
    output_key: str = "stable_image",
    device: str = "",
    amp: bool = True,
    detail_guidance: bool = True,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    checkpoint_path = Path(checkpoint_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if output_key not in OUTPUT_KEYS:
        raise ValueError(f"Unsupported output key: {output_key}")

    selected_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, _ = load_v2_checkpoint(checkpoint_path, selected_device)
    image = Image.open(input_path).convert("RGB")
    prediction = predict_pil_v2(
        model,
        image,
        selected_device,
        output_key=output_key,
        amp=amp,
        use_detail_guidance=detail_guidance,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(prediction).save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_image")
    parser.add_argument("output_image")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/tripaf_v2/seed_42/best.pt",
    )
    parser.add_argument("--output-key", choices=OUTPUT_KEYS, default="stable_image")
    parser.add_argument("--device", default="")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--detail-guidance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    saved_path = run(
        arguments.input_image,
        arguments.output_image,
        arguments.checkpoint,
        output_key=arguments.output_key,
        device=arguments.device,
        amp=arguments.amp,
        detail_guidance=arguments.detail_guidance,
    )
    print(f"Saved: {saved_path}")
