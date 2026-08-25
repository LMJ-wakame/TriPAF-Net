"""Strict checkpoint loading and inference for TriPAF-Net v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.tripafnet_v2 import TriPAFNetV2
from utils.classical import dcp_bcp_dehaze
from utils.image_io import (
    pad_inputs,
    prepare_priors,
    prepare_priors_on_device,
    tensor_to_pil,
)


def _box_blur(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Box blur a CHW tensor with reflected borders."""

    height, width = tensor.shape[-2:]
    kernel_size = min(kernel_size, 2 * min(height, width) - 1)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size <= 1:
        return tensor
    radius = kernel_size // 2
    batched = tensor.unsqueeze(0)
    padded = F.pad(batched, (radius, radius, radius, radius), mode="reflect")
    return F.avg_pool2d(padded, kernel_size, stride=1)[0]


def _inject_classical_detail(
    model_image: torch.Tensor,
    direct: torch.Tensor,
    classical_image: np.ndarray,
    sky_protection: torch.Tensor,
) -> torch.Tensor:
    """Fuse classical structure with TriPAF color and edge-aware contrast."""

    classical = torch.from_numpy(
        np.ascontiguousarray(classical_image.transpose(2, 0, 1))
    ).to(device=model_image.device, dtype=model_image.dtype)

    weights = model_image.new_tensor((0.299, 0.587, 0.114)).view(3, 1, 1)

    model_luma = (model_image * weights).sum(0, keepdim=True)
    classical_luma = (classical * weights).sum(0, keepdim=True)
    direct_luma = (direct * weights).sum(0, keepdim=True)

    direct_low = _box_blur(direct_luma.clamp_min(0.0).sqrt(), 63)
    classical_low = _box_blur(classical_luma, 63)

    refined_luma = 0.85 * direct_low + 0.15 * classical_low
    refined_luma = refined_luma + classical_luma - classical_low
    fine_detail = refined_luma - _box_blur(refined_luma, 5)
    detail_gate = fine_detail.abs() / (fine_detail.abs() + 0.015)
    sharpen_strength = 0.06 * (1.0 - sky_protection.clamp(0.0, 1.0)) * detail_gate
    refined_luma = refined_luma + sharpen_strength * fine_detail

    # Apply the visible contrast and brightness lift after structure injection;
    # otherwise the final luminance replacement would hide the model refinement.
    local_luma = _box_blur(refined_luma, 31)
    local_detail = refined_luma - local_luma
    local_gate = local_detail.abs() / (local_detail.abs() + 0.020)
    non_sky = 1.0 - sky_protection.clamp(0.0, 1.0)
    refined_luma = refined_luma + non_sky * (0.10 + 0.08 * local_gate) * local_detail
    refined_luma = refined_luma + 0.035 * non_sky * (1.0 - refined_luma)
    refined_luma = refined_luma.clamp(0.0, 1.0)

    # Keep TriPAF chroma and only inject luminance structure.
    return (model_image + (refined_luma - model_luma)).clamp(0.0, 1.0)


def load_v2_checkpoint(
    path: str | Path,
    device: torch.device | str,
    debug: bool = False,
) -> tuple[TriPAFNetV2, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"TriPAF-Net v2 checkpoint not found: {checkpoint_path}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(
            f"Expected a TriPAF v2 checkpoint payload, got: {checkpoint_path}"
        )
    adaptive_fusion = payload.get("adaptive_fusion", False)
    model = TriPAFNetV2(
        base_channels=int(payload.get("base_channels", 24)),
        adaptive_fusion=bool(adaptive_fusion),
        residual_scale=float(payload.get("residual_scale", 0.5)),
        t_min=float(payload.get("t_min", payload.get("prior_min_transmission", 0.08))),
    )
    if int(payload.get("format_version", 0)) != 3:
        raise ValueError(f"Expected checkpoint format 3: {checkpoint_path}")
    model.load_state_dict(payload["model"], strict=True)
    checkpoint_state = payload["model"]
    payload = dict(payload)
    payload["adaptive_fusion"] = bool(adaptive_fusion)
    payload["checkpoint_compatibility"] = {
        "loaded_keys": len(checkpoint_state),
        "missing_keys": [],
        "unexpected_keys": [],
    }
    model.to(device).eval()
    if debug:
        loaded_elements = sum(value.numel() for value in checkpoint_state.values())
        print(f"checkpoint: {checkpoint_path.resolve()}")
        print(f"model: {type(model).__name__} adaptive_fusion={model.adaptive_fusion}")
        print(f"loaded: {len(checkpoint_state)} tensors, {loaded_elements} elements")
        print("missing keys: []")
        print("unexpected keys: []")
    return model, payload


@torch.inference_mode()
def predict_pil_v2_outputs(
    model: TriPAFNetV2,
    image: Image.Image,
    device: torch.device | str,
    amp: bool = True,
    debug: bool = False,
    classical_guidance: np.ndarray | None = None,
    use_detail_guidance: bool = True,
) -> dict[str, torch.Tensor]:
    """Predict all v2 branches with one full-resolution model pass."""

    device = torch.device(device)
    inputs = (
        prepare_priors_on_device(image, device)
        if device.type == "cuda"
        else {key: value.to(device) for key, value in prepare_priors(image).items()}
    )
    height, width = inputs["rgb"].shape[-2:]
    padded, pad_h, pad_w = pad_inputs(inputs)
    use_amp = bool(amp and device.type == "cuda")
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        outputs = model(**padded, return_aux=True)
    predictions = {}
    for key, value in outputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if (
            value.ndim == 4
            and value.shape[-2:] == padded["rgb"].shape[-2:]
            and (pad_h or pad_w)
        ):
            value = value[..., :height, :width]
        predictions[key] = (
            value[0].float()
            if value.ndim > 0 and value.shape[0] == 1
            else value.float()
        )
    predictions["model_stable_image"] = predictions["stable_image"]
    if use_detail_guidance:
        if classical_guidance is None:
            classical_guidance, _, _ = dcp_bcp_dehaze(np.asarray(image.convert("RGB")))
        predictions["stable_image"] = _inject_classical_detail(
            predictions["model_stable_image"],
            predictions["direct"],
            classical_guidance,
            predictions["color_sky_protection"],
        )
    if debug:
        for key in ("physical", "restoration", "direct"):
            value = predictions[key]
            print(
                f"{key}: min={value.min().item():.6f} "
                f"max={value.max().item():.6f} mean={value.mean().item():.6f}"
            )
        transmission = predictions["transmission"]
        atmospheric = predictions["atmospheric_light"]
        print(
            f"transmission: min={transmission.min().item():.6f} "
            f"max={transmission.max().item():.6f} mean={transmission.mean().item():.6f}"
        )
        print(
            f"atmospheric_light: min={atmospheric.min().item():.6f} "
            f"max={atmospheric.max().item():.6f} mean={atmospheric.mean().item():.6f}"
        )
        print(
            f"final image: min={predictions['stable_image'].min().item():.6f} "
            f"max={predictions['stable_image'].max().item():.6f} "
            f"mean={predictions['stable_image'].mean().item():.6f}"
        )
        print(f"blend mean: {predictions['blend'].mean().item():.6f}")
        final = predictions["stable_image"]
        weights = final.new_tensor((0.299, 0.587, 0.114)).view(3, 1, 1)
        final_luma = (final * weights).sum(0, keepdim=True)
        final_chroma = final - final_luma
        final_chroma_mean = final_chroma.square().mean(0).add(1e-8).sqrt().mean().item()
        print(f"color alpha mean: {predictions['color_alpha_mean'].mean().item():.6f}")
        print(
            f"color contrast mean: {predictions['color_contrast_mean'].mean().item():.6f}"
        )
        print(
            f"color sky protection mean: {predictions['color_sky_protection_mean'].mean().item():.6f}"
        )
        print(
            f"color target gate mean: {predictions['color_target_gate_mean'].mean().item():.6f}"
        )
        print(
            f"color RGB pre: {predictions['color_pre_mean_rgb'].detach().cpu().tolist()}"
        )
        print(
            f"color RGB post: {predictions['color_post_mean_rgb'].detach().cpu().tolist()}"
        )
        print(
            f"color chroma pre/post: "
            f"{predictions['color_pre_mean_chroma'].mean().item():.6f}/"
            f"{predictions['color_post_mean_chroma'].mean().item():.6f}; "
            f"final={final_chroma_mean:.6f}"
        )
        print(f"final RGB mean: {final.mean(dim=(-2, -1)).detach().cpu().tolist()}")
        print(
            f"output_weights: {predictions['output_weights'].detach().cpu().tolist()}"
        )
    return predictions


@torch.inference_mode()
def predict_pil_v2(
    model: TriPAFNetV2,
    image: Image.Image,
    device: torch.device | str,
    output_key: str = "stable_image",
    amp: bool = True,
    use_detail_guidance: bool = True,
) -> torch.Tensor:
    """Predict the stable inference image or an explicitly selected raw branch."""

    outputs = predict_pil_v2_outputs(
        model,
        image,
        device,
        amp=amp,
        use_detail_guidance=use_detail_guidance,
    )
    if output_key not in outputs:
        raise KeyError(f"Unknown v2 output key: {output_key}")
    return outputs[output_key]


__all__ = [
    "load_v2_checkpoint",
    "predict_pil_v2",
    "predict_pil_v2_outputs",
    "prepare_priors",
    "prepare_priors_on_device",
    "tensor_to_pil",
]
