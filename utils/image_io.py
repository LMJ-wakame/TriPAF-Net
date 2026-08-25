"""Image conversion, prior preparation, and padding utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.bcp import bright_channel
from utils.dcp import dark_channel
from utils.sky_mask import simple_sky_mask_pil


def prepare_priors(image: Image.Image) -> dict[str, torch.Tensor]:
    image = image.convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    rgb = (
        torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        .float()
        .div_(255.0)
    )
    dark = torch.from_numpy(
        dark_channel(array, size=7).astype(np.float32) / 255.0
    ).unsqueeze(0)
    bright = torch.from_numpy(
        bright_channel(array, size=7).astype(np.float32) / 255.0
    ).unsqueeze(0)
    sky = torch.from_numpy(simple_sky_mask_pil(image).astype(np.float32)).unsqueeze(0)
    return {
        "rgb": rgb.unsqueeze(0),
        "dark": dark.unsqueeze(0),
        "bright": bright.unsqueeze(0),
        "sky": sky.unsqueeze(0),
    }


def _rgb_to_hsv(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum, maximum_index = rgb.max(dim=1, keepdim=True)
    minimum = rgb.min(dim=1, keepdim=True).values
    delta = maximum - minimum
    saturation = delta / maximum.clamp_min(1e-6)
    red, green, blue = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    hue_parts = torch.cat(
        [
            torch.remainder((green - blue) / delta.clamp_min(1e-6), 6.0),
            (blue - red) / delta.clamp_min(1e-6) + 2.0,
            (red - green) / delta.clamp_min(1e-6) + 4.0,
        ],
        dim=1,
    )
    hue = hue_parts.gather(1, maximum_index) / 6.0
    hue = torch.where(delta > 1e-6, hue, torch.zeros_like(hue))
    return hue, saturation, maximum


def _binary_morphology(mask: torch.Tensor) -> torch.Tensor:
    kernel = mask.new_tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    kernel_sum = float(kernel.sum())

    def erode(value: torch.Tensor) -> torch.Tensor:
        return (F.conv2d(value, kernel, padding=1) >= kernel_sum).to(value.dtype)

    def dilate(value: torch.Tensor) -> torch.Tensor:
        return (F.conv2d(value, kernel, padding=1) > 0).to(value.dtype)

    opened = dilate(erode(mask))
    return erode(dilate(opened))


def prepare_priors_on_device(
    image: Image.Image, device: torch.device | str
) -> dict[str, torch.Tensor]:
    device = torch.device(device)
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb = (
        torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )
    dark = -F.max_pool2d(
        -rgb.min(dim=1, keepdim=True).values, kernel_size=7, stride=1, padding=3
    )
    bright = F.max_pool2d(
        rgb.max(dim=1, keepdim=True).values, kernel_size=7, stride=1, padding=3
    )
    hue, saturation, value = _rgb_to_hsv(rgb)
    sky = (
        (value > (202.0 / 255.0))
        & (saturation < (50.0 / 255.0))
        & (hue < (50.0 / 180.0))
    ).to(rgb.dtype)
    sky = _binary_morphology(sky)
    return {"rgb": rgb, "dark": dark, "bright": bright, "sky": sky}


def pad_inputs(
    inputs: dict[str, torch.Tensor], multiple: int = 16
) -> tuple[dict[str, torch.Tensor], int, int]:
    height, width = inputs["rgb"].shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return inputs, 0, 0
    mode = "reflect" if height > 1 and width > 1 else "replicate"
    padded = {
        key: F.pad(value, (0, pad_w, 0, pad_h), mode=mode)
        for key, value in inputs.items()
    }
    return padded, pad_h, pad_w


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB")
