import cv2
import numpy as np
import torch
import torch.nn as nn

# Lightweight sky mask heuristic + optional tiny seg net placeholder


def simple_sky_mask_pil(
    img_pil, threshold=202, saturation_threshold=50, hue_threshold=50
):
    """Return a binary sky mask for an RGB PIL image.

    Sky regions tend to be bright and relatively desaturated. We use the HSV
    value/saturation channels and clean the mask with morphology so small
    bright artifacts are not marked as sky.
    """
    img_np = np.array(img_pil.convert("RGB"), dtype=np.uint8)
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        raise ValueError("Expected an RGB image")

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32)
    value = hsv[..., 2].astype(np.float32)
    saturation = hsv[..., 1].astype(np.float32)

    mask = (
        (value > (threshold))
        & (saturation < saturation_threshold)
        & (hue < hue_threshold)
    ).astype(np.uint8)
    if not np.any(mask):
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


class TinySkySeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))
