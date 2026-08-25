import cv2
import numpy as np

from .bcp import bright_channel


def dark_channel(img, size=5):
    """Compute the dark channel of an RGB image.

    Accepts uint8 images in [0,255] or float images in [0,1].
    """
    if img.dtype != np.uint8 and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return dark


def estimate_atmospheric_light(img, dark, top_percent=0.001, bright_percent=0.001):
    """Estimate atmospheric light A using dark-channel candidates and bright-channel guidance."""
    img = np.asarray(img)
    if img.dtype != np.uint8 and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    h, w = dark.shape
    num = max(1, int(h * w * top_percent))
    flat_dark = dark.reshape(-1)
    if num >= flat_dark.size:
        idx_dark = np.arange(flat_dark.size)
    else:
        idx_dark = np.argpartition(flat_dark, -num)[-num:]

    img_flat = img.reshape(-1, img.shape[-1]).astype(np.float32)
    dark_candidates = img_flat[idx_dark]

    if bright_percent is not None and bright_percent > 0:
        bright = bright_channel(img)
        flat_bright = bright.reshape(-1)
        num_bright = max(1, int(h * w * bright_percent))
        if num_bright >= flat_bright.size:
            idx_bright = np.arange(flat_bright.size)
        else:
            idx_bright = np.argpartition(flat_bright, -num_bright)[-num_bright:]
        idx_combined = np.intersect1d(idx_dark, idx_bright)
        if idx_combined.size > 0:
            candidates = img_flat[idx_combined]
        else:
            candidates = dark_candidates
    else:
        candidates = dark_candidates

    if candidates.size == 0:
        candidates = img_flat[idx_dark]

    sums = np.sum(candidates, axis=1)
    top_n = max(1, int(len(candidates) * 0.1))
    top_idx = np.argpartition(sums, -top_n)[-top_n:]
    A = candidates[top_idx].mean(axis=0)

    image_mean = img_flat.mean(axis=0)
    A = np.minimum(A, np.maximum(image_mean * 1.05, 230.0))
    A = np.clip(A, 0, 255)
    return A / 255.0


def transmission_from_dark_channel(
    img, atmospheric_light, omega=0.85, t0=0.12, size=15
):
    """Compute a classical DCP transmission estimate from an RGB image and atmospheric light."""
    img = np.asarray(img).astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    atm = np.clip(np.asarray(atmospheric_light, dtype=np.float32), 1e-4, 1.0)
    norm_img = img / atm[None, None, :]
    norm_img = np.clip(norm_img, 0.0, 1.0)
    dark = dark_channel(norm_img, size=size).astype(np.float32) / 255.0
    t = np.clip(1.0 - omega * dark, t0, 1.0)
    return t
