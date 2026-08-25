"""Classical dark/bright-channel baselines used in controlled evaluations."""

from __future__ import annotations

import cv2
import numpy as np

from utils.bcp import bright_channel
from utils.dcp import (
    dark_channel,
    estimate_atmospheric_light,
    transmission_from_dark_channel,
)


def guided_filter_gray(
    guide: np.ndarray, source: np.ndarray, radius: int = 20, eps: float = 1e-3
) -> np.ndarray:
    guide = guide.astype(np.float32)
    source = source.astype(np.float32)
    kernel = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(source, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
    corr_i = cv2.boxFilter(
        guide * guide, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT
    )
    corr_ip = cv2.boxFilter(
        guide * source, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT
    )
    variance_i = corr_i - mean_i * mean_i
    covariance_ip = corr_ip - mean_i * mean_p
    a = covariance_ip / (variance_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
    return mean_a * guide + mean_b


def dcp_dehaze(
    image: np.ndarray,
    omega: float = 0.82,
    minimum_transmission: float = 0.35,
    patch_size: int = 7,
    guided_radius: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dehazed RGB, refined transmission, and atmospheric light.

    Conservative defaults are intentional: an earlier configuration amplified
    estimation errors into black regions and white edge halos.
    """
    uint8 = (
        image
        if image.dtype == np.uint8
        else np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    )
    rgb = uint8.astype(np.float32) / 255.0
    dark = dark_channel(uint8, size=patch_size)
    atmospheric = estimate_atmospheric_light(uint8, dark)
    transmission = transmission_from_dark_channel(
        uint8,
        atmospheric,
        omega=omega,
        t0=minimum_transmission,
        size=patch_size,
    )
    guide = cv2.cvtColor(uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    transmission = guided_filter_gray(
        guide, transmission, radius=guided_radius, eps=1e-3
    )
    transmission = np.clip(transmission, minimum_transmission, 1.0)
    restored = (rgb - atmospheric[None, None, :]) / transmission[
        ..., None
    ] + atmospheric[None, None, :]
    return np.clip(restored, 0.0, 1.0), transmission, atmospheric


def dcp_bcp_dehaze(
    image: np.ndarray,
    omega: float = 0.95,
    minimum_transmission: float = 0.10,
    patch_size: int = 15,
    guided_radius: int = 40,
    bcp_adjustment: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dehaze with fused dark- and bright-channel transmission estimates.

    This is an efficient, deterministic reimplementation of Li et al. (2023):
    Otsu supplies the sky fraction, DCP estimates non-sky transmission, the
    improved BCP estimates bright/sky transmission, and the two physical
    estimates are fused before guided refinement.  Particle-swarm Otsu and
    gradient-domain guided filtering are replaced by their standard fast
    counterparts so the baseline remains practical at 1024 x 1024.
    """
    uint8 = (
        image
        if image.dtype == np.uint8
        else np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    )
    rgb = uint8.astype(np.float32) / 255.0
    dark = dark_channel(uint8, size=patch_size)
    bright = bright_channel(uint8, size=patch_size).astype(np.float32) / 255.0

    atmospheric_dcp = estimate_atmospheric_light(uint8, dark)
    transmission_dcp = transmission_from_dark_channel(
        uint8, atmospheric_dcp, omega=omega, t0=minimum_transmission, size=patch_size
    )

    # Bright-channel atmospheric light: mean RGB of the top 0.1% BCP pixels.
    count = max(1, int(bright.size * 0.001))
    indices = np.argpartition(bright.reshape(-1), -count)[-count:]
    atmospheric_bcp = rgb.reshape(-1, 3)[indices].mean(axis=0)
    atmospheric_bcp = np.clip(atmospheric_bcp, 0.05, 0.98)
    denominator = np.clip(1.0 - float(atmospheric_bcp.max()), 0.02, 1.0)
    difference = np.abs(bright - float(atmospheric_bcp.max()))
    transmission_bcp = difference / denominator
    transmission_bcp = np.where(
        difference < difference.mean(),
        (difference + bcp_adjustment) / denominator,
        transmission_bcp,
    )
    transmission_bcp = np.clip(transmission_bcp, minimum_transmission, 1.0)

    gray = cv2.cvtColor(uint8, cv2.COLOR_RGB2GRAY)
    _, sky_binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sky_fraction = float(sky_binary.mean())
    non_sky_fraction = max(1e-6, 1.0 - sky_fraction)
    sigma = 0.10 * np.exp(-sky_fraction / non_sky_fraction) - 0.05
    transmission = (
        sky_fraction * transmission_bcp
        + (1.0 - sky_fraction) * transmission_dcp
        - sigma
    )
    guide = gray.astype(np.float32) / 255.0
    transmission = guided_filter_gray(
        guide, transmission, radius=guided_radius, eps=1e-3
    )
    transmission = np.clip(transmission, minimum_transmission, 1.0)

    atmospheric = (
        sky_fraction * atmospheric_dcp + (1.0 - sky_fraction) * atmospheric_bcp
    )
    restored = (rgb - atmospheric[None, None, :]) / transmission[
        ..., None
    ] + atmospheric[None, None, :]
    return np.clip(restored, 0.0, 1.0), transmission, atmospheric
