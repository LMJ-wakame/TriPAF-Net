"""Losses used by the v2 paired dehazing experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from losses.ssim import ssim_loss


def charbonnier_loss(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3
) -> torch.Tensor:
    difference = prediction - target
    return torch.sqrt(difference * difference + epsilon * epsilon).mean()


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


def color_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = prediction.mean(dim=(-2, -1))
    target_mean = target.mean(dim=(-2, -1))
    pred_std = prediction.std(dim=(-2, -1), correction=0)
    target_std = target.std(dim=(-2, -1), correction=0)
    return F.l1_loss(pred_mean, target_mean) + 0.5 * F.l1_loss(pred_std, target_std)


def edge_aware_transmission_loss(
    transmission: torch.Tensor, hazy: torch.Tensor
) -> torch.Tensor:
    luminance = 0.299 * hazy[:, :1] + 0.587 * hazy[:, 1:2] + 0.114 * hazy[:, 2:3]
    t_x = transmission[..., :, 1:] - transmission[..., :, :-1]
    t_y = transmission[..., 1:, :] - transmission[..., :-1, :]
    i_x = luminance[..., :, 1:] - luminance[..., :, :-1]
    i_y = luminance[..., 1:, :] - luminance[..., :-1, :]
    return (t_x.abs() * torch.exp(-10.0 * i_x.abs())).mean() + (
        t_y.abs() * torch.exp(-10.0 * i_y.abs())
    ).mean()


def restoration_loss(
    outputs: dict[str, torch.Tensor],
    hazy: torch.Tensor,
    clean: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or {}
    prediction = outputs["image"]
    transmission = outputs["transmission"]
    atmospheric = outputs["atmospheric_light"]

    terms = {
        "charbonnier": charbonnier_loss(prediction, clean),
        "ssim": ssim_loss(prediction, clean),
        "gradient": gradient_loss(prediction, clean),
        "color": color_loss(prediction, clean),
        "transmission_smooth": edge_aware_transmission_loss(transmission, hazy),
        "physics": F.l1_loss(
            prediction * transmission + atmospheric * (1.0 - transmission), hazy
        ),
    }
    default = {
        "charbonnier": 1.0,
        "ssim": 0.25,
        "gradient": 0.10,
        "color": 0.05,
        "transmission_smooth": 0.01,
        "physics": 0.05,
    }
    total = sum(
        weights.get(name, default[name]) * value for name, value in terms.items()
    )
    return total, terms
