"""Loss functions used by the active TriPAF v2 training path."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from losses.ssim import ssim_loss


def charbonnier(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3
) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + epsilon * epsilon).mean()


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return functional.l1_loss(pred_dx, target_dx) + functional.l1_loss(
        pred_dy, target_dy
    )


def severity_target(fog_density: torch.Tensor) -> torch.Tensor:
    """CARLA-only severity supervision; no density is needed at inference."""
    return ((fog_density.float() - 20.0) / 30.0).clamp(0.0, 1.0).view(-1, 1)


def tripaf_v2_loss(
    outputs: dict[str, torch.Tensor],
    clean: torch.Tensor,
    fog_density: torch.Tensor | None,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    restoration = charbonnier(outputs["image"], clean)
    # Variance subtraction in SSIM is sensitive to fp16 cancellation.
    structural = ssim_loss(outputs["image"].float(), clean.float())
    gradients = gradient_loss(outputs["image"], clean)
    total = (
        weights.get("charbonnier", 1.0) * restoration
        + weights.get("ssim", 0.0) * structural
        + weights.get("gradient", 0.0) * gradients
    )
    terms = {
        "restoration": restoration,
        "ssim": structural,
        "gradient": gradients,
    }
    if fog_density is not None:
        target = severity_target(fog_density).to(outputs["severity"].device)
        severity = functional.smooth_l1_loss(outputs["severity"], target)
        identity_weight = (
            (1.0 - target).pow(weights.get("severity_gamma", 2.0)).view(-1, 1, 1, 1)
        )
        identity = (
            (identity_weight * (outputs["image"] - outputs["input"]).abs()).mean()
            if "input" in outputs
            else outputs["image"].new_zeros(())
        )
        total = (
            total
            + weights.get("severity", 0.0) * severity
            + weights.get("identity", 0.0) * identity
        )
        terms.update({"severity": severity, "identity": identity})
    return total, terms
