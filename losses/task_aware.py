"""Optional detail and detector criteria retained for controlled experiments."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.restoration import (
    charbonnier_loss,
    color_loss,
    edge_aware_transmission_loss,
    gradient_loss,
)
from losses.ssim import ssim_loss


def laplacian_pyramid_loss(
    prediction: torch.Tensor, target: torch.Tensor, levels: int = 3
) -> torch.Tensor:
    """Match residual detail at several spatial scales."""

    total = prediction.new_zeros(())
    pred_level, target_level = prediction, target
    for level in range(levels):
        pred_blur = F.avg_pool2d(pred_level, 3, stride=1, padding=1)
        target_blur = F.avg_pool2d(target_level, 3, stride=1, padding=1)
        total = total + F.l1_loss(
            pred_level - pred_blur, target_level - target_blur
        ) / (2**level)
        if level + 1 < levels:
            pred_level = F.avg_pool2d(pred_level, 2)
            target_level = F.avg_pool2d(target_level, 2)
    return total


def task_aware_restoration_loss(
    outputs: dict[str, torch.Tensor],
    hazy: torch.Tensor,
    clean: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    task_image = outputs["image"]
    prediction = outputs["restoration"]
    direct = outputs["direct"]
    transmission = outputs["transmission"]
    atmospheric = outputs["atmospheric_light"]
    blend = outputs["blend"]
    input_detail = hazy - F.avg_pool2d(hazy, 3, stride=1, padding=1)
    detail_energy = input_detail.abs().mean(dim=1, keepdim=True).detach()

    terms = {
        "mse": F.mse_loss(prediction, clean),
        "charbonnier": charbonnier_loss(prediction, clean),
        "ssim": ssim_loss(prediction, clean),
        "gradient": gradient_loss(prediction, clean),
        "laplacian": laplacian_pyramid_loss(prediction, clean),
        "color": color_loss(prediction, clean),
        "direct": charbonnier_loss(direct, clean),
        "task_mse": F.mse_loss(task_image, clean),
        "task_charbonnier": charbonnier_loss(task_image, clean),
        "task_gradient": gradient_loss(task_image, clean),
        "transmission_smooth": edge_aware_transmission_loss(transmission, hazy),
        "physics": F.l1_loss(
            prediction * transmission + atmospheric * (1.0 - transmission), hazy
        ),
        # Penalize use of the more fragile physical branch at strong input edges.
        "edge_blend": ((1.0 - blend) * detail_energy).mean(),
    }
    weights = {
        # MSE is explicit because PSNR is the primary v1 acceptance threshold;
        # SSIM/color/detail terms retain v2's existing perceptual advantages.
        "mse": 1.0,
        "charbonnier": 0.75,
        "ssim": 0.25,
        "gradient": 0.12,
        "laplacian": 0.08,
        "color": 0.05,
        "direct": 0.10,
        "task_mse": 1.0,
        "task_charbonnier": 0.50,
        "task_gradient": 0.05,
        "transmission_smooth": 0.005,
        "physics": 0.01,
        "edge_blend": 0.01,
    }
    total = sum(weights[name] * value for name, value in terms.items())
    return total, terms


class FrozenYOLOBackbone(nn.Module):
    """Frozen YOLOv8 backbone stages used as a task-aware feature criterion."""

    def __init__(
        self, weights: str | Path, layer_indices: tuple[int, ...] = (4, 6, 9)
    ) -> None:
        super().__init__()
        config_dir = Path("artifacts/ultralytics_config").resolve()
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Detector feature loss requires an environment containing ultralytics. "
                "Use requirements-yolo.txt or pass --detector-loss-weight 0."
            ) from error

        detector = YOLO(str(weights)).model
        self.layer_indices = tuple(sorted(layer_indices))
        self.layers = nn.ModuleList(list(detector.model[: max(self.layer_indices) + 1]))
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenYOLOBackbone":
        super().train(False)
        return self

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        features = []
        value = image
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index in self.layer_indices:
                features.append(value)
        return features


def detector_feature_loss(
    backbone: FrozenYOLOBackbone,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Match clear-image YOLO features while allowing gradients only to restoration."""

    predicted_features = backbone(prediction)
    with torch.no_grad():
        target_features = backbone(target)
    total = prediction.new_zeros(())
    for predicted, reference in zip(predicted_features, target_features):
        scale = reference.detach().abs().mean().clamp_min(0.05)
        relative_l1 = F.smooth_l1_loss(predicted, reference, beta=0.02) / scale
        cosine = 1.0 - F.cosine_similarity(predicted, reference, dim=1).mean()
        total = total + 0.5 * relative_l1 + 0.5 * cosine
    return total / max(1, len(predicted_features))


class FrozenYOLOTaskNetwork(nn.Module):
    """Complete frozen YOLOv8 model for differentiable detection distillation."""

    def __init__(self, weights: str | Path) -> None:
        super().__init__()
        config_dir = Path("artifacts/ultralytics_config").resolve()
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Detection distillation requires an environment containing ultralytics."
            ) from error
        self.detector = YOLO(str(weights)).model
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenYOLOTaskNetwork":
        super().train(False)
        return self

    def forward(
        self, image: torch.Tensor
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        output = self.detector(image)
        if not isinstance(output, tuple) or not isinstance(output[1], dict):
            raise RuntimeError(
                "Unexpected Ultralytics inference output for detection distillation"
            )
        return output[1]


def detection_distillation_loss(
    detector: FrozenYOLOTaskNetwork,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Match clear-image pre-NMS class, box-distribution, and neck features."""

    predicted = detector(prediction)
    with torch.no_grad():
        reference = detector(target)
    score_loss = F.smooth_l1_loss(predicted["scores"], reference["scores"], beta=0.1)
    box_loss = F.smooth_l1_loss(predicted["boxes"], reference["boxes"], beta=0.1)
    feature_loss = prediction.new_zeros(())
    for predicted_feature, reference_feature in zip(
        predicted["feats"], reference["feats"]
    ):
        scale = reference_feature.detach().abs().mean().clamp_min(0.05)
        feature_loss = (
            feature_loss
            + F.smooth_l1_loss(predicted_feature, reference_feature, beta=0.02) / scale
        )
    feature_loss = feature_loss / max(1, len(predicted["feats"]))
    return 0.50 * score_loss + 0.25 * box_loss + 0.25 * feature_loss
