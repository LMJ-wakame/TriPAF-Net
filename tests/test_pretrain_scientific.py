from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn

from losses.task_aware import detector_feature_loss
from losses.tripaf import tripaf_v2_loss
from models.tripafnet_v2 import TriPAFNetV2
from tools.generate_carla_dataset import actor_class_id
from training.train_tripaf_v2 import read_locked_splits
from utils.inference_v2 import load_v2_checkpoint


def test_required_learned_components_receive_nonzero_gradients() -> None:
    torch.manual_seed(17)
    model = TriPAFNetV2(base_channels=8, adaptive_fusion=True)
    rgb = torch.rand(2, 3, 64, 64)
    prior = torch.rand(2, 1, 64, 64)
    outputs = model(rgb, prior, prior, prior)
    outputs["input"] = rgb
    loss, _ = tripaf_v2_loss(
        outputs,
        torch.rand_like(rgb),
        torch.tensor([22.0, 47.0]),
        {
            "charbonnier": 1.0,
            "ssim": 0.2,
            "gradient": 0.1,
            "severity": 0.2,
            "identity": 0.15,
            "severity_gamma": 2.0,
        },
    )
    loss.backward()
    required = [
        "decoder.0",
        "severity_head",
        "final_fusion_head",
    ]
    for scale in range(5):
        required.extend(
            (
                f"fusions.{scale}.channel_gate",
                f"fusions.{scale}.spatial_gate",
                f"fusions.{scale}.context_gate",
                f"fusions.{scale}.adaptivity_scale",
                f"fusions.{scale}.detail_gate",
                f"fusions.{scale}.detail_scale",
            )
        )
    parameters = dict(model.named_parameters())
    for prefix in required:
        gradients = [
            parameter.grad
            for name, parameter in parameters.items()
            if name == prefix or name.startswith(prefix + ".")
        ]
        assert gradients, prefix
        assert any(
            gradient is not None and gradient.abs().sum() > 0 for gradient in gradients
        ), prefix


def test_fixed_ablation_is_parameter_matched_and_uses_half_gates() -> None:
    adaptive = TriPAFNetV2(base_channels=8, adaptive_fusion=True)
    fixed = TriPAFNetV2(base_channels=8, adaptive_fusion=False)
    assert sum(value.numel() for value in adaptive.parameters()) == sum(
        value.numel() for value in fixed.parameters()
    )
    rgb = torch.rand(1, 3, 64, 64)
    prior = torch.rand(1, 1, 64, 64)
    outputs = fixed(rgb, prior, prior, prior)
    torch.testing.assert_close(
        outputs["mean_prior_gate"],
        torch.full_like(outputs["mean_prior_gate"], 0.5),
    )
    torch.testing.assert_close(
        outputs["mean_detail_gate"],
        torch.full_like(outputs["mean_detail_gate"], 0.5),
    )


class DummyFrozenDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Conv2d(3, 4, 3, padding=1)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        return [self.layer(image)]


def test_frozen_detector_loss_backpropagates_only_to_prediction() -> None:
    detector = DummyFrozenDetector()
    prediction = torch.rand(1, 3, 16, 16, requires_grad=True)
    target = torch.rand_like(prediction)
    detector_feature_loss(detector, prediction, target).backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in detector.parameters())


def test_locked_split_is_seed_independent(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    fields = ["id", "split", "fog_density"]
    rows = [
        {"id": "a", "split": "train", "fog_density": "22"},
        {"id": "b", "split": "val", "fog_density": "35"},
        {"id": "c", "split": "test", "fog_density": "47"},
    ]
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    first, _, first_hash = read_locked_splits(metadata, {"a", "b", "c"})
    second, _, second_hash = read_locked_splits(metadata, {"a", "b", "c"})
    assert first == second
    assert first_hash == second_hash


def test_carla_native_labels_use_frozen_yolo_coco_ids() -> None:
    class Actor:
        def __init__(self, type_id: str, base_type: str) -> None:
            self.type_id = type_id
            self.attributes = {"base_type": base_type}

    assert actor_class_id(Actor("walker.pedestrian.0001", "")) == 0
    assert actor_class_id(Actor("vehicle.test", "bicycle")) == 1
    assert actor_class_id(Actor("vehicle.test", "car")) == 2
    assert actor_class_id(Actor("vehicle.test", "motorcycle")) == 3
    assert actor_class_id(Actor("vehicle.test", "bus")) == 5
    assert actor_class_id(Actor("vehicle.test", "truck")) == 7


def test_format_v3_checkpoints_load_strictly(tmp_path: Path) -> None:
    model = TriPAFNetV2(base_channels=8)
    payload = {
        "format_version": 3,
        "adaptive_fusion": True,
        "base_channels": 8,
        "model": model.state_dict(),
    }
    valid = tmp_path / "valid.pt"
    torch.save(payload, valid)
    loaded, metadata = load_v2_checkpoint(valid, "cpu")
    assert loaded.adaptive_fusion
    assert metadata["checkpoint_compatibility"]["missing_keys"] == []
    assert metadata["checkpoint_compatibility"]["unexpected_keys"] == []

    incomplete_state = dict(model.state_dict())
    incomplete_state.pop(next(iter(incomplete_state)))
    invalid = tmp_path / "invalid.pt"
    torch.save({**payload, "model": incomplete_state}, invalid)
    try:
        load_v2_checkpoint(invalid, "cpu")
    except RuntimeError:
        pass
    else:
        raise AssertionError("format-v3 checkpoints must not load partial weights")
