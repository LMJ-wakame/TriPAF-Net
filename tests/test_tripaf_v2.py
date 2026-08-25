from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets.paired_dehaze import deterministic_split, discover_pairs
from evaluation.protocol import require_model_selection_split, validate_yolo_row
from models.tripafnet_v2 import TriPAFNetV2
from utils.bcp import bright_channel
from utils.dcp import dark_channel
from utils.sky_mask import simple_sky_mask_pil


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = torch.rand(2, 3, 64, 80)
    return (
        rgb,
        torch.rand(2, 1, 64, 80),
        torch.rand(2, 1, 64, 80),
        torch.rand(2, 1, 64, 80),
    )


def test_exact_pair_matching_rejects_missing_members() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "hazy").mkdir()
        (root / "clean").mkdir()
        Image.new("RGB", (8, 8)).save(root / "hazy" / "a.png")
        Image.new("RGB", (8, 8)).save(root / "clean" / "b.png")
        try:
            discover_pairs(root / "hazy", root / "clean")
        except ValueError as error:
            assert "Pair validation failed" in str(error)
        else:
            raise AssertionError("missing pair members must be rejected")


def test_priors_and_sky_have_valid_ranges() -> None:
    image = np.full((24, 32, 3), 250, dtype=np.uint8)
    assert dark_channel(image, 7).min() >= 0
    assert dark_channel(image, 7).max() <= 255
    assert bright_channel(image, 7).min() >= 0
    assert bright_channel(image, 7).max() <= 255
    sky = simple_sky_mask_pil(Image.fromarray(image))
    assert set(np.unique(sky)).issubset({0, 1})


def test_adaptive_outputs_are_bounded_and_differentiable() -> None:
    model = TriPAFNetV2(base_channels=8, adaptive_fusion=True)
    rgb, dark, bright, sky = inputs()
    outputs = model(rgb, dark, bright, sky)
    assert outputs["image"].shape == rgb.shape
    assert torch.isfinite(outputs["image"]).all()
    assert outputs["image"].min() >= 0 and outputs["image"].max() <= 1
    assert (
        outputs["transmission"].min() >= model.t_min
        and outputs["transmission"].max() <= 1
    )
    assert outputs["severity"].min() >= 0 and outputs["severity"].max() <= 1
    torch.testing.assert_close(
        outputs["output_weights"].sum(1), torch.ones(rgb.shape[0])
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.fusions[0].adaptivity_scale.detach().clone()
    outputs["image"].mean().backward()
    assert model.fusions[0].adaptivity_scale.grad is not None
    optimizer.step()
    assert not torch.equal(before, model.fusions[0].adaptivity_scale.detach())


def test_fixed_ablation_locks_gates_to_half() -> None:
    model = TriPAFNetV2(base_channels=8, adaptive_fusion=False)
    outputs = model(*inputs())
    torch.testing.assert_close(
        outputs["mean_prior_gate"], torch.full_like(outputs["mean_prior_gate"], 0.5)
    )
    torch.testing.assert_close(
        outputs["mean_detail_gate"], torch.full_like(outputs["mean_detail_gate"], 0.5)
    )


def test_split_and_protocol_safeguards() -> None:
    split = deterministic_split(
        [str(index) for index in range(20)], train_fraction=0.70, val_fraction=0.15
    )
    assert not (set(split["train"]) & set(split["val"]))
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["val"]) & set(split["test"]))
    try:
        require_model_selection_split("test")
    except ValueError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("test split must not be usable for calibration")
    validate_yolo_row(2, 0.5, 0.5, 0.2, 0.2)
    try:
        validate_yolo_row(2, 1.1, 0.5, 0.2, 0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid YOLO values must be rejected")
