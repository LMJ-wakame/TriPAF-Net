import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets.paired_dehaze import PairedDehazeDataset, deterministic_split
from models.tripafnet_v2 import TriPAFNetV2, haze_descriptor
from utils.classical import dcp_bcp_dehaze


class PipelineTests(unittest.TestCase):
    def test_split_is_complete_disjoint_and_deterministic(self):
        stems = [f"{index:04d}" for index in range(100)]
        first = deterministic_split(stems, seed=7)
        second = deterministic_split(reversed(stems), seed=7)
        self.assertEqual(first, second)
        train, val, test = map(set, (first["train"], first["val"], first["test"]))
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(train | val | test, set(stems))

    def test_dataset_keeps_binary_sky_mask_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "hazy").mkdir()
            (root / "clean").mkdir()
            image = np.full((64, 64, 3), 250, dtype=np.uint8)
            Image.fromarray(image).save(root / "hazy" / "sample.png")
            Image.fromarray(image).save(root / "clean" / "sample.png")
            dataset = PairedDehazeDataset(
                root / "hazy", root / "clean", crop_size=64, training=False
            )
            sample = dataset[0]
            self.assertEqual(float(sample["sky"].max()), 1.0)
            self.assertGreater(float(sample["sky"].sum()), 0.0)

    def test_dcp_baseline_is_finite(self):
        image = np.full((48, 64, 3), 180, dtype=np.uint8)
        image[12:36, 16:48] = (60, 90, 120)
        restored, transmission, atmospheric = dcp_bcp_dehaze(image)
        self.assertTrue(np.isfinite(restored).all())
        self.assertTrue(np.isfinite(transmission).all())
        self.assertTrue(np.isfinite(atmospheric).all())
        self.assertGreaterEqual(float(transmission.min()), 0.35)

    def test_v2_is_parameter_matched_and_outputs_context(self):
        adaptive = TriPAFNetV2(base_channels=8, adaptive_fusion=True).eval()
        fixed = TriPAFNetV2(base_channels=8, adaptive_fusion=False).eval()
        adaptive.load_state_dict(fixed.state_dict())
        self.assertEqual(
            sum(parameter.numel() for parameter in adaptive.parameters()),
            sum(parameter.numel() for parameter in fixed.parameters()),
        )
        rgb = torch.rand(1, 3, 64, 80)
        prior = torch.rand(1, 1, 64, 80)
        with torch.no_grad():
            outputs = adaptive(rgb, prior, prior, prior)
            fixed_outputs = fixed(rgb, prior, prior, prior)
        self.assertEqual(outputs["image"].shape, rgb.shape)
        self.assertEqual(outputs["severity"].shape, (1, 1))
        self.assertTrue(torch.isfinite(outputs["image"]).all())
        self.assertTrue(
            torch.allclose(fixed_outputs["mean_prior_gate"], torch.full((1, 5), 0.5))
        )
        self.assertTrue(torch.allclose(outputs["output_weights"].sum(1), torch.ones(1)))
        self.assertEqual(haze_descriptor(rgb, prior, prior, prior).shape, (1, 5))


if __name__ == "__main__":
    unittest.main()
