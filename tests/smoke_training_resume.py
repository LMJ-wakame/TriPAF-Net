"""Run a two-epoch tiny-data training/resume smoke check."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from training.train_tripaf_v2 import parse_args, train


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tripaf_resume_") as temporary:
        root = Path(temporary)
        hazy_dir = root / "hazy"
        clean_dir = root / "clean"
        hazy_dir.mkdir()
        clean_dir.mkdir()
        rows = []
        splits = ("train", "train", "train", "val", "test", "test")
        for index, split in enumerate(splits):
            identifier = f"{index:06d}"
            generator = np.random.default_rng(index)
            clean = generator.integers(0, 220, size=(40, 40, 3), dtype=np.uint8)
            hazy = np.clip(clean.astype(np.int16) + 25, 0, 255).astype(np.uint8)
            Image.fromarray(clean).save(clean_dir / f"{identifier}.png")
            Image.fromarray(hazy).save(hazy_dir / f"{identifier}.png")
            rows.append(
                {
                    "id": identifier,
                    "split": split,
                    "fog_density": 20 + index * 5,
                }
            )
        metadata = root / "metadata.csv"
        with metadata.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        output = root / "checkpoints"
        common = [
            "--hazy-dir",
            str(hazy_dir),
            "--clean-dir",
            str(clean_dir),
            "--metadata-csv",
            str(metadata),
            "--output-dir",
            str(output),
            "--base-channels",
            "8",
            "--crop-size",
            "32",
            "--batch-size",
            "1",
            "--accumulation-steps",
            "2",
            "--workers",
            "0",
            "--device",
            "cuda" if torch.cuda.is_available() else "cpu",
        ]
        train(parse_args([*common, "--epochs", "1"]))
        train(parse_args([*common, "--epochs", "2"]))
        payload = torch.load(
            output / "seed_42" / "last.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert payload["epoch"] == 2
        for key in (
            "train_model",
            "ema_model",
            "optimizer",
            "scheduler",
            "scaler",
            "rng_state",
            "config_sha256",
            "split_manifest_sha256",
        ):
            assert key in payload
        print("tiny training/resume smoke passed")


if __name__ == "__main__":
    main()
