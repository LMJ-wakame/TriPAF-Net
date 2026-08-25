"""Paired image loading for reproducible dehazing experiments.

The original loader paired two independently sorted directory listings.  A
missing file could therefore silently associate a hazy image with the wrong
clear target.  This loader matches files by stem, validates the dataset, and
applies exactly the same spatial augmentation to both members of a pair.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.bcp import bright_channel
from utils.dcp import dark_channel
from utils.sky_mask import simple_sky_mask_pil

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _images_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in result:
                raise ValueError(f"Duplicate image stem {path.stem!r} in {directory}")
            result[path.stem] = path
    return result


def discover_pairs(
    hazy_dir: str | Path, clean_dir: str | Path
) -> List[Tuple[str, Path, Path]]:
    hazy = _images_by_stem(Path(hazy_dir))
    clean = _images_by_stem(Path(clean_dir))
    common = sorted(set(hazy) & set(clean))
    missing_clean = sorted(set(hazy) - set(clean))
    missing_hazy = sorted(set(clean) - set(hazy))
    if missing_clean or missing_hazy:
        details = []
        if missing_clean:
            details.append(f"{len(missing_clean)} hazy files lack clear targets")
        if missing_hazy:
            details.append(f"{len(missing_hazy)} clear files lack hazy inputs")
        raise ValueError("Pair validation failed: " + "; ".join(details))
    if not common:
        raise ValueError(f"No paired images found in {hazy_dir} and {clean_dir}")
    return [(stem, hazy[stem], clean[stem]) for stem in common]


def deterministic_split(
    stems: Iterable[str],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, List[str]]:
    """Split stable identifiers without depending on filesystem order."""
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError(
            "Expected train_fraction > 0, val_fraction >= 0, and train+val < 1"
        )
    keyed = []
    for stem in stems:
        digest = hashlib.sha256(f"{seed}:{stem}".encode("utf-8")).hexdigest()
        keyed.append((digest, stem))
    ordered = [stem for _, stem in sorted(keyed)]
    n = len(ordered)
    n_train = max(1, int(round(n * train_fraction)))
    n_val = int(round(n * val_fraction))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


def _resize_minimum(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h >= size and w >= size:
        return image
    scale = max(size / max(h, 1), size / max(w, 1))
    new_w = max(size, int(round(w * scale)))
    new_h = max(size, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(array).float().div_(255.0)


class PairedDehazeDataset(Dataset):
    def __init__(
        self,
        hazy_dir: str | Path,
        clean_dir: str | Path,
        stems: Sequence[str] | None = None,
        crop_size: int = 384,
        training: bool = True,
        horizontal_flip: bool = True,
    ) -> None:
        pairs = discover_pairs(hazy_dir, clean_dir)
        if stems is not None:
            wanted = set(stems)
            pairs = [pair for pair in pairs if pair[0] in wanted]
            absent = wanted - {pair[0] for pair in pairs}
            if absent:
                raise ValueError(f"Requested {len(absent)} unknown pair stems")
        if not pairs:
            raise ValueError("The selected dataset split is empty")
        self.pairs = pairs
        self.crop_size = int(crop_size)
        self.training = bool(training)
        self.horizontal_flip = bool(horizontal_flip)

    def __len__(self) -> int:
        return len(self.pairs)

    def _aligned_crop(
        self, hazy: np.ndarray, clean: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        size = self.crop_size
        hazy = _resize_minimum(hazy, size)
        clean = _resize_minimum(clean, size)
        h, w = hazy.shape[:2]
        if clean.shape[:2] != (h, w):
            clean = cv2.resize(clean, (w, h), interpolation=cv2.INTER_CUBIC)
        if self.training:
            top = random.randint(0, h - size)
            left = random.randint(0, w - size)
        else:
            top = (h - size) // 2
            left = (w - size) // 2
        hazy = hazy[top : top + size, left : left + size]
        clean = clean[top : top + size, left : left + size]
        if self.training and self.horizontal_flip and random.random() < 0.5:
            hazy = hazy[:, ::-1]
            clean = clean[:, ::-1]
        return np.ascontiguousarray(hazy), np.ascontiguousarray(clean)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        stem, hazy_path, clean_path = self.pairs[index]
        hazy = np.asarray(Image.open(hazy_path).convert("RGB"), dtype=np.uint8)
        clean = np.asarray(Image.open(clean_path).convert("RGB"), dtype=np.uint8)
        hazy, clean = self._aligned_crop(hazy, clean)

        dark = dark_channel(hazy, size=7).astype(np.float32) / 255.0
        bright = bright_channel(hazy, size=7).astype(np.float32) / 255.0
        sky = simple_sky_mask_pil(Image.fromarray(hazy)).astype(np.float32)

        return {
            "id": stem,
            "hazy": _to_tensor(hazy),
            "clean": _to_tensor(clean),
            "dark": torch.from_numpy(dark).unsqueeze(0),
            "bright": torch.from_numpy(bright).unsqueeze(0),
            "sky": torch.from_numpy(sky).unsqueeze(0),
        }
