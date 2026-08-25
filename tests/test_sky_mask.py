import numpy as np
from PIL import Image

from utils.sky_mask import simple_sky_mask_pil


def test_sky_mask_handles_non_sky_and_sky_regions():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :32] = 200
    img[:, 32:] = 250

    pil = Image.fromarray(img, mode="RGB")
    mask = simple_sky_mask_pil(pil, threshold=220)

    assert mask.shape == (64, 64)
    assert mask[:, :32].sum() == 0
    assert mask[:, 32:].sum() > 0
