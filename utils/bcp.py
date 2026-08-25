import cv2
import numpy as np


def bright_channel(img, size=5):
    if img.dtype != np.uint8 and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    max_channel = np.max(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    bright = cv2.dilate(max_channel, kernel)
    return bright
