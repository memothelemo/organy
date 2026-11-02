"""
Quick and dirty reusable utility functions for OpenCV2.
"""

import cv2

from numpy.typing import NDArray


def downscale(image: NDArray, longest_side: int) -> NDArray:
    """
    Shrinks an image where its longer side is at most `longest_side` pixels
    while keeping its aspect ratio. Smaller images are returned as they are
    to retain its image quality.
    """
    height, width = image.shape[:2]
    scale = longest_side / max(height, width)
    if scale >= 1: return image

    size = (max(1, round(width * scale)), max(1, round(height * scale)))

    # INTER_AREA averages neighbouring pixels, which avoids aliasing when shrinking.
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
