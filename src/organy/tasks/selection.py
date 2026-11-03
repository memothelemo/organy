"""
Tasks related to suggesting the best photo of each stack, by how sharp and
how well exposed each photo is.
"""

import cv2
import numpy as np

from collections.abc import Sequence
from typing import NamedTuple

from organy.models import Photograph


# Sharpness is how steep the strongest edges are: the gradient that only this
# share of pixels exceeds (as a percentile, 99.5 means the top 0.5%). Blur
# flattens the strongest edges, while noise barely changes them once a
# median filter has removed it.
#
# On the 49 stock photos, each original was compared with damaged copies of
# itself. The original measured sharper than every blurred copy and every
# heavily compressed copy, and than a noisy copy in 98 to 100% of photos.
# The variance of the Laplacian, used before, picked the noisy copy every time.
SHARPNESS_PERCENTILE = 99.5

# Pixel brightness (0 to 255) from which a pixel counts as blown out, too
# bright for any detail to be left in it.
HIGHLIGHT_LEVEL = 250

# Largest share of blown out pixels (0 to 1) for a photo to count as well
# exposed. Stage lights and bright windows are often a little blown out, so
# a small share is allowed.
#
# Dark pixels are not counted, since stage and night photos are mostly black
# on purpose.
MAX_HIGHLIGHTS = 0.02


class PhotoQuality(NamedTuple):
    """
    Raw measurements of one photo's downscaled copy. They only make sense
    compared with photos of the same stack, and none of them is a confidence.
    """

    """
    How steep the strongest edges of the grey photo are (see
    `SHARPNESS_PERCENTILE`): higher when edges are crisper. Blur and camera
    shake lower it, and noise barely changes it.
    """
    sharpness: float

    """Average grey level, from 0 (black) to 255 (white)."""
    brightness: float

    """Share of blown out pixels (see `HIGHLIGHT_LEVEL`), from 0 to 1."""
    highlights: float


def measure_quality(photo: Photograph) -> PhotoQuality:
    """
    Measures how sharp and how bright a photo is, on its downscaled copy
    (about 10 ms per photo).
    """
    gray = cv2.cvtColor(photo.decode_downscaled(), cv2.COLOR_BGR2GRAY)

    # The median filter removes pixel noise but keeps edges, so noise cannot
    # pass for detail. CV_32F keeps the negative side of each edge, which
    # 8-bit output would clip.
    smooth = cv2.medianBlur(gray, 3)
    gradient = cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))

    return PhotoQuality(
        sharpness=float(np.percentile(gradient, SHARPNESS_PERCENTILE)),
        brightness=float(gray.mean()),
        highlights=float(np.count_nonzero(gray >= HIGHLIGHT_LEVEL)) / gray.size,
    )


def suggest_best_photo(stack: Sequence[Photograph]) -> tuple[Photograph, dict[Photograph, PhotoQuality]]:
    """
    Suggests the best photo of a stack: the sharpest of its well exposed
    photos (at most `MAX_HIGHLIGHTS` blown out), or the sharpest of all of
    them when none is well exposed.

    It also returns every photo's measurements, so the user can see why
    and pick another one. The stack must not be empty.
    """
    if not stack: raise ValueError("cannot suggest the best photo of an empty stack")

    qualities = {photo: measure_quality(photo) for photo in stack}
    exposed = [photo for photo in stack if qualities[photo].highlights <= MAX_HIGHLIGHTS]

    best = max(exposed or stack, key=lambda photo: qualities[photo].sharpness)
    return best, qualities
