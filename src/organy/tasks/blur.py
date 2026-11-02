import math

import cv2
import numpy as np

from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Any

from organy.models import Photograph

# Blur radii, in pixels of the downscaled photo, of the fine and coarse views
# compared below. Detail that is already gone at the fine scale cannot get
# much weaker at the coarse one, which is what gives a smeared direction away.
FINE_SCALE = 1.0
COARSE_SCALE = 2.0

# The photo is split into a GRID x GRID set of tiles, and only tiles with some
# detail in them are measured. Flat areas such as sky or a plain wall carry
# little besides sensor noise and JPEG blocking, which say nothing about blur.
GRID = 8

# Least average edge strength (out of about 1,000 for a black to white step)
# for a tile to count as having detail, and the fewest such tiles a photo needs
# before it is judged at all.
MIN_TILE_DETAIL = 3.0
MIN_DETAILED_TILES = 6

# Detail falloff is how fast detail fades between the fine and coarse scale,
# measured along one direction: about 0.5 for sharp edges, up to 1 for fine
# texture, and near 0 once a blur has already removed the fine detail.
#
# A photo is flagged when, along one direction, the falloff is at most
# MAX_BLURRED_FALLOFF and it is at least MIN_FALLOFF_SPREAD lower than along
# the direction with the most falloff. Defocus lowers every direction alike,
# so the spread stays small, while a sharp photo never falls this low.
#
# On synthetic test images, no sharp photo was flagged (the lowest falloff was
# 0.36). A straight blur of 9 px or more was flagged in 24 of 25 photos for
# textured scenes, buildings and people, but only 10 of 25 for photos filled
# with stripes, and blurs of 5 px were mostly missed. Strong defocus was
# flagged in 20 of 25 photos filled with stripes and 5 of 25 people scenes.
# Real camera shake follows a curved path rather than a straight line, so these
# values still need checking against real event photos.
MAX_BLURRED_FALLOFF = 0.3
MIN_FALLOFF_SPREAD = 0.08

# Directions that are measured, in radians, every 15 degrees.
_ANGLES = np.radians(np.arange(0, 180, 15))


@dataclass(frozen=True)
class MotionBlur:
    """Signs that a photo is smeared along one direction, such as from camera shake."""

    """
    Direction of the smear in degrees, counter-clockwise from horizontal as the
    photo is shown on screen: 0 is a sideways smear and 90 an up and down one.
    """
    direction: float

    """
    Whether the shutter was open longer than 1 / focal length (35 mm
    equivalent), the usual limit for a hand-held shot to stay sharp. None when
    the EXIF lacks ExposureTime or FocalLengthIn35mmFilm. It is only a hint:
    stabilisers and tripods allow much slower shutters.
    """
    slow_shutter: bool | None


def compute_motion_blur(photo: Photograph) -> MotionBlur | None:
    """
    Checks whether the whole photo is smeared along one direction, as camera
    shake does, and returns None when it is not. Nothing about the photo is
    changed; the result is only a flag for the user to review.

    Motion blur removes fine detail along one direction only, while defocus
    removes it evenly in every direction. So the photo is measured at a fine
    and a coarse scale, and for each direction it checks how fast detail fades
    between the two. A direction where detail barely fades, because the blur
    already took it, while other directions fade normally, means a smear.

    A moving subject in an otherwise sharp photo is not flagged, and neither is
    an intentional blur such as a panning shot; telling those apart is left to
    the user.
    """
    gray = cv2.cvtColor(photo.downscaled, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fine = _tile_tensors(gray, FINE_SCALE)
    coarse = _tile_tensors(gray, COARSE_SCALE)

    detailed = np.sqrt(coarse[:, 0] + coarse[:, 1]) > MIN_TILE_DETAIL
    if np.count_nonzero(detailed) < MIN_DETAILED_TILES:
        return None

    # Edge energy shrinks like scale^(-2 * falloff), so taking logs of the
    # fine to coarse energy ratio and the scale ratio gives the falloff.
    fine_energy = _directional_energy(fine[detailed].sum(axis=0))
    coarse_energy = _directional_energy(coarse[detailed].sum(axis=0))
    falloff = np.log(fine_energy / coarse_energy) / (2 * math.log(COARSE_SCALE / FINE_SCALE))

    lowest = float(falloff.min())
    if lowest > MAX_BLURRED_FALLOFF or float(falloff.max()) - lowest < MIN_FALLOFF_SPREAD:
        return None

    # Image rows grow downwards, so the angle is flipped to read
    # counter-clockwise on screen.
    angle = math.degrees(float(_ANGLES[int(falloff.argmin())]))
    return MotionBlur(direction=(-angle) % 180, slow_shutter=_is_slow_shutter(photo.exif))


def flag_motion_blurred_photos(photos: list[Photograph]) -> list[tuple[Photograph, MotionBlur]]:
    """
    Returns every photo that `compute_motion_blur` flags, paired with what was
    found, in the order the photos were given. No photo is removed.
    """
    flagged: list[tuple[Photograph, MotionBlur]] = []
    for photo in photos:
        blur = compute_motion_blur(photo)
        if blur is not None:
            flagged.append((photo, blur))
    return flagged


def _tile_tensors(gray: NDArray[np.float32], scale: float) -> NDArray[np.float64]:
    """
    Smooths the image by `scale` and returns, for each tile, the average of
    gx², gy² and gx·gy, where gx and gy are its horizontal and vertical edge
    strengths. These three numbers give the edge strength along any direction.
    """
    smooth = cv2.GaussianBlur(gray, (0, 0), scale)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)

    # Shrinking with INTER_AREA to one pixel per tile averages each tile.
    # The few rows and columns past the last whole tile are left out.
    height, width = gray.shape
    rows, cols = height // GRID * GRID, width // GRID * GRID
    tensors = np.empty((GRID * GRID, 3))
    for index, product in enumerate((gx * gx, gy * gy, gx * gy)):
        tile_means = cv2.resize(product[:rows, :cols], (GRID, GRID), interpolation=cv2.INTER_AREA)
        tensors[:, index] = tile_means.ravel()
    return tensors


def _directional_energy(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Edge strength squared along each of `_ANGLES`, from summed gx², gy² and gx·gy."""
    xx, yy, xy = tensor
    cos, sin = np.cos(_ANGLES), np.sin(_ANGLES)
    return cos * cos * xx + 2 * cos * sin * xy + sin * sin * yy


def _is_slow_shutter(exif: dict[str, Any]) -> bool | None:
    """
    Compares ExposureTime with 1 / FocalLengthIn35mmFilm, or returns None when
    either is missing or unreadable.
    """
    try:
        exposure = float(exif["ExposureTime"])
        focal_length = float(exif["FocalLengthIn35mmFilm"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None

    if exposure <= 0 or focal_length <= 0:
        return None
    return exposure > 1 / focal_length
