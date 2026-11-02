"""Quick and dirty reusable utility image related functions."""

import cv2

from numpy.typing import NDArray
from pathlib import Path
from PIL import Image, ExifTags
from typing import Any


def read_exif(path: Path) -> dict[str, Any]:
    """
    Reads the camera details saved inside a photo file (its EXIF) and returns
    them by name. It uses Pillow because OpenCV cannot read EXIF.

    Both are combined into one dictionary. Details stored as raw bytes, such as MakerNote,
    are left out because they can be tens of kilobytes and each camera maker uses
    its own format for them. A photo without EXIF gives an empty dictionary.
    """
    with Image.open(path) as image:
        exif = image.getexif()
        tags = {**exif, **exif.get_ifd(ExifTags.IFD.Exif)}

    return {
        ExifTags.TAGS.get(tag, str(tag)): value
        for tag, value in tags.items()
        if not isinstance(value, bytes)
    }


# Supported image formats by Organy
SUPPORTED_PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}


def is_maybe_photo(path: Path) -> bool:
    """
    Checks whether a path looks like a supported photo (JPEG or PNG).

    Names starting with a dot are skipped because macOS leaves "._DSC_0001.jpg"
    metadata files next to actual photos.
    """
    return path.suffix.lower() in SUPPORTED_PHOTO_SUFFIXES and not path.name.startswith(".")


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
