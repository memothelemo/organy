"""
Collects photographs from the files and folders a user selected.
"""

import cv2
import hashlib
import numpy as np

from collections.abc import Iterable
from numpy.typing import NDArray
from pathlib import Path

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Longest side, in pixels, of a downscaled photograph.
MAX_SIDE = 1000


def downscale(image: NDArray) -> NDArray:
    """
    Shrinks an image so its longer side is at most MAX_SIDE pixels while
    keeping its aspect ratio. Smaller images are returned as they are, since
    upscaling only adds blur.
    """
    height, width = image.shape[:2]
    scale = MAX_SIDE / max(height, width)
    if scale >= 1:
        return image

    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    # INTER_AREA averages neighbouring pixels, which avoids aliasing when shrinking.
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


class Photograph:
    """
    Imported photograph where it contains the raw downscaled image 
    data, attributions, EXIF and SHA1 hash of the original image.
    """

    # Downscaled version of the original photograph.
    downscaled: NDArray

    # SHA1 hash of the original photograph.
    sha1: str

    # Original path of the imported photograph
    path: Path

    def __init__(self, path: Path, binary: NDArray) -> None:
        self.path = path
        self.sha1 = hashlib.sha1(binary.flatten()).hexdigest()
        self.downscaled = downscale(binary)


    def __str__(self) -> str:
        return f"Photograph({self.path}, {self.sha1[0:5]})"


def is_maybe_photo(path: Path) -> bool:
    """
    Checks whether a path looks like a supported photo (JPEG or PNG).

    Names starting with a dot are skipped because macOS leaves "._DSC_0001.jpg"
    metadata files next to actual photos.
    """
    # TODO: Add an actual is_photo function where it checks the file contents.
    return path.suffix.lower() in PHOTO_SUFFIXES and not path.name.startswith(".")


def load_photos(photos: list[Path]) -> list[Photograph]:
    """
    Loads all photo files passed from the photos parameter.
    """
    results: list[Photograph] = list()
    for path in photos:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR_BGR)
        if image is None:
            raise FileNotFoundError(f"no such file: {path}")

        cached = Photograph(path, image)
        results.append(cached)

    return results


def collect_photos(sources: Iterable[Path]) -> list[Path]:
    """
    Expands selected files and folders into one sorted list of possible photo files.

    Folders are searched recursively because cameras typically nest photos in
    subfolders (e.g. DCIM/100NIKON).
    """
    photos: set[Path] = set()
    for source in sources:
        if source.is_dir():
            # It uses glob pattern to conveniently walk to its descending entries.
            candidates = (path for path in source.rglob("*") if path.is_file())
        elif source.is_file():
            candidates = [source]
        else:
            raise FileNotFoundError(f"no such file or folder: {source}")

        # Transform its relative paths into absolute paths.
        photos.update(path.resolve() for path in candidates if is_maybe_photo(path))

    return sorted(photos)
