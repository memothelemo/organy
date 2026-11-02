import cv2
import numpy as np

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from organy.models import Photograph
from organy.utils.os import read_bytes

# Supported image formats by Organy
SUPPORTED_PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_photos(paths: Iterable[Path], workers=1) -> list[Photograph]:
    """
    Loads all photographs from the given paths, in the same order.

    Threads are used instead of processes because OpenCV releases the GIL
    while decoding and resizing, so threads already run in parallel without
    the start-up and memory cost of extra processes (measured: ~150 ms per
    photo sequentially, ~50 ms with 4 threads, the same as 4 processes).

    `workers` is how many photos are decoded at the same time. The default of
    1 loads them one by one. Each full-size 24 MP photo takes about 72 MB of
    memory while it is being decoded, so peak memory grows with `workers`
    (about 300 MB at 4). Speed stops improving much past 4 on a typical
    laptop. It must be at least 1, or ThreadPoolExecutor raises ValueError.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_load_photo, paths))


def is_maybe_photo(path: Path) -> bool:
    """
    Checks whether a path looks like a supported photo (JPEG or PNG).

    Names starting with a dot are skipped because macOS leaves "._DSC_0001.jpg"
    metadata files next to actual photos.
    """
    return path.suffix.lower() in SUPPORTED_PHOTO_SUFFIXES and not path.name.startswith(".")


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


def _load_photo(path: Path) -> Photograph:
    """Decodes one photograph per path"""
    image = cv2.imdecode(read_bytes(path), cv2.IMREAD_COLOR_BGR)
    if image is None:
        raise ValueError(f"cannot decode photo: {path}")

    return Photograph(image, path)
