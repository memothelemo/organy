import cv2
import numpy as np

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import ExifTags, Image
from typing import Any

from organy.models import Photograph
from organy.utils.os import read_bytes

# Supported image formats by Organy
SUPPORTED_PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_photos(photos: Mapping[Path, str], workers=1) -> list[Photograph]:
    """
    Loads every photo in `photos`, which pairs each photo's path with its
    photographer (this is what `collect_photos` returns). The loaded photos
    come back in the same order.

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
        return list(pool.map(_load_photo, photos.keys(), photos.values()))


def is_maybe_photo(path: Path) -> bool:
    """
    Checks whether a path looks like a supported photo (JPEG or PNG).

    Names starting with a dot are skipped because macOS leaves "._DSC_0001.jpg"
    metadata files next to actual photos.
    """
    return path.suffix.lower() in SUPPORTED_PHOTO_SUFFIXES and not path.name.startswith(".")


def collect_photos(sources: Iterable[Path]) -> dict[Path, str]:
    """
    Finds the photos inside the selected files and folders and works out who
    took each one. The result is sorted by path and pairs each photo with its
    photographer.

    The photographer is the name of the folder the user selected, so
    importing "Ana/" credits every photo inside it to Ana. A file selected on
    its own is credited to the folder it is in. If the same photo is selected twice,
    for example through its folder and on its own, it keeps the photographer
    from the first selection.

    Folders are searched recursively because cameras typically nest photos in
    subfolders (e.g. DCIM/100NIKON).
    """
    photos: dict[Path, str] = {}
    for source in sources:
        if source.is_dir():
            # It uses glob pattern to conveniently walk to its descending entries.
            candidates = (path for path in source.rglob("*") if path.is_file())
        elif source.is_file():
            candidates = [source]
        else:
            raise FileNotFoundError(f"no such file or folder: {source}")

        # resolve() turns a relative path such as "." into the real folder, so it
        # has a name. A drive root such as "D:\" has no name, so its full path is used.
        folder = source.resolve() if source.is_dir() else source.resolve().parent
        photographer = folder.name or str(folder)

        # Transform its relative paths into absolute paths.
        for path in candidates:
            if is_maybe_photo(path):
                photos.setdefault(path.resolve(), photographer)

    return dict(sorted(photos.items()))


def _load_photo(path: Path, photographer: str) -> Photograph:
    """Decodes one photograph per path"""
    image = cv2.imdecode(read_bytes(path), cv2.IMREAD_COLOR_BGR)
    if image is None:
        raise ValueError(f"cannot decode photo: {path}")

    # Opening the file again is cheaper than reusing the bytes from read_bytes.
    #
    # Pillow reads only the ~6 KB header here, while io.BytesIO(data) would copy
    # the whole file (about 10 MB for a 24 MP JPEG) just to read that header.
    return Photograph(image, path, photographer, _read_exif(path))


def _read_exif(path: Path) -> dict[str, Any]:
    """
    Reads the camera details saved inside a photo file (its EXIF) and returns
    them by name. It uses Pillow because OpenCV cannot read EXIF.

    Image.open reads only the start of the file, not the image itself, so this
    is quick. EXIF keeps its details in two places: general ones such as Make,
    Model and Artist, and shooting details such as DateTimeOriginal and
    ExposureTime.
    
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
