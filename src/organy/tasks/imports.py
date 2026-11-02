"""Importing photos related tasks"""

import cv2
import threading
import time

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from pathlib import Path
from PIL import ExifTags
from typing import Any

from organy.models import Photograph
from organy.utils.os import read_bytes
from organy.utils.image import is_maybe_photo, read_exif


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
        return list(pool.map(_load_photo_in_thread, photos.keys(), photos.values()))


def resolve_photographer(path: Path, exif: dict[str, Any] | None = None) -> str | None:
    """
    Finds the name of the photographer who took a photo.

    The Artist field in the photo's EXIF is used first. If it is missing, the
    name of the folder that holds the photo is used instead, since the user
    usually keeps each photographer's photos in their own folder. It returns
    None when neither gives a name.

    `exif` is the photo's EXIF as returned by `read_exif`. Pass it when it
    has already been read, so the file is not opened again. Otherwise the
    EXIF is read from `path`.
    """
    if exif is None:
        # TODO: Cache EXIF metadata correlating to the path
        exif = read_exif(path)

    # `read_exif` keys the EXIF by tag name, so the tag number has to be turned into its name first.
    artist = exif.get(ExifTags.TAGS[ExifTags.Base.Artist])
    if isinstance(artist, str) and artist.strip():
        return artist.strip()

    # A bare file name like "a.jpg" has no folder name to fall back on.
    return path.parent.name or None


def collect_photos(sources: Iterable[Path]) -> dict[Path, str]:
    """
    Finds every possible photo in the files and folders the user selected.

    Each source can be a single file or a folder. Folders are searched
    recursively, including all of their subfolders. Only paths that look like
    supported photos (JPEG or PNG, judged by their extension) are kept; see
    `is_maybe_photo`. The files themselves are not opened or read here.

    It returns a dictionary that maps each photo's absolute path to its
    photographer. The photographer is "<unknown>" for now. A photo found
    through more than one source is listed only once, and the dictionary is
    sorted by path.

    Raises `FileNotFoundError` if any of the sources does not exist.
    """
    photos: dict[Path, str] = {}

    for source in sources:
        last_photo_count = len(photos)

        if source.is_dir():
            # It uses glob pattern to conveniently walk to its descending entries.
            candidates = (path for path in source.rglob("*") if path.is_file())
        elif source.is_file():
            candidates = [source]
        else:
            raise FileNotFoundError(f"no such file or folder: {source}")

        # Transform its relative paths into absolute paths.
        for path in candidates:
            if is_maybe_photo(path):
                photos.setdefault(path.resolve(), "<unknown>")

        maybe_photos = len(photos) - last_photo_count
        logger.debug("found {} candidates at {}", maybe_photos, source)

    # Sort the entire dictionary by photographer
    return dict(sorted(photos.items()))


def _load_photo_in_thread(path: Path, photographer: str) -> Photograph:
    """
    Loads one photo like `_load_photo`, tagging its log lines with the ID of
    the worker thread that loads it, so lines from parallel loads can be told
    apart.
    """
    # Pool threads don't inherit the caller's context, so I need to set it
    # inside each task rather than around pool.map.
    with logger.contextualize(thread_id=threading.get_native_id()):
        """Decodes one photograph per path"""
        start = time.perf_counter()
        image = cv2.imdecode(read_bytes(path), cv2.IMREAD_COLOR_BGR)
        if image is None: raise ValueError(f"cannot decode photo: {path}")

        # Opening the file again is cheaper than reusing the bytes from read_bytes.
        #
        # Pillow reads only the ~6 KB header here, while io.BytesIO(data) would copy
        # the whole file (about 10 MB for a 24 MP JPEG) just to read that header.
        photograph = Photograph(image, path, photographer, read_exif(path))
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.trace(
            "loaded picture in {:.1f} ms (phash={})",
            elapsed_ms,
            photograph.phash.tobytes().hex()
        )

        return photograph
