"""Quick and dirty reusable utility image related functions."""

import cv2

from datetime import datetime, timedelta, timezone
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
    its own format for them. A photo without EXIF, or one whose file cannot be
    read or parsed, gives an empty dictionary.
    """
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            tags = {**exif, **exif.get_ifd(ExifTags.IFD.Exif)}
    except Exception:
        # Maybe the image is invalid, so we return an empty EXIF dictionary early
        # besides UnidentifiedImageError is for files that are not images.
        return {}

    return {
        ExifTags.TAGS.get(tag, str(tag)): value
        for tag, value in tags.items()
        if not isinstance(value, bytes)
    }


def read_capture_time(exif: dict[str, Any]) -> datetime | None:
    """
    Reads when the camera says a photo was taken from its EXIF (as returned by `read_exif`):
    - "DateTimeOriginal" gives the date and time, such as "2026:11:02 14:05:09"
    - "SubsecTimeOriginal" adds fractions of a second, such as "35" for 0.35 s
    - "OffsetTimeOriginal" adds the time zone, such as "+08:00"

    Without a time zone, the returned datetime has none either. It returns
    None when the photo has no date and time, or a blank or invalid one such
    as "0000:00:00 00:00:00", which some cameras write when their clock was
    never set.
    """

    # EXIF strings are often padded with spaces or null characters.
    def text(name: str) -> str:
        value = exif.get(name)
        return value.strip(" \x00") if isinstance(value, str) else ""

    try:
        captured = datetime.strptime(text("DateTimeOriginal"), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None

    subsec = text("SubsecTimeOriginal")
    if subsec.isdigit():
        captured += timedelta(seconds=int(subsec) / 10 ** len(subsec))

    try:
        offset = datetime.strptime(text("OffsetTimeOriginal"), "%z").tzinfo
    except ValueError:
        offset = None

    return captured.replace(tzinfo=offset) if isinstance(offset, timezone) else captured


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
