"""
Collect photographs from the files and folders a user selected.

Only paths are gathered here; no image is opened, so even thousands of
selections stay cheap. Originals are only read, never modified.
"""

from collections.abc import Iterable
from pathlib import Path

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}


def is_photo(path: Path) -> bool:
    """Tell whether a path looks like a supported photo (JPEG or PNG).

    Names starting with a dot are skipped: macOS leaves "._DSC_0001.jpg"
    metadata files next to real photos on shared drives.
    """
    return path.suffix.lower() in PHOTO_SUFFIXES and not path.name.startswith(".")


def collect_photos(sources: Iterable[Path]) -> list[Path]:
    """Expand selected files and folders into one sorted list of photo paths.

    Folders are searched recursively because cameras nest photos in
    subfolders (e.g. DCIM/100NIKON). A photo selected twice, directly and
    through its folder, is listed once.
    """
    photos: set[Path] = set()
    for source in sources:
        if source.is_dir():
            candidates = (path for path in source.rglob("*") if path.is_file())
        elif source.is_file():
            candidates = [source]
        else:
            raise FileNotFoundError(f"no such file or folder: {source}")
        photos.update(path.resolve() for path in candidates if is_photo(path))
    return sorted(photos)
