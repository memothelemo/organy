"""
Collects photographs from the files and folders a user selected.
"""

from collections.abc import Iterable
from pathlib import Path

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}


def is_maybe_photo(path: Path) -> bool:
    """
    Checks whether a path looks like a supported photo (JPEG or PNG).

    Names starting with a dot are skipped because macOS leaves "._DSC_0001.jpg"
    metadata files next to actual photos.
    """
    # TODO: Add an actual is_photo function where it checks the file contents.
    return path.suffix.lower() in PHOTO_SUFFIXES and not path.name.startswith(".")


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
