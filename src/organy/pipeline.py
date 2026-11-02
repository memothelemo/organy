"""
Organy's processing stages, run one after another on the photos the user
selected. Each stage does its work and logs a short summary of it.
"""

import time

from collections.abc import Iterable
from loguru import logger
from pathlib import Path

from organy.models import Photograph
from organy.tasks import collect_photos, get_duplicated_photos, imports


def import_photos(sources: Iterable[Path]) -> dict[Path, str]:
    """
    Stage 1: Finds every photo in `sources` and works out who took each one.

    It returns a dictionary that maps each photo's absolute path to its
    photographer, sorted by path. A photo whose photographer cannot be found
    keeps "<unknown>".
    """
    sources = list(sources)
    photos = collect_photos(sources)
    logger.info("found {} photo(s) from {} source(s)", len(photos), len(sources))

    resolved = 0
    for photo in photos.keys():
        photographer = imports.resolve_photographer(photo)
        if photographer is None:
            logger.warning("cannot resolve photographer for {}", photo)
            continue

        photos[photo] = photographer
        resolved += 1

    logger.info("resolved photographer for {} of {} photos", resolved, len(photos))
    return photos


def load_photos(photos: dict[Path, str], workers: int = 1) -> list[Photograph]:
    """
    Stage 2: Decodes every photo and makes its downscaled copy and fingerprint
    (the `Photograph` class does both while loading).

    `photos` is what `import_photos` returns. `workers` is how many photos
    are loaded at the same time; see `organy.tasks.imports.load_photos`.
    """
    start = time.perf_counter()
    loaded = imports.load_photos(photos, workers=workers)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info("loaded {} photo(s) (took {:.1f} ms)", len(loaded), elapsed_ms)
    return loaded


def remove_duplicates(photos: list[Photograph]) -> tuple[list[Photograph], list[list[Photograph]]]:
    """
    Stage 3: Drops photos that are copies of another photo, so later stages
    evaluate each picture once.
    - The first photo of each group of copies is kept, the rest are dropped.
    - Photos without copies are kept.
    - Only the returned list changes. The files stay as they are.

    It returns both the deduplicated photos and the duplicated photos useful
    for resolving conflicts soon.
    """
    start = time.perf_counter()
    duplicated_photos = get_duplicated_photos(photos)

    copies = {photo.path for group in duplicated_photos for photo in group[1:]}
    kept = [photo for photo in photos if photo.path not in copies]
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "removed {} duplicated photo(s), {} of {} left (took {:.1f} ms)",
        len(photos) - len(kept),
        len(kept),
        len(photos),
        elapsed_ms,
    )
    return kept, duplicated_photos

