"""
Organy's processing stages, run one after another on the photos the user
selected. Each stage does its work and logs a short summary of it.
"""

import time

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from loguru import logger
from numpy.typing import NDArray
from pathlib import Path

from organy.models import Photograph
from organy.tasks import (
    collect_photos, embedding, events, faces, get_duplicated_photos, imports, selection, stacks, tagging,
    time_adjustment,
)


def compute_embeddings(
    photos: list[Photograph],
    model_path: Path,
    workers: int = 1
) -> dict[Photograph, NDArray]:
    """
    Stage 4.1.: Compute every photograph's own embedding to check for its
    similarity later in 4.2. This is like phash but it is more accurate representation
    of a fingerprint of an image, to try and minimize the duplicates as possible.
    """
    start = time.perf_counter()
    embeddings = embedding.compute_embeddings(photos, model_path, workers=workers)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info("computed {} embedding(s) (took {:.1f} ms)", len(embeddings), elapsed_ms)
    return embeddings


def estimate_camera_times(
    embeddings: dict[Photograph, NDArray],
    offsets: Mapping[str, timedelta] | None = None,
) -> dict[Photograph, datetime | None]:
    """
    Stage 4.3.: Corrects every photographer's camera clock against one
    reference photographer, so photos from every camera share one timeline.
    The clock offsets are estimated from photos of the same moment unless the
    user gives them; see `organy.tasks.time_adjustment.estimate_camera_times`.
    """
    start = time.perf_counter()
    times = time_adjustment.estimate_camera_times(embeddings, offsets)
    elapsed_ms = (time.perf_counter() - start) * 1000

    timed = sum(1 for value in times.values() if value is not None)
    logger.info("estimated time of {} of {} photo(s) (took {:.1f} ms)", timed, len(times), elapsed_ms)
    return times



def split_events(
    times: Mapping[Photograph, datetime | None],
    gap: timedelta = events.DEFAULT_EVENT_GAP,
) -> tuple[list[list[Photograph]], list[Photograph]]:
    """
    Stage 4.4.: Splits the shared timeline into events wherever photos are
    more than `gap` apart. Events are never merged automatically. Photos
    without a time are returned apart; see `organy.tasks.events.split_events`.
    """
    found, untimed = events.split_events(times, gap)
    logger.info("split {} photo(s) into {} event(s)", sum(len(event) for event in found), len(found))
    if untimed:
        logger.warning("{} photo(s) have no shared time and belong to no event", len(untimed))

    return found, untimed

def find_stacks(
    photo_events: list[list[Photograph]],
    embeddings: Mapping[Photograph, NDArray],
    times: Mapping[Photograph, datetime | None] | None = None,
    face_models: tuple[Path, Path] | None = None,
) -> list[list[list[Photograph]]]:
    """
    Stage 4.5.: Groups the photos of each event into stacks of the same scene
    or moment, and returns the stacks of each event in the same order. Stacks
    never cross events, so a photo's event never changes.

    `face_models` is the YuNet and SFace model files, used to stack close calls
    showing the same person. Faces are not checked without them. See
    `organy.tasks.stacks.find_stacks` for the rest.
    """
    start = time.perf_counter()
    recognizer = None if face_models is None else faces.FaceRecognizer(*face_models)
    found = [stacks.find_stacks(event, embeddings, times, recognizer) for event in photo_events]
    elapsed_ms = (time.perf_counter() - start) * 1000

    count = sum(len(event) for event in found)
    stacked = sum(1 for event in found for stack in event if len(stack) > 1)
    logger.info(
        "found {} stack(s) in {} event(s), {} of them with more than one photo (took {:.1f} ms)",
        count, len(photo_events), stacked, elapsed_ms,
    )
    return found


def tag_photos(
    embeddings: Mapping[Photograph, NDArray],
    text_model_path: Path,
    vocab_path: Path,
    merges_path: Path,
    tags: Mapping[str, Sequence[str]] = tagging.DEFAULT_TAGS,
) -> dict[Photograph, list[tuple[str, float]]]:
    """
    Stage 4.6.: Tags every photo with what it shows, by comparing its
    embedding with the embeddings of each tag's prompts (`tags`, see
    `organy.tasks.tagging.DEFAULT_TAGS`) and of a few neutral captions. The
    prompts are embedded once with the CLIP text model and its tokenizer files
    ("vocab.json" and "merges.txt"); see `organy.tasks.tagging.tag_photos`.
    """
    start = time.perf_counter()
    tokenizer = tagging.ClipTokenizer(vocab_path, merges_path)
    model = tagging.load_text_model(text_model_path)
    prompts = tagging.embed_prompts(model, tokenizer, tags)
    neutral = tagging.embed_texts(model, tokenizer, tagging.NEUTRAL_PROMPTS)
    tagged = tagging.tag_photos(embeddings, prompts, neutral)
    elapsed_ms = (time.perf_counter() - start) * 1000

    count = sum(1 for photo_tags in tagged.values() if photo_tags)
    logger.info("tagged {} of {} photo(s) (took {:.1f} ms)", count, len(tagged), elapsed_ms)
    return tagged


def suggest_best_photos(
    photo_stacks: Iterable[list[Photograph]],
) -> list[tuple[Photograph, dict[Photograph, selection.PhotoQuality]]]:
    """
    Stage 4.7.: Suggests the best photo of every stack by sharpness and
    exposure, along with every photo's raw measurements so the user can pick
    another one. See `organy.tasks.selection.suggest_best_photo`.
    """
    start = time.perf_counter()
    suggestions = [selection.suggest_best_photo(stack) for stack in photo_stacks]
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info("suggested the best photo of {} stack(s) (took {:.1f} ms)", len(suggestions), elapsed_ms)
    return suggestions


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

