"""
Tasks related to correcting the camera clocks of every photographer, so photos
from different cameras can be placed on one shared timeline.
"""

import numpy as np

from collections.abc import Mapping
from datetime import datetime, timedelta
from loguru import logger
from numpy.typing import NDArray

from organy.models import Photograph


# Lowest embedding similarity (see `compute_photo_embedding`) for two photos
# from different photographers to be checked as a possible shot of the same
# moment. Copies and same-scene shots score about 0.9, unrelated photos much
# lower. This still needs checking against real event photos.
MIN_PAIR_SIMILARITY = 0.8

# Widest spread of time differences, in seconds, for candidate pairs to
# agree on one clock offset. Two photographers rarely press the shutter at the
# same instant, so pairs of the same moment differ by up to a minute or two.
AGREEMENT_WINDOW = 120.0

# Candidate pairs that must agree on an offset before it is trusted. Wrong
# pairs (similar scenes at different times) give scattered differences, so
# several pairs agreeing by chance is unlikely.
MIN_AGREEING_PAIRS = 3


def estimate_camera_times(
    photos: Mapping[Photograph, NDArray[np.float32]],
    offsets: Mapping[str, timedelta] | None = None,
) -> dict[Photograph, datetime | None]:
    """
    Estimates when every photo in `photos` (each photo with its embedding, as
    `compute_embeddings` returns) was taken, on the clock of one reference
    photographer, so photos from every photographer can be sorted together.

    `offsets` is how much to add to each photographer's camera times, as
    `estimate_clock_offsets` returns. They are estimated when not given, so
    the user can correct the estimates and pass them back in.

    The times have no time zone, since they follow the reference camera's
    clock rather than a real one. A photo is None when it has no camera time
    (see `Photograph.camera_time`), when its photographer has no offset, or
    when the offset moves it outside the dates Python can hold.
    """
    if offsets is None:
        offsets = estimate_clock_offsets(photos)

    times: dict[Photograph, datetime | None] = {}
    for photo in photos:
        time = _wall_clock(photo)
        offset = offsets.get(photo.photographer)
        if time is None or offset is None:
            times[photo] = None
            continue

        # A camera whose clock was never set may write a date near year 1 or
        # 9999, and moving it by the offset can go past what datetime holds.
        try:
            times[photo] = time + offset
        except OverflowError:
            times[photo] = None

    return times


def estimate_clock_offsets(photos: Mapping[Photograph, NDArray[np.float32]]) -> dict[str, timedelta]:
    """
    Estimates how far each photographer's camera clock is from the clock of a
    reference photographer: the one with the most photos that have a camera
    time. Adding a photographer's offset to their camera times puts them on
    the reference clock, whose own offset is zero.
    - For each two photographers, their most similar photos (by embedding)
      are paired (see `_find_candidate_pairs`).
    - Their clock offset is the one that the most pairs agree on (see
      `_find_agreeing_offset`). Pairs of the same moment agree, while pairs of
      similar scenes at different times scatter.
    - Photographers are linked to the reference through the photographers they
      share the most agreeing pairs with.

    Pairs are not checked one by one with ORB matches: photographers stand in
    different places in front of a scene with moving people, which a
    homography cannot fit, so almost no real pairs passed on event photos.

    A photographer that shares too few agreeing pairs with the others (see
    `MIN_AGREEING_PAIRS`) gets no offset, since there is nothing to compare
    their clock with. They are left out and logged so the user can set one.
    """
    by_photographer: dict[str, list[Photograph]] = {}
    for photo in photos:
        if _wall_clock(photo) is None: continue
        by_photographer.setdefault(photo.photographer, []).append(photo)

    if not by_photographer: return {}

    # (a, b) -> (a's time minus b's time that the pairs agree on, number of agreeing pairs)
    links: dict[tuple[str, str], tuple[timedelta, int]] = {}
    names = sorted(by_photographer)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            candidates = _find_candidate_pairs(by_photographer[a], by_photographer[b], photos)

            differences: list[float] = []
            for photo_a, photo_b in candidates:
                time_a, time_b = _wall_clock(photo_a), _wall_clock(photo_b)
                assert time_a is not None and time_b is not None
                differences.append((time_a - time_b).total_seconds())

            agreement = _find_agreeing_offset(differences)
            if agreement is None:
                logger.debug("no offset agreed by {} candidate pair(s) between {} and {}", len(differences), a, b)
                continue

            offset, support = agreement
            logger.debug(
                "{} of {} candidate pair(s) between {} and {} agree on {:+.1f} s",
                support, len(differences), a, b, offset,
            )
            links[(a, b)] = (timedelta(seconds=offset), support)

    # The reference is the photographer with the most timed photos, the name
    # breaking ties so the same photos always give the same result.
    reference = min(names, key=lambda name: (-len(by_photographer[name]), name))
    offsets = {reference: timedelta(0)}

    # Grow the set of linked photographers one at a time, always through the
    # link with the most agreeing pairs, so each offset comes from the best
    # supported chain back to the reference (a maximum spanning tree).
    while True:
        best: tuple[int, str, timedelta] | None = None
        for (a, b), (difference, support) in links.items():
            if a in offsets and b not in offsets:
                candidate = (support, b, offsets[a] + difference)
            elif b in offsets and a not in offsets:
                candidate = (support, a, offsets[b] - difference)
            else:
                continue

            if best is None or candidate[0] > best[0]: best = candidate

        if best is None: break
        _, name, offset = best
        offsets[name] = offset

    logger.info("estimated clock offsets of {} photographer(s) against {}", len(offsets), reference)
    for name in names:
        if name in offsets:
            logger.debug("clock offset of {}: {:+.1f} s", name, offsets[name].total_seconds())
        else:
            logger.warning("cannot estimate clock offset of {}, too few shared scenes", name)

    return offsets


def _find_candidate_pairs(
    photos_a: list[Photograph],
    photos_b: list[Photograph],
    embeddings: Mapping[Photograph, NDArray[np.float32]],
) -> list[tuple[Photograph, Photograph]]:
    """
    Pairs photos of one photographer with photos of another that are each
    other's most similar photo (by embedding), at least `MIN_PAIR_SIMILARITY`.

    Both photos must be each other's best match, so one photo is never paired
    with many photos of a burst from the other photographer.
    """
    # Embeddings have a length of 1, so this table holds the similarity of
    # every photo in `photos_a` (rows) with every photo in `photos_b` (columns).
    similarity = np.vstack([embeddings[p] for p in photos_a]) @ np.vstack([embeddings[p] for p in photos_b]).T

    best_b = similarity.argmax(axis=1)
    best_a = similarity.argmax(axis=0)

    return [
        (photos_a[i], photos_b[j])
        for i, j in enumerate(best_b)
        if best_a[j] == i and similarity[i, j] >= MIN_PAIR_SIMILARITY
    ]


def _find_agreeing_offset(differences: list[float]) -> tuple[float, int] | None:
    """
    Finds the time difference (in seconds) that the most candidate pairs agree
    on: the most pairs whose differences fit within `AGREEMENT_WINDOW`. It
    returns the median difference of those pairs and how many there are.

    It returns None when fewer than `MIN_AGREEING_PAIRS` agree, or when
    another, separate group of pairs agrees just as well, since there is then
    no telling which group is right.
    """
    values = np.sort(np.asarray(differences, dtype=np.float64))

    # For each pair (in order of difference), `ends[i]` is one past the last
    # pair within the window that starts at it, so `ends[i] - i` pairs agree.
    ends = np.searchsorted(values, values + AGREEMENT_WINDOW, side="right")
    counts = ends - np.arange(len(values))
    if len(values) == 0 or counts.max() < MIN_AGREEING_PAIRS: return None

    start = int(counts.argmax())
    end = int(ends[start])

    # Another window as full as the best one, not overlapping it, makes the
    # offset ambiguous.
    low, high = values[start] - AGREEMENT_WINDOW, values[end - 1] + AGREEMENT_WINDOW
    for other in np.flatnonzero(counts == counts[start]):
        if values[other] > high or values[ends[other] - 1] < low: return None

    return float(np.median(values[start:end])), end - start


def _wall_clock(photo: Photograph) -> datetime | None:
    """
    The photo's camera time as the camera's clock showed it, without its time
    zone. Some cameras save a time zone and others do not, and naive and
    aware times cannot be subtracted from each other. Any time zone difference
    between two cameras ends up in their clock offset instead.
    """
    return None if photo.camera_time is None else photo.camera_time.replace(tzinfo=None)
