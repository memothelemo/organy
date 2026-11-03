"""
Tasks related to correcting the camera clocks of every photographer, so photos
from different cameras can be placed on one shared timeline.
"""

import cv2
import numpy as np

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from loguru import logger
from numpy.typing import NDArray

from organy.models import Photograph


# Lowest embedding similarity (see `compute_photo_embedding`) for two photos
# from different photographers to be checked as a possible shot of the same
# moment. Copies and same-scene shots score about 0.9, unrelated photos much
# lower. This still needs checking against real event photos.
MIN_PAIR_SIMILARITY = 0.8

# Most candidate pairs checked with ORB for each two photographers. Only the
# most similar pairs are checked, since each check costs a few tens of ms.
MAX_PAIRS_PER_PHOTOGRAPHERS = 20

# ORB and matching settings used to confirm that two photos show the same scene:
# - ORB_FEATURES: corners found in each downscaled photo
# - MATCH_RATIO: Lowe's ratio test, a match is kept only when it is clearly
#                better than the second best one
# - RANSAC_THRESHOLD: how far (in pixels of the downscaled photo) a matched
#                     corner may land from where the homography puts it
# - MIN_INLIERS: matches that must agree with one homography for the pair
#                to count as the same scene
ORB_FEATURES = 1_000
MATCH_RATIO = 0.75
RANSAC_THRESHOLD = 5.0
MIN_INLIERS = 20

# Confirmed pairs that two photographers need before their clock offset is
# trusted. A single pair could be two shots of the same spot at different
# times, and the median of a few pairs leaves such outliers out.
MIN_CONFIRMED_PAIRS = 3


# Corners found by ORB in a photo, and their descriptors (None when no corner was found).
Features = tuple[Sequence[cv2.KeyPoint], NDArray[np.uint8] | None]


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
    (see `Photograph.camera_time`), or when its photographer has no offset.
    """
    if offsets is None:
        offsets = estimate_clock_offsets(photos)

    times: dict[Photograph, datetime | None] = {}
    for photo in photos:
        time = _wall_clock(photo)
        offset = offsets.get(photo.photographer)
        times[photo] = None if time is None or offset is None else time + offset

    return times


def estimate_clock_offsets(photos: Mapping[Photograph, NDArray[np.float32]]) -> dict[str, timedelta]:
    """
    Estimates how far each photographer's camera clock is from the clock of a
    reference photographer: the one with the most photos that have a camera
    time. Adding a photographer's offset to their camera times puts them on
    the reference clock, whose own offset is zero.
    - For each two photographers, their most similar photos (by embedding) are
      paired, and each pair is confirmed with ORB matches and a RANSAC homography.
    - The median time difference of the confirmed pairs is their clock offset.
    - Photographers are linked to the reference through the photographers they
      share the most confirmed pairs with.

    A photographer that shares too few confirmed pairs with the others (see
    `MIN_CONFIRMED_PAIRS`) gets no offset, since there is nothing to compare
    their clock with. They are left out and logged so the user can set one.
    """
    by_photographer: dict[str, list[Photograph]] = {}
    for photo in photos:
        if _wall_clock(photo) is None: continue
        by_photographer.setdefault(photo.photographer, []).append(photo)

    if not by_photographer: return {}

    # Each photo is decoded and run through ORB at most once, however many
    # candidate pairs it is in.
    features: dict[Photograph, Features] = {}
    orb = cv2.ORB.create(nfeatures=ORB_FEATURES)
    matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING)

    def features_of(photo: Photograph) -> Features:
        if photo not in features:
            gray = cv2.cvtColor(photo.decode_downscaled(), cv2.COLOR_BGR2GRAY)
            features[photo] = orb.detectAndCompute(gray, None)
        return features[photo]

    # (a, b) -> (median of a's time minus b's time, number of confirmed pairs)
    links: dict[tuple[str, str], tuple[timedelta, int]] = {}
    names = sorted(by_photographer)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            candidates = _find_candidate_pairs(by_photographer[a], by_photographer[b], photos)

            differences: list[float] = []
            for photo_a, photo_b in candidates:
                inliers = _count_inliers(features_of(photo_a), features_of(photo_b), matcher)
                if inliers < MIN_INLIERS: continue

                time_a, time_b = _wall_clock(photo_a), _wall_clock(photo_b)
                assert time_a is not None and time_b is not None
                differences.append((time_a - time_b).total_seconds())

            logger.debug(
                "confirmed {} of {} candidate pair(s) between {} and {}",
                len(differences), len(candidates), a, b,
            )
            if len(differences) < MIN_CONFIRMED_PAIRS: continue

            links[(a, b)] = (timedelta(seconds=float(np.median(differences))), len(differences))

    # The reference is the photographer with the most timed photos, the name
    # breaking ties so the same photos always give the same result.
    reference = min(names, key=lambda name: (-len(by_photographer[name]), name))
    offsets = {reference: timedelta(0)}

    # Grow the set of linked photographers one at a time, always through the
    # link with the most confirmed pairs, so each offset comes from the best
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
    Only the `MAX_PAIRS_PER_PHOTOGRAPHERS` most similar pairs are returned,
    most similar first.

    Both photos must be each other's best match, so one photo is never paired
    with many photos of a burst from the other photographer.
    """
    # Embeddings have a length of 1, so this table holds the similarity of
    # every photo in `photos_a` (rows) with every photo in `photos_b` (columns).
    similarity = np.vstack([embeddings[p] for p in photos_a]) @ np.vstack([embeddings[p] for p in photos_b]).T

    best_b = similarity.argmax(axis=1)
    best_a = similarity.argmax(axis=0)

    pairs = [
        (float(similarity[i, j]), i, int(j))
        for i, j in enumerate(best_b)
        if best_a[j] == i and similarity[i, j] >= MIN_PAIR_SIMILARITY
    ]
    pairs.sort(reverse=True)
    return [(photos_a[i], photos_b[j]) for _, i, j in pairs[:MAX_PAIRS_PER_PHOTOGRAPHERS]]


def _count_inliers(a: Features, b: Features, matcher: cv2.BFMatcher) -> int:
    """
    Counts the ORB matches between two photos that agree with one homography
    found by RANSAC. It is 0 when there are too few matches to find one.
    """
    (keypoints_a, descriptors_a), (keypoints_b, descriptors_b) = a, b
    if descriptors_a is None or descriptors_b is None: return 0

    matches = [
        pair[0]
        for pair in matcher.knnMatch(descriptors_a, descriptors_b, k=2)
        if len(pair) == 2 and pair[0].distance < MATCH_RATIO * pair[1].distance
    ]

    # A homography needs at least 4 matches, and fewer than MIN_INLIERS could
    # never pass anyway.
    if len(matches) < max(4, MIN_INLIERS): return 0

    points_a = np.array([keypoints_a[m.queryIdx].pt for m in matches], dtype=np.float32)
    points_b = np.array([keypoints_b[m.trainIdx].pt for m in matches], dtype=np.float32)
    _, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, RANSAC_THRESHOLD)
    return 0 if mask is None else int(np.count_nonzero(mask))


def _wall_clock(photo: Photograph) -> datetime | None:
    """
    The photo's camera time as the camera's clock showed it, without its time
    zone. Some cameras save a time zone and others do not, and naive and
    aware times cannot be subtracted from each other. Any time zone difference
    between two cameras ends up in their clock offset instead.
    """
    return None if photo.camera_time is None else photo.camera_time.replace(tzinfo=None)
