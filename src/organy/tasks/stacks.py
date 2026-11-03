"""
Tasks related to grouping photos of the same scene or moment into stacks,
inside one event.
"""

import numpy as np
import time

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from loguru import logger
from numpy.typing import NDArray

from organy.models import Photograph
from organy.tasks.faces import FaceRecognizer, share_face
from organy.tasks.matching import SceneMatcher


# Embedding similarities (see `compute_photo_embedding`) that decide whether
# two photos belong in the same stack:
# - STACK_SIMILARITY or more: stacked without any other check
# - MIN_CLOSE_CALL_SIMILARITY up to STACK_SIMILARITY: a close call, stacked
#   only when another check agrees (see `find_stacks`)
# - less: never stacked
#
# On the 49 stock photos (one music trade show, one camera):
# - Shots of the same performance scored 0.78 to 0.92, mostly 0.84 to 0.92.
# - Different scenes scored up to 0.87, but the ones above 0.78 were taken
#   on other days, more than MAX_MOMENT_GAP apart, or of the same band.
#
# These numbers still need checking against real event photos.
STACK_SIMILARITY = 0.88
MIN_CLOSE_CALL_SIMILARITY = 0.78

# Longest time between two photos of a close call for them to count as the
# same moment. On the 49 stock photos, the shots of one song spanned up to
# about 3 minutes.
MAX_MOMENT_GAP = timedelta(minutes=3)


def find_stacks(
    photos: Sequence[Photograph],
    embeddings: Mapping[Photograph, NDArray[np.float32]],
    times: Mapping[Photograph, datetime | None] | None = None,
    faces: FaceRecognizer | None = None,
) -> list[list[Photograph]]:
    """
    Groups the photos of one event into stacks of the same scene or moment.
    Every photo ends up in exactly one stack, and a photo unlike any other is
    a stack of its own. Stacks only ever hold photos of the event they are
    given, so stacking never changes a photo's event.

    `embeddings` holds each photo's embedding (as `compute_embeddings`
    returns), and `times` each photo's time on the shared timeline (as
    `estimate_camera_times` returns). A close call (see `STACK_SIMILARITY`)
    is stacked when any of these agree:
    - both photos were taken at most `MAX_MOMENT_GAP` apart (needs `times`)
    - both photos show the same person (needs `faces`)
    - ORB matches show the same scene (see `SceneMatcher`)

    Stacks come in time order, and each stack's first photo is the earliest.
    """
    if not photos: return []

    # Photos without a time go last, in the order they were given.
    order = sorted(photos, key=lambda photo: _time_key(photo, times))
    table = np.vstack([embeddings[photo] for photo in order])

    matcher = SceneMatcher()
    face_features: dict[Photograph, NDArray[np.float32]] = {}

    def faces_of(photo: Photograph) -> NDArray[np.float32]:
        assert faces is not None
        if photo not in face_features:
            face_features[photo] = faces.describe(photo.decode_downscaled())
        return face_features[photo]

    def confirm(a: Photograph, b: Photograph) -> bool:
        # The checks go from the cheapest to the most expensive.
        time_a, time_b = (None, None) if times is None else (times.get(a), times.get(b))
        if time_a is not None and time_b is not None and abs(time_a - time_b) <= MAX_MOMENT_GAP:
            return True
        if faces is not None and share_face(faces_of(a), faces_of(b)):
            return True
        return matcher.is_same_scene(a, b)

    # Like `get_duplicated_photos`, each photo is only compared with the first
    # photo of its stack, so a long series of slowly changing shots cannot
    # chain into one stack whose first and last photos look different.
    stacks: list[list[Photograph]] = []
    stacked = np.zeros(len(order), dtype=bool)
    for index in range(len(order)):
        if stacked[index]: continue

        start = time.perf_counter()
        stacked[index] = True
        stack = [order[index]]

        # Only the first photo of each stack is compared with the others, so its
        # similarities are computed here, one row at a time. A full table of
        # every photo against every other would take 1.6 GB for 20,000 photos.
        similarity = table @ table[index]
        for other in np.flatnonzero(~stacked & (similarity >= MIN_CLOSE_CALL_SIMILARITY)):
            if similarity[other] >= STACK_SIMILARITY or confirm(order[index], order[other]):
                stacked[other] = True
                stack.append(order[other])

        stacks.append(stack)
        logger.trace(
            "found stack with {} photo(s) ({} of {} photos stacked) in {:.1f} ms",
            len(stack), int(np.count_nonzero(stacked)), len(order), (time.perf_counter() - start) * 1000,
        )

    return stacks


def _time_key(photo: Photograph, times: Mapping[Photograph, datetime | None] | None) -> tuple[bool, datetime]:
    """Sorts photos by time, with photos without one last."""
    time = None if times is None else times.get(photo)
    return (time is None, time or datetime.min)
