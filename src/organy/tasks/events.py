"""
Tasks related to splitting the shared timeline of photos into events.
"""

from collections.abc import Mapping
from datetime import datetime, timedelta

from organy.models import Photograph


# Default shortest gap between two photos that starts a new event. On the 49
# stock photos, events were compared with the performer and day saved in each
# photo's EXIF:
# - 30 minutes merged back-to-back performances: only 61% of the photo pairs
#   put in one event belonged to the same performance
# - 10 to 20 minutes all put at least 84% of each performance together, with
#   at least 89% of the pairs in one event belonging to one performance
# - 15 minutes, used here, gave 88% and 95%
#
# It was chosen on those same photos, so it still needs checking against
# real event photos.
DEFAULT_EVENT_GAP = timedelta(minutes=15)


def split_events(
    times: Mapping[Photograph, datetime | None],
    gap: timedelta = DEFAULT_EVENT_GAP,
) -> tuple[list[list[Photograph]], list[Photograph]]:
    """
    Sorts the photos in `times` (each photo with its time on the shared
    timeline, as `estimate_camera_times` returns) and starts a new event
    wherever two photos in a row are more than `gap` apart.

    It returns the events in time order, each with its photos in time order,
    and the photos without a time, which cannot be placed in any event.

    Events are only ever split, never merged, so two events that look alike
    stay separate until the user merges them.
    """
    timed = sorted(
        ((photo, time) for photo, time in times.items() if time is not None),
        key=lambda item: item[1],
    )
    untimed = [photo for photo, time in times.items() if time is None]

    events: list[list[Photograph]] = []
    previous: datetime | None = None
    for photo, time in timed:
        if previous is None or time - previous > gap:
            events.append([])
        events[-1].append(photo)
        previous = time

    return events, untimed
