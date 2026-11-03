"""
Tasks related to splitting the shared timeline of photos into events.
"""

from collections.abc import Mapping
from datetime import datetime, timedelta

from organy.models import Photograph


# Default shortest gap between two photos that starts a new event. Shots of
# one performance on the 49 stock photos were at most a few minutes apart,
# while separate performances were usually more than half an hour apart.
#
# This still needs checking against real event photos.
DEFAULT_EVENT_GAP = timedelta(minutes=30)


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
