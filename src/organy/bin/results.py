"""
Organy's results gathered for the user to review, either printed as text or
shown in the results window (`organy.bin.results_window`).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from organy.models import Photograph
from organy.tasks.selection import PhotoQuality


@dataclass
class ResultStack:
    """One stack with its suggested best photo and every photo's measurements."""

    photos: list[Photograph]
    best: Photograph
    qualities: dict[Photograph, PhotoQuality]


@dataclass
class ResultGroup:
    """
    The stacks of one event, or of the photos without a shared time when
    `is_event` is False.
    """

    stacks: list[ResultStack]
    is_event: bool

    @property
    def photo_count(self) -> int:
        return sum(len(stack.photos) for stack in self.stacks)


@dataclass
class Results:
    """Everything the user reviews: every group in order, and each photo's time and tags."""

    groups: list[ResultGroup]
    times: Mapping[Photograph, datetime | None]
    tags: Mapping[Photograph, list[tuple[str, float]]]

    def title(self, group: ResultGroup) -> str:
        """Names a group, such as "Event 2: 2015-01-23 19:46:34 to 19:49:22"."""
        if not group.is_event: return "Photos without a shared time"

        # Groups are found by identity, since comparing them would compare every
        # photo and measurement inside them.
        position = next(index for index, other in enumerate(self.groups) if other is group)
        number = sum(1 for other in self.groups[:position] if other.is_event) + 1
        times = [time for stack in group.stacks for photo in stack.photos if (time := self.times.get(photo))]
        start, end = min(times), max(times)

        # The end only shows its time when the event starts and ends on the same day.
        end_text = end.strftime("%H:%M:%S") if start.date() == end.date() else format_time(end)
        return f"Event {number}: {format_time(start)} to {end_text}"

    def tags_text(self, photo: Photograph) -> str:
        """A photo's tags with their gains (see `organy.tasks.tagging.tag_photos`), such as "stage 0.061, instrument 0.034"."""
        return ", ".join(f"{tag} {gain:.3f}" for tag, gain in self.tags.get(photo, [])) or "(none)"


def format_time(time: datetime | None) -> str:
    """Writes a time to the second, or "(unknown)" when there is none."""
    return "(unknown)" if time is None else time.strftime("%Y-%m-%d %H:%M:%S")


def format_quality(quality: PhotoQuality) -> str:
    """
    Writes a photo's raw measurements. They are never written as a percentage
    or a confidence.
    """
    return (
        f"sharpness {quality.sharpness:.1f}, brightness {quality.brightness:.1f}, "
        f"highlights {quality.highlights:.4f}"
    )


def print_report(results: Results) -> None:
    """Prints every group with its stacks for the user to review."""
    print("* marks the suggested best photo of each stack.")

    for group in results.groups:
        print()
        print(f"{results.title(group)}, {group.photo_count} photo(s) in {len(group.stacks)} stack(s)")

        for index, stack in enumerate(group.stacks, start=1):
            print(f"  Stack {index}")
            for photo in stack.photos:
                print(f"    {'*' if photo is stack.best else ' '} {photo.path}")
                print(f"        photographer {photo.photographer}, time {format_time(results.times.get(photo))}")
                print(f"        {format_quality(stack.qualities[photo])}")
                print(f"        tags {results.tags_text(photo)}")
