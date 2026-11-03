from organy.tasks.deduplicate import get_duplicated_photos
import organy.tasks.embedding as embedding

from organy.tasks.imports import collect_photos, load_photos
import organy.tasks.imports as imports

import organy.tasks.time_adjustment as time_adjustment

import organy.tasks.events as events
import organy.tasks.faces as faces
import organy.tasks.matching as matching
import organy.tasks.selection as selection
import organy.tasks.stacks as stacks
import organy.tasks.tagging as tagging

__all__ = [
    "get_duplicated_photos",

    "embedding",

    "imports",
    "collect_photos",
    "load_photos",

    "time_adjustment",

    "events",
    "faces",
    "matching",
    "selection",
    "stacks",
    "tagging",
]
