"""Quick and dirty reusable operating system related functions."""

import numpy as np

from numpy.typing import NDArray
from multiprocessing import cpu_count
from pathlib import Path


def get_recommended_workers() -> int:
    """
    Returns how many worker threads Organy should use.

    It is calculated by multiplying the 75% of the logical CPU cores, then
    rounded it down. A remaining cores are left free for the operating system,
    the UI and OpenCV's own threads stay responsive.
    """
    count = cpu_count()
    three_fourths = int(count * 0.75)
    return three_fourths if three_fourths > 0 else count


def read_bytes(path: Path) -> NDArray:
    """
    Reads a whole file into a 1D uint8 array, ready for `cv2.imdecode`.

    `cv2.imread` opens files with the C library's narrow-character API on
    Windows, so a path with non-ASCII characters (e.g. "写真/IMG_ñ.jpg") fails
    and it silently returns None. Python opens files with the Unicode API, so
    reading the bytes here and decoding them in memory with `cv2.imdecode`
    works for any path on every OS.

    `path.read_bytes()` + `np.frombuffer` would work too, but `np.fromfile`
    does it in one call and returns a writable array instead of a read-only
    view of a `bytes` object.
    """
    return np.fromfile(path, dtype=np.uint8)
