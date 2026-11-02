"""
Quick and dirty reusable operating system related functions.
"""

import numpy as np

from numpy.typing import NDArray
from pathlib import Path

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
