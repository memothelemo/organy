import cv2

from numpy.typing import NDArray
from pathlib import Path
from typing import Any

from organy.utils.cv import downscale


class Photograph:
    """
    Imported photograph where it contains the raw downscaled image
    data, attributions, EXIF and SHA1 hash of the original image.
    """

    """Downscaled version of the original photograph."""
    downscaled: NDArray

    """Original file path of the imported photograph"""
    path: Path

    """Photographer who took this photo, named after the folder it was imported from."""
    photographer: str

    """
    Camera details saved inside the photo file, such as the camera model ("Model")
    and when the photo was taken ("DateTimeOriginal"). It is empty if the file has none.
    """
    exif: dict[str, Any]

    """
    A 64-bit fingerprint of what the photo looks like. A resized copy or a
    copy saved at lower quality gets the same or a very close fingerprint,
    which a SHA1 hash cannot do.
    
    To compare two photos, call `cv2.img_hash.PHash.create().compare(a, b)`.
    It counts how many of the 64 bits differ: 0 means the photos look the same,
    and two unrelated photos usually differ in about 32.
    """
    phash: NDArray

    def __init__(self, image: NDArray, path: Path, photographer: str, exif: dict[str, Any]) -> None:
        """
        Instantiates a new Photograph object, where it expects a raw image
        data, with its originating file path, photographer and EXIF tags.

        The image will then be downscaled where its longest side is in 1,000 pixels
        for optimization purposes (and impractical to load a very large image).
        """
        self.downscaled = downscale(image, 1_000)
        self.path = path
        self.photographer = photographer
        self.exif = exif

        # The fingerprint only looks at a tiny 32x32 version of the photo, so the
        # downscaled copy gives the same result faster. Each photo gets its own
        # hasher because a hasher keeps scratch data inside it, and several
        # photos are loaded at the same time on different threads.
        self.phash = cv2.img_hash.PHash.create().compute(self.downscaled)
