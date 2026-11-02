import cv2
import numpy as np

from datetime import datetime
from numpy.typing import NDArray
from pathlib import Path
from typing import Any

from organy.utils.image import downscale, read_capture_time


# JPEG settings of the stored downscaled copy, about 8 times smaller than
# raw pixels. Measured on 49 stock photos, how similar two photos' embeddings
# are changed by:
# - quality 90 (default 4:2:0 colour): up to 0.036 on close pairs, too much
#   next to a same-scene score of about 0.9
# - quality 95, full colour (4:4:4): up to 0.008, 0.005 on average
#
# Halving colour detail (4:2:0), not the quality setting, causes most of the
# change; even quality 100 with it changed embeddings by about as much as 90.
DOWNSCALED_JPEG_PARAMS = [
    cv2.IMWRITE_JPEG_QUALITY, 95,
    cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
]


class Photograph:
    """
    Imported photograph where it contains a compressed downscaled copy of
    the image, its attribution, EXIF and fingerprint of the original image.
    """

    """
    Downscaled copy of the original photograph, stored as JPEG to save
    memory. A 1,000 px copy takes about 280 KB this way instead of about
    2.2 MB. Use `decode_downscaled` to get its pixels.
    """
    downscaled_jpeg: NDArray[np.uint8]

    """Original file path of the imported photograph"""
    path: Path

    """Photographer who took this photo, named after the folder it was imported from."""
    photographer: str

    """
    When the photo was taken, according to the clock of the camera that took it.
    It comes from the photo's EXIF ("DateTimeOriginal", with "SubsecTimeOriginal"
    and "OffsetTimeOriginal" when present, see `read_capture_time`).

    Each camera's clock may be set a little or a lot off, so times from different
    photographers cannot be compared directly until their clock offsets are corrected.
    
    The file's own dates (its creation or modification time) are never used,
    not even as a fallback. They change whenever a photo is copied or transferred,
    so they do not tell when it was taken.

    It is None when the photo has no usable capture time in its EXIF.
    """
    camera_time: datetime | None

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
        for optimization purposes (and impractical to store a very large image).
        """
        downscaled = downscale(image, 1_000)
        self.path = path
        self.photographer = photographer
        self.exif = exif
        self.camera_time = read_capture_time(exif)

        # The fingerprint only looks at a tiny 32x32 version of the photo, so the
        # downscaled copy gives the same result faster. Each photo gets its own
        # hasher because a hasher keeps scratch data inside it, and several
        # photos are loaded at the same time on different threads.
        self.phash = cv2.img_hash.PHash.create().compute(downscaled)

        # The fingerprint above uses the exact pixels, before JPEG compression
        # changes them slightly.
        ok, encoded = cv2.imencode(".jpg", downscaled, DOWNSCALED_JPEG_PARAMS)
        if not ok: raise ValueError(f"cannot compress downscaled copy of {path}")
        self.downscaled_jpeg = np.asarray(encoded, dtype=np.uint8)

    def decode_downscaled(self) -> NDArray:
        """
        Decodes the downscaled copy of the photo into BGR pixels, with its
        longest side at most 1,000 px. It is decoded again on every call
        (a few milliseconds), so a stage that needs the pixels more than once
        should keep the result while it works on the photo.
        """
        image = cv2.imdecode(self.downscaled_jpeg, cv2.IMREAD_COLOR_BGR)
        if image is None: raise ValueError(f"cannot decode downscaled copy of {self.path}")
        return image
