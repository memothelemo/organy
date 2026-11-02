from numpy.typing import NDArray
from pathlib import Path

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

    def __init__(self, image: NDArray, path: Path) -> None:
        """
        Instantiates a new Photograph object, where it expects a raw image
        data, with its originating file path.

        The image will then be downscaled where its longest side is in 1,000 pixels
        for optimization purposes (and impractical to load a very large image).
        """
        self.downscaled = downscale(image, 1_000)
        self.path = path
