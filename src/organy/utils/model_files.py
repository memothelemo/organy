"""Quick and dirty reusable utility for loading Organy's downloaded model files."""

import cv2

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class ModelLoadError(Exception):
    """
    Raised when a model file cannot be read or understood, such as one cut
    short by an interrupted download. The message names the file, so the
    user knows which one to download again.
    """


@contextmanager
def loading_model(path: Path) -> Iterator[None]:
    """
    Turns any failure while loading the model file at `path` into a
    `ModelLoadError` naming that file:
    - cv2.error: OpenCV cannot parse the file (a damaged or wrong ONNX file)
    - OSError: the file cannot be read
    - ValueError: a text file (such as the tokenizer's) is not valid UTF-8 or JSON
    """
    try:
        yield
    except (cv2.error, OSError, ValueError) as error:
        # OpenCV's full message also names its own source file and function,
        # while its `err` attribute only says what failed.
        reason = str(getattr(error, "err", "") or error).strip().rstrip(".") or type(error).__name__
        raise ModelLoadError(f"cannot load model file {path}: {reason}") from error
