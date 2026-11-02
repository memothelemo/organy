"""
Tasks related to embedding (an image fingerprint like phash, but
with more data and more accurate)
"""

import cv2
import itertools
import threading
import time
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from numpy.typing import NDArray
from pathlib import Path

from organy.models import Photograph


# CLIP was trained on 224x224 photos whose colours were shifted and scaled by
# these per-channel (red, green, blue) values, so every photo has to be
# prepared the same way or the embeddings come out wrong.
CLIP_INPUT_SIZE = (224, 224)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(1, 3, 1, 1)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(1, 3, 1, 1)


def compute_embeddings(
    photos: list[Photograph],
    model_path: Path,
    workers: int = 1
) -> dict[Photograph, NDArray[np.float32]]:
    """
    Computes the embedding of every photo in `photos` (see
    `compute_photo_embedding`) and returns them by photo, in the same order.

    `model_path` is the CLIP ONNX file that `load_embedding_model` takes.
    `workers` is how many photos are processed at the same time:
    - Each worker builds its own copy of the model, about 350 MB of memory
      each, because one model cannot be used by two threads at once.
    - OpenCV already spreads a single photo over all CPU cores, so extra
      workers only compete for them. On 49 photos, 1 worker took about
      68 ms per photo, 2 took 77 ms and 4 took 107 ms, so 1 is the best
      choice on a CPU.
    """
    # A loaded model cannot be copied (OpenCV refuses to pickle it), so the
    # file is read once here and every worker builds its own model from these
    # bytes, which saves reading 350 MB from disk again for each worker.
    start = time.perf_counter()
    model_bytes = read_embedding_model(model_path)
    logger.debug(
        "read embedding model ({:.0f} MB) in {:.1f} ms",
        model_bytes.nbytes / 1_000_000,
        (time.perf_counter() - start) * 1000,
    )

    # Each pool thread keeps its model here and builds it on its first photo.
    local = threading.local()

    # Counts finished photos for the "(1 of 500)" progress in the log, the
    # same way `load_photos` does, since photos finish out of order across
    # threads.
    total = len(photos)
    finished = itertools.count(1)
    lock = threading.Lock()

    def compute(photo: Photograph) -> NDArray[np.float32]:
        with logger.contextualize(thread_id=threading.get_native_id()):
            model: cv2.dnn.Net | None = getattr(local, "model", None)
            if model is None:
                start = time.perf_counter()
                model = local.model = load_embedding_model(model_bytes)
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.debug("loaded embedding model in {:.1f} ms", elapsed_ms)

            start = time.perf_counter()
            embedding = compute_photo_embedding(model, photo)

            elapsed_ms = (time.perf_counter() - start) * 1000
            with lock: done = next(finished)

            logger.trace("computed photo embedding ({} of {}) in {:.1f} ms", done, total, elapsed_ms)
            return embedding

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(photos, pool.map(compute, photos)))


def read_embedding_model(path: Path) -> NDArray[np.uint8]:
    """
    Reads a CLIP ONNX file into memory, ready for `load_embedding_model`.
    Several models can then be built from the same bytes without reading the
    file again.
    """
    # readNetFromONNX opens a path with the narrow-character API, which fails
    # on non-ASCII paths on Windows, so Python reads the bytes instead, like
    # read_bytes does for photos.
    return np.fromfile(path, dtype=np.uint8)


def load_embedding_model(model: Path | NDArray[np.uint8]) -> cv2.dnn.Net:
    """
    Loads the image part of a CLIP model, either from an ONNX file, such as
    "onnx/vision_model.onnx" from the Xenova/clip-vit-base-patch32 repository
    on Hugging Face (about 350 MB, downloaded once), or from that file's bytes
    as returned by `read_embedding_model`.

    It runs on the CPU. A loaded model is not safe to use from several threads
    at the same time, so each thread needs its own.
    """
    if isinstance(model, Path):
        model = read_embedding_model(model)

    # OpenCV 5's graph engine always runs on the CPU and warns if a target
    # is set, so none is set here.
    return cv2.dnn.readNetFromONNX(model)


def compute_photo_embedding(model: cv2.dnn.Net, photo: Photograph) -> NDArray[np.float32]:
    """
    Computes the embedding of a photo: 512 numbers that describe what the
    photo shows. Photos of the same scene get close embeddings even with
    different colour grading or viewpoints.

    The embedding is scaled to a length of 1, so the similarity of two photos
    is `a @ b`. It is about 1 for the same picture and lower the more the
    photos differ. It is a similarity value, not a confidence.

    Only the centre square of the photo is looked at, because CLIP takes
    square input. Photos are cropped rather than squashed so people and
    objects keep their shape.
    """

    # blobFromImage resizes the shorter side to 224, crops the centre square,
    # turns BGR into RGB and scales the pixels from 0-255 to 0-1.
    blob = cv2.dnn.blobFromImage(
        photo.decode_downscaled(),
        scalefactor=1 / 255,
        size=CLIP_INPUT_SIZE,
        swapRB=True,
        crop=True,
    )
    model.setInput((blob - CLIP_MEAN) / CLIP_STD)

    embedding = np.asarray(model.forward(), dtype=np.float32).ravel()
    return embedding / np.linalg.norm(embedding)
