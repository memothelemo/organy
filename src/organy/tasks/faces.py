"""
Tasks related to finding faces in photos and telling whether two photos
show the same person.
"""

import cv2
import numpy as np
import threading

from numpy.typing import NDArray
from pathlib import Path

from organy.utils.os import read_bytes


# Lowest detector score for a face to be kept, from 0 to 1. Faces below it
# are often hands, patterns or blurry faces in the crowd.
MIN_FACE_SCORE = 0.9

# Smallest face, in pixels of the downscaled copy, whose features are compared.
# Smaller faces are mostly far away in the crowd, and enlarging them to the
# 112x112 px the recognizer needs gives unreliable features.
MIN_FACE_SIZE = 40

# Lowest similarity of two faces' features to count as the same person.
# 0.363 is the value recommended by the authors of the SFace model on OpenCV.
SAME_FACE_SIMILARITY = 0.363


class FaceRecognizer:
    """
    Finds faces with YuNet (`cv2.FaceDetectorYN`) and describes each one with
    SFace (`cv2.FaceRecognizerSF`), so faces from different photos can be
    compared. Both models come from the OpenCV model zoo, such as
    "face_detection_yunet_2023mar.onnx" (about 0.2 MB) and
    "face_recognition_sface_2021dec.onnx" (about 37 MB), downloaded once.

    It runs on the CPU. It is not safe to use from several threads, so it
    must only be used on the thread that created it.
    """

    def __init__(self, detector_path: Path, recognizer_path: Path) -> None:
        self._thread = threading.get_ident()

        # Both models are read by Python for non-ASCII paths on Windows, see `read_bytes`.
        empty = np.empty(0, dtype=np.uint8)
        self._detector = cv2.FaceDetectorYN.create(
            "onnx", read_bytes(detector_path), empty, (320, 320), MIN_FACE_SCORE,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(
            "onnx", read_bytes(recognizer_path), empty,
        )

    def describe(self, image: NDArray) -> NDArray[np.float32]:
        """
        Finds the faces in a BGR image and returns one row of 128 numbers per
        face of at least `MIN_FACE_SIZE` px. Each row is scaled to a length of
        1, so the similarity of two faces is `a @ b`. A photo without such
        faces gives an empty table with 0 rows.
        """
        assert threading.get_ident() == self._thread, "FaceRecognizer used outside the thread that created it"
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)

        features: list[NDArray[np.float32]] = []
        if faces is not None:
            # Each row is one face: its box (x, y, width, height), five
            # landmarks used to align it, and its score.
            for face in np.asarray(faces, dtype=np.float32):
                if min(face[2], face[3]) < MIN_FACE_SIZE: continue

                aligned = self._recognizer.alignCrop(image, face)
                feature = np.asarray(self._recognizer.feature(aligned), dtype=np.float32).ravel()
                features.append(feature / np.linalg.norm(feature))

        if not features: return np.empty((0, 128), dtype=np.float32)
        return np.vstack(features)


def share_face(a: NDArray[np.float32], b: NDArray[np.float32]) -> bool:
    """
    Whether any face in `a` and any face in `b` (as `FaceRecognizer.describe`
    returns them) look like the same person.
    """
    if len(a) == 0 or len(b) == 0: return False
    return float((a @ b.T).max()) >= SAME_FACE_SIMILARITY
