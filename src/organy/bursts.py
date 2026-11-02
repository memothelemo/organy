"""
Detects bursts: runs of consecutive shots of the same scene from one camera.

A burst is a relation between photos, not a property of one photo, so photos
are compared with each other. Cameras number files in shooting order, so each
photo is compared only with the next file in the same folder; this keeps the
work linear instead of comparing every pair. Bursts are stacks and never
change a photo's event.
"""

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

# Long side, in pixels, that photos are shrunk to before comparing.
SIZE = 512

# Perceptual hash distances (out of 64 bits). At or below SAME_SCENE the pair
# is a burst; above DIFFERENT_SCENE it is not; in between, ORB decides.
SAME_SCENE = 8
DIFFERENT_SCENE = 24

# ORB matches that must agree on one homography to confirm a borderline pair.
MIN_INLIERS = 40

_phash = cv2.img_hash.PHash.create()
_orb = cv2.ORB.create(nfeatures=1000)
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


def load_gray(path: Path) -> np.ndarray | None:
    """
    Decodes a small grayscale copy of a photo, or None if it cannot be read.

    np.fromfile + cv2.imdecode is used because cv2.imread fails on non-ASCII
    paths on Windows. The REDUCED flag lets the JPEG decoder skip detail.
    """
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if image is None:
        return None

    scale = SIZE / max(image.shape)
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def count_inliers(a: np.ndarray, b: np.ndarray) -> int:
    """
    Counts ORB matches between two images that agree on one homography.

    The ratio test drops ambiguous matches; RANSAC then keeps only the matches
    explained by a single perspective transform, i.e. the same scene seen
    from a slightly moved camera.
    """
    keypoints_a, descriptors_a = _orb.detectAndCompute(a, None)
    keypoints_b, descriptors_b = _orb.detectAndCompute(b, None)
    if descriptors_a is None or descriptors_b is None or len(descriptors_b) < 2:
        return 0

    good = [
        pair[0]
        for pair in _matcher.knnMatch(descriptors_a, descriptors_b, k=2)
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good) < 4:  # a homography needs at least four points
        return 0

    source = np.array([keypoints_a[match.queryIdx].pt for match in good], dtype=np.float32)
    target = np.array([keypoints_b[match.trainIdx].pt for match in good], dtype=np.float32)
    _, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
    return 0 if mask is None else int(mask.sum())


def is_burst_pair(a: np.ndarray, b: np.ndarray) -> bool:
    """Tells whether two consecutive photos show the same scene."""
    distance = _phash.compare(_phash.compute(a), _phash.compute(b))
    if distance <= SAME_SCENE:
        return True
    if distance > DIFFERENT_SCENE:
        return False
    return count_inliers(a, b) >= MIN_INLIERS


def find_bursts(photos: Iterable[Path]) -> list[list[Path]]:
    """
    Groups photos into bursts of two or more, in shooting order.

    Single shots are left out. A photo that cannot be decoded ends the
    current burst instead of stopping the whole run.
    """
    folders: defaultdict[Path, list[Path]] = defaultdict(list)
    for photo in photos:
        folders[photo.parent].append(photo)

    bursts: list[list[Path]] = []
    for files in folders.values():
        run: list[Path] = []
        previous = None
        for path in sorted(files):
            current = load_gray(path)
            if previous is not None and current is not None and is_burst_pair(previous, current):
                run.append(path)
            else:
                if len(run) > 1:
                    bursts.append(run)
                run = [path]
            previous = current

        if len(run) > 1:
            bursts.append(run)

    return bursts
