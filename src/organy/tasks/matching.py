"""
Tasks related to checking whether two photos show the same scene, by matching
ORB corners between them.
"""

import cv2
import numpy as np
import threading

from collections.abc import Sequence
from numpy.typing import NDArray

from organy.models import Photograph


# ORB and matching settings used to confirm that two photos show the same scene:
# - ORB_FEATURES: corners found in each downscaled photo
# - MATCH_RATIO: Lowe's ratio test, a match is kept only when it is clearly
#                better than the second best one
# - RANSAC_THRESHOLD: how far (in pixels of the downscaled photo) a matched
#                     corner may land from where the homography puts it
# - MIN_INLIERS: matches that must agree with one homography for the pair
#                to count as the same scene
ORB_FEATURES = 1_000
MATCH_RATIO = 0.75
RANSAC_THRESHOLD = 5.0
MIN_INLIERS = 20

# Size in pixels of the square ORB describes around each corner (OpenCV's default).
ORB_PATCH_SIZE = 31

# Smallest part of the photo that the agreeing matches must cover, from 0
# to 1 (see `SceneMatcher.measure`). On the 49 stock photos, every photo
# carries the same small watermark, and it alone gave unrelated photos up to
# 243 agreeing matches covering only about 0.005 of the photo. Real scene
# matches spread over the photo instead.
MIN_COVERAGE = 0.05


# Corners found by ORB in a photo, their descriptors (None when no corner was
# found), and the photo's area in pixels.
Features = tuple[Sequence[cv2.KeyPoint], NDArray[np.uint8] | None, int]


class SceneMatcher:
    """
    Checks whether two photos show the same scene. Each photo is decoded and
    run through ORB at most once, however many times it is compared.

    It is not safe to use from several threads, so it must only be used on
    the thread that created it.
    """

    def __init__(self) -> None:
        self._thread = threading.get_ident()
        self._orb = cv2.ORB.create(nfeatures=ORB_FEATURES)
        self._matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING)
        self._features: dict[Photograph, Features] = {}

    def features(self, photo: Photograph) -> Features:
        """Finds the ORB corners of a photo's downscaled copy, or returns them from before."""
        assert threading.get_ident() == self._thread, "SceneMatcher used outside the thread that created it"
        if photo not in self._features:
            gray = cv2.cvtColor(photo.decode_downscaled(), cv2.COLOR_BGR2GRAY)

            # ORB fails on photos smaller than its patch (OpenCV raises an
            # assertion while building its image pyramid), and finds no corners
            # in them anyway.
            if min(gray.shape) < ORB_PATCH_SIZE:
                self._features[photo] = ((), None, gray.shape[0] * gray.shape[1])
            else:
                keypoints, descriptors = self._orb.detectAndCompute(gray, None)
                self._features[photo] = (keypoints, descriptors, gray.shape[0] * gray.shape[1])
        return self._features[photo]

    def measure(self, a: Photograph, b: Photograph) -> tuple[int, float]:
        """
        Matches the ORB corners of two photos and returns:
        - how many matches agree with one homography found by RANSAC
        - how much of photo `a` those matches cover, from 0 to 1: the area of
          the box holding the middle 90% of them along each side, so a few
          stray matches far from the rest do not count

        Both are 0 when there are too few matches to find a homography.
        """
        keypoints_a, descriptors_a, area = self.features(a)
        keypoints_b, descriptors_b, _ = self.features(b)
        if descriptors_a is None or descriptors_b is None: return 0, 0.0

        matches = [
            pair[0]
            for pair in self._matcher.knnMatch(descriptors_a, descriptors_b, k=2)
            if len(pair) == 2 and pair[0].distance < MATCH_RATIO * pair[1].distance
        ]

        # A homography needs at least 4 matches, and fewer than MIN_INLIERS
        # could never pass anyway.
        if len(matches) < max(4, MIN_INLIERS): return 0, 0.0

        points_a = np.array([keypoints_a[m.queryIdx].pt for m in matches], dtype=np.float32)
        points_b = np.array([keypoints_b[m.trainIdx].pt for m in matches], dtype=np.float32)
        # When no homography fits, such as when every match lies on one line,
        # OpenCV returns no homography and a mask of all zeros rather than None.
        homography, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, RANSAC_THRESHOLD)
        if homography is None or mask is None: return 0, 0.0

        inliers = points_a[mask.ravel() != 0]
        if len(inliers) == 0: return 0, 0.0

        low, high = np.percentile(inliers, [5, 95], axis=0)
        width, height = high - low
        return len(inliers), float(width * height) / area

    def is_same_scene(self, a: Photograph, b: Photograph) -> bool:
        """
        Whether at least `MIN_INLIERS` ORB matches agree with one homography
        and cover at least `MIN_COVERAGE` of the photo.
        """
        inliers, coverage = self.measure(a, b)
        return inliers >= MIN_INLIERS and coverage >= MIN_COVERAGE
