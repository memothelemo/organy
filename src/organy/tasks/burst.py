import cv2
import numpy as np

from datetime import datetime
from typing import Any

from organy.models import Photograph

# Longest time, in seconds, between two consecutive shots for them to count as
# part of the same burst. Cameras shoot bursts at 3 to 20 frames per second, so
# frames in a burst are well under a second apart. Photos without sub-second
# EXIF details only have whole seconds, so frames that straddle a second
# boundary appear 1 s apart, which is why the limit is 1 s and not lower.
MAX_BURST_GAP_SECONDS = 1.0

# Fewest keypoints two frames must share, all agreeing on one camera movement,
# before they are compared at all. Frames of the same scene shared 60 to 400 on
# synthetic test images, while unrelated photos shared at most 13.
MIN_BURST_MATCHES = 40

# How much two pixels may differ in any colour (out of 255), after the frames
# are lined up, before the pixel counts as changed. It is high enough to ignore
# sensor noise, slight blur and small exposure shifts between frames.
CHANGED_PIXEL_LEVEL = 40

# Largest share of the frame (0 to 1) that may change for two frames to count
# as the same burst. On synthetic test images, the same pose with camera shake
# changed at most 0.003 of the frame, a raised arm 0.019 and a step sideways
# 0.19, so a subject in a different pose stays a separate photo even when it
# was shot a fraction of a second later. A small, distant subject changes less
# of the frame, so this still needs checking against real event photos.
MAX_BURST_CHANGED_AREA = 0.01

# EXIF format of DateTimeOriginal, e.g. "2026:11:02 14:30:05".
_EXIF_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"


def compute_burst_photos(photos: list[Photograph]) -> list[list[Photograph]]:
    """
    Groups photos that were shot in a burst, meaning nearly identical frames of
    the same moment. What the frames look like decides: once lined up with the
    first frame of the burst, at most `MAX_BURST_CHANGED_AREA` of the picture
    may have changed. A subject in a different pose is a different photo, even
    when it was shot a fraction of a second later.

    Capture time only supports that decision. Frames are only compared when
    the same camera of the same photographer took them at most
    `MAX_BURST_GAP_SECONDS` apart, because each photographer's camera clock is
    set differently and a photo without a readable DateTimeOriginal in its
    EXIF is never part of a burst. Being close in time is never enough on its
    own.

    Every photo ends up in exactly one group. The first photo of each group is
    the sharpest one, which is the one to keep and process further, and the
    rest follow in the order they were shot. A photo outside any burst forms a
    group of one, so `[group[0] for group in groups]` gives one photo per
    burst. Groups come in the order their earliest shot appears in `photos`.
    Nothing is deleted; the other frames are only set aside.

    Each frame is compared with the first frame of its burst, not with the
    shot just before it, so a series of slowly changing poses is never chained
    into one burst.
    """
    # Photos are put in shooting order per camera, so each burst becomes a run
    # of consecutive entries. The index breaks ties between photos taken at
    # the same time and remembers where the photo came from.
    timed: list[tuple[tuple[str, ...], datetime, int]] = []
    untimed: list[int] = []
    for index, photo in enumerate(photos):
        taken = _capture_time(photo.exif)
        if taken is None:
            untimed.append(index)
        else:
            timed.append((_camera_key(photo), taken, index))
    timed.sort()

    bursts: list[list[int]] = [[index] for index in untimed]
    previous: tuple[tuple[str, ...], datetime, int] | None = None
    for entry in timed:
        camera, taken, index = entry
        if (
            previous is not None
            and camera == previous[0]
            and (taken - previous[1]).total_seconds() <= MAX_BURST_GAP_SECONDS
            and _is_same_frame(photos[bursts[-1][0]], photos[index])
        ):
            bursts[-1].append(index)
        else:
            bursts.append([index])
        previous = entry

    # Untimed photos were added before the timed ones, so the groups are put
    # back in the order of their earliest photo in `photos`.
    bursts.sort(key=min)

    groups: list[list[Photograph]] = []
    for burst in bursts:
        group = [photos[index] for index in burst]
        if len(group) > 1:
            # Moving the sharpest frame to the front keeps the others in shooting order.
            sharpest = max(group, key=_sharpness)
            group.remove(sharpest)
            group.insert(0, sharpest)
        groups.append(group)

    return groups


def remove_burst_photos(photos: list[Photograph]) -> list[Photograph]:
    """
    Keeps one photo per burst, the sharpest frame, and leaves out the other
    frames. Photos outside any burst are all kept. See `compute_burst_photos`
    for how bursts are found and how the result is ordered.
    """
    return [group[0] for group in compute_burst_photos(photos)]


def _is_same_frame(first: Photograph, other: Photograph) -> bool:
    """
    Checks whether `other` shows the same moment as `first`, allowing for the
    small camera shake between burst frames.

    Keypoints matched between the two frames work out how the camera moved,
    and `other` is warped onto `first` to undo that movement. What still
    differs after that is the subject itself, so the frames count as the same
    when only a tiny share of the picture changed.
    """
    first_gray = cv2.cvtColor(first.downscaled, cv2.COLOR_BGR2GRAY)
    other_gray = cv2.cvtColor(other.downscaled, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB.create(1_000)
    first_points, first_descriptors = orb.detectAndCompute(first_gray, None)
    other_points, other_descriptors = orb.detectAndCompute(other_gray, None)
    if first_descriptors is None or other_descriptors is None:
        return False

    # Lowe's ratio test keeps a match only when it is clearly better than the
    # next best one, which drops matches on repeated patterns.
    matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING)
    good: list[cv2.DMatch] = []
    for pair in matcher.knnMatch(other_descriptors, first_descriptors, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])
    if len(good) < MIN_BURST_MATCHES:
        return False

    source = np.array([other_points[match.queryIdx].pt for match in good], dtype=np.float32)
    target = np.array([first_points[match.trainIdx].pt for match in good], dtype=np.float32)
    homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
    if homography is None or int(np.count_nonzero(inliers)) < MIN_BURST_MATCHES:
        return False

    # The two photos come from the same camera, so their downscaled sizes match.
    # Colour is compared, not brightness alone, because a subject can be as
    # bright as the background behind it while having a different colour.
    height, width = first_gray.shape
    aligned = cv2.warpPerspective(other.downscaled, homography, (width, height))
    covered = cv2.warpPerspective(np.full_like(other_gray, 255), homography, (width, height)) > 0

    # Blurring hides leftover misalignment of a pixel or two along sharp edges,
    # and matching the average of each colour hides small exposure shifts.
    first_blurred = cv2.GaussianBlur(first.downscaled, (0, 0), 3).astype(np.float32)
    aligned_blurred = cv2.GaussianBlur(aligned, (0, 0), 3).astype(np.float32)
    aligned_blurred += first_blurred[covered].mean(axis=0) - aligned_blurred[covered].mean(axis=0)

    # A pixel counts as changed when any of its colours moved past the level.
    difference = np.abs(first_blurred - aligned_blurred).max(axis=2)
    changed = (difference > CHANGED_PIXEL_LEVEL) & covered
    return bool(np.count_nonzero(changed) <= MAX_BURST_CHANGED_AREA * np.count_nonzero(covered))


def _camera_key(photo: Photograph) -> tuple[str, ...]:
    """
    Identifies the camera that took a photo. Two photographers can own the
    same camera model, so the photographer is part of it, and the serial
    number tells apart two bodies of the same model used by one photographer.
    """
    exif = photo.exif
    return (
        photo.photographer,
        str(exif.get("Make", "")),
        str(exif.get("Model", "")),
        str(exif.get("BodySerialNumber", "")),
    )


def _capture_time(exif: dict[str, Any]) -> datetime | None:
    """
    Reads when a photo was taken from its EXIF DateTimeOriginal, adding the
    fraction of a second from SubSecTimeOriginal when the camera saved it.
    Returns None when the time is missing or unreadable, such as the
    "0000:00:00 00:00:00" some cameras write when their clock was never set.
    """
    value = exif.get("DateTimeOriginal")
    if not isinstance(value, str):
        return None

    try:
        # Some cameras pad EXIF strings with NUL characters or spaces.
        taken = datetime.strptime(value.strip("\x00 "), _EXIF_DATETIME_FORMAT)
    except ValueError:
        return None

    subsec = exif.get("SubSecTimeOriginal")
    if isinstance(subsec, str):
        digits = subsec.strip("\x00 ")
        if digits.isdigit():
            # "05" means 0.05 s, so the digits are read as a decimal fraction.
            # Six digits is the most datetime can hold (microseconds).
            taken = taken.replace(microsecond=int(digits[:6].ljust(6, "0")))

    return taken


def _sharpness(photo: Photograph) -> float:
    """
    Measures how sharp a photo is as the variance of its Laplacian: crisp
    edges give strong, varied responses, while blur from camera shake or a
    missed focus flattens them. The value only makes sense when comparing
    photos of the same scene at the same size, such as frames of one burst.
    """
    gray = cv2.cvtColor(photo.downscaled, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
