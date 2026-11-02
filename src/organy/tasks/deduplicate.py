"""Deduplicating photos task"""

import numpy as np

from organy.models import Photograph


# Highest number of differing fingerprint bits (out of 64) for two photos to
# count as duplicates. On synthetic test images:
# - 0 to 3 bits: copies that were resized, recompressed, brightened or colour
#                graded. They fall within the limit, so they are grouped as duplicates.
# - 6 to 10 bits: different shots of the same scene (a moved camera, a new
#                 subject, a crop). They are over the limit, so they stay separate.
# - About 32 bits: unrelated photos. They stay separate.
#
# The limit of 4 sits in the gap between copies (at most 3) and different
# shots (at least 6), leaving a little room for copies that change more.
#
# These numbers still need checking against real event photos.
MAX_DUPLICATE_DISTANCE = 4


def get_duplicated_photos(photos: list[Photograph]) -> list[list[Photograph]]:
    """
    Groups photos that look like copies of each other, by comparing their
    fingerprints (`Photograph.phash`) bit by bit. Two photos count as copies
    when their fingerprints differ in at most `MAX_DUPLICATE_DISTANCE` of the
    64 bits.

    Only photos that have at least one copy are returned, and each of them is
    in exactly one group. The first photo of each group is the one to keep.
    """
    if not photos: return []

    # Put every fingerprint into one table, one row per photo and one column
    # per bit (0 or 1). Each fingerprint is 8 bytes, so stacking them gives an
    # N x 8 table, and unpackbits splits each byte into its 8 bits, giving N x 64.
    #
    # This table is the whole optimization. Calling OpenCV's compare() on
    # every pair would cross from Python into OpenCV millions of times (about
    # 12.5 million calls for 5,000 photos). With one table, NumPy can compare
    # one photo with all the others in a single step, which handles 5,000
    # photos in about 1.5 s.
    bits = np.unpackbits(np.vstack([photo.phash for photo in photos]), axis=1)

    # `grouped` marks photos that already belong to a group. It starts as all False.
    groups: list[list[Photograph]] = []
    grouped = np.zeros(len(photos), dtype=bool)

    # Visit the photos in order. A photo already in a group is skipped, so the
    # table is scanned once per group instead of once per photo. The more
    # copies there are, the less work is left.
    #
    # Photos are compared only with the first photo of their group, never
    # with the other members. Otherwise copies could chain (A is close to B,
    # B to C) and a series of slowly changing shots would end up in one group
    # even though its first and last photos look different.
    #
    # Example with 4 photos, where B is a copy of A, and D is a copy of C:
    #   visit A: A, B are close to A    -> group [A, B], A and B now grouped
    #   visit B: already grouped        -> skipped
    #   visit C: C, D are close to C    -> group [C, D]
    #   visit D: already grouped        -> skipped
    for index in range(0, len(photos)):
        if grouped[index]: continue

        # Compare this photo with every photo at once. `bits != bits[index]`
        # lines its row up against every row and marks each bit that differs,
        # and count_nonzero counts the marks per row. `distances[i]` is then
        # how many bits photo i differs in (0 for the photo itself).
        distances = np.count_nonzero(bits != bits[index], axis=1)

        # Keep the photos that are close enough and not already in a group.
        members = np.flatnonzero((distances <= MAX_DUPLICATE_DISTANCE) & ~grouped)

        # Mark them as grouped so later visits skip them, then store the group
        # as Photograph objects.
        grouped[members] = True
        groups.append([photos[member] for member in members])

    # Remove any groups that only have exactly one member
    groups = [group for group in groups if len(group) > 1]
    return groups
