import numpy as np

from organy.models import Photograph

# Highest number of differing fingerprint bits (out of 64) for two photos to
# count as duplicates. On synthetic test images, copies that were resized,
# recompressed, brightened or colour graded differed in 0 to 3 bits, while
# different shots of the same scene (a moved camera, a new subject, a crop)
# differed in 6 to 10, and unrelated photos in about 32. These numbers still
# need checking against real event photos.
MAX_DUPLICATE_DISTANCE = 4


def deduplicate_photos(photos: list[Photograph]) -> list[list[Photograph]]:
    """
    Groups photos that are copies of each other, judged by how close their
    fingerprints (`Photograph.phash`) are. This finds exact copies as well as
    copies that were resized or saved again at a different quality.

    Every photo ends up in exactly one group, in the order the photos were
    given. The first photo of each group is the one to keep and process
    further, and the rest are its duplicates. A photo without duplicates forms
    a group of one, so `[group[0] for group in groups]` gives every unique
    photo. Nothing is deleted; the duplicates are only set aside.

    Each photo is compared with the first photo of a group, never with the
    other duplicates in it. This keeps a series of slightly different photos
    from being chained into one group even though its first and last photos
    look different.
    """
    # np.vstack below needs at least one fingerprint, so an empty list is
    # answered right away.
    if not photos: return []

    # Each fingerprint is stored as 8 bytes (a 1x8 array). Stacking them gives
    # an N x 8 table, one row per photo, and unpackbits splits every byte into
    # its 8 bits, giving an N x 64 table of 0s and 1s. With all fingerprints in
    # one table, a single NumPy operation can compare one photo with every
    # other photo. A Python loop over every pair calling OpenCV's compare()
    # would be far slower (25 million calls for 5,000 photos), while this
    # handles 5,000 photos in about 1.5 s.
    bits = np.unpackbits(np.vstack([photo.phash for photo in photos]), axis=1)

    # `grouped` remembers which photos already belong to a group, so no photo
    # is put into two groups. It starts as all False.
    groups: list[list[Photograph]] = []
    grouped = np.zeros(len(photos), dtype=bool)

    # The photos are visited in their given order. The first photo that is not
    # yet in a group starts a new group and becomes the one that is kept.
    # Other photos are compared with this first photo only, never with the
    # other members. Otherwise copies could chain (A is close to B, B to C) and
    # a series of slowly changing shots would end up in one group even though
    # its first and last photos look different.
    #
    # Example with 4 photos, where B is a copy of A, and D is a copy of C:
    #   visit A: A, B are close to A    -> group [A, B], A and B now grouped
    #   visit B: already grouped        -> skipped
    #   visit C: C, D are close to C    -> group [C, D]
    #   visit D: already grouped        -> skipped
    for index, photo in enumerate(photos):
        if grouped[index]:
            continue

        # Compare this photo's 64 bits with every row of the table at once.
        # `bits != bits[index]` marks each differing bit, and count_nonzero
        # counts them per row, so `distances[i]` is how many bits photo i
        # differs from this photo in (0 for the photo itself).
        distances = np.count_nonzero(bits != bits[index], axis=1)
        members = np.flatnonzero((distances <= MAX_DUPLICATE_DISTANCE) & ~grouped)

        # Mark the members as grouped so later visits skip them, then store
        # the group as Photograph objects.
        grouped[members] = True
        groups.append([photos[member] for member in members])

    return groups
