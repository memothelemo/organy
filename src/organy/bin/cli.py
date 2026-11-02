"""
Command line interface entrypoint for Organy.

The real work itself lives in Organy's library modules.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys

from pathlib import Path

from organy.importer import collect_photos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="organy",
        description="Assisted categorization of photos from multiple photographers.",
    )
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help="photo files or folders to import (opens a selection window if omitted)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows writes piped output in the locale's code page (e.g. cp1252),
    # which cannot encode many non-ASCII paths; always write UTF-8 instead.
    # backslashreplace keeps undecodable Linux filenames from crashing print.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

    args = build_parser().parse_args(argv)

    sources = args.sources
    if not sources:
        from organy.bin.dialogs import ask_photo_sources

        sources = ask_photo_sources()
        if not sources:
            print("Nothing selected.")
            return 0

    try:
        photos = collect_photos(sources)
    except FileNotFoundError as error:
        print(f"organy: {error}", file=sys.stderr)
        return 1


    import cv2
    # from organy.processor import is_burst_photo

    # for photo in photos:
    #     print(photo)

    #     image = cv2.imread(photo)
    #     print(is_burst_photo(image))

    sample_photo = cv2.imread(photos[0])
    if sample_photo is None: return 1

    output = cv2.cvtColor(sample_photo, cv2.COLOR_BGR2GRAY)

    # Perform FFT on the image
    fft_image = np.fft.fft2(output)
    fft_shifted = np.fft.fftshift(fft_image)
    magnitude_spectrum = 20 * np.log(np.abs(fft_shifted))

    output = cv2.resize(sample_photo, (500, 500))
    cv2.putText(
        output,
        "{}".format(str(photos[0])),
        (0, 50),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("hello", magnitude_spectrum)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0
