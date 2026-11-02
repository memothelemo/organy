"""
Command line interface entrypoint for Organy.

The real work itself lives in Organy's library modules.
"""

import argparse
import io
import sys

from pathlib import Path

from organy.importer import collect_photos, load_photos


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
        # Streams can be replaced (e.g. by test runners) with ones lacking reconfigure().
        if isinstance(stream, io.TextIOWrapper):
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

    for photo in load_photos(photos):
        print(photo)
    print(f"Found {len(photos)} photo(s).")

    # from organy.bursts import find_bursts

    # for number, burst in enumerate(find_bursts(photos), 1):
    #     print(f"Burst {number}: {burst[0].parent}")
    #     for photo in burst:
    #         print(f"  {photo.name}")

    return 0
