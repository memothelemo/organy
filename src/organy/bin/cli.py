"""
Command line interface entrypoint for Organy.

The real work itself lives in Organy's library modules.
"""

import argparse
import io
import sys

from pathlib import Path

from organy.importer import collect_photos, load_photos


def main(argv: list[str] | None = None) -> int:
    # Windows writes piped output in the locale's code page (e.g. cp1252),
    # which cannot encode many non-ASCII paths. Always write UTF-8 instead.
    # backslashreplace keeps undecodable Linux filenames from crashing print.
    for stream in (sys.stdout, sys.stderr):
        # Streams can be replaced (e.g. by test runners) with ones lacking reconfigure().
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")

    args = _build_parser().parse_args(argv)

    try:
        photos = collect_photos(args.sources)
    except FileNotFoundError as error:
        print(f"organy: {error}", file=sys.stderr)
        return 1

    print(load_photos(photos, workers=12))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="organy",
        description="Assisted categorization of photos from multiple photographers.",
    )
    parser.add_argument(
        "sources",
        # TODO: Switch back to `*` once we implement a selection UI
        nargs="+",
        type=Path,
        help="photo files or folders to import",
    )
    return parser
