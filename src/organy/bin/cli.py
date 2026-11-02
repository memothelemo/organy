"""
Command line interface entrypoint for Organy.

The real work itself lives in Organy's library modules.
"""

import argparse
import io
import sys

from pathlib import Path

from organy.importer import collect_photos, load_photos
from organy.tasks import compute_burst_photos, deduplicate_photos
from organy.utils.os import get_recommended_workers


def main(argv: list[str] | None = None) -> int:
    # Windows writes piped output in the locale's code page (e.g. cp1252),
    # which cannot encode many non-ASCII paths. Always write UTF-8 instead.
    # backslashreplace keeps undecodable Linux filenames from crashing print.
    for stream in (sys.stdout, sys.stderr):
        # Streams can be replaced (e.g. by test runners) with ones lacking reconfigure().
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")

    args = _build_parser().parse_args(argv)

    # Stage #1: Collect all photos from given sources with argv
    try:
        photos = collect_photos(args.sources)
    except FileNotFoundError as error:
        print(f"organy: {error}", file=sys.stderr)
        return 1

    # Stage 2: Downscaling all photographs (Photograph class does this automatically)
    photos = load_photos(photos, workers=args.workers)

    # Stage 3: Remove any duplicated/burst photographs
    groups = deduplicate_photos(photos)
    for group in groups:
        if len(group) > 1:
            print(f"Keeping {group[0].path}")
            for duplicate in group[1:]:
                print(f"  duplicate: {duplicate.path}")

    photos = [group[0] for group in groups]
    print(f"{len(photos)} unique photo(s) out of {sum(len(group) for group in groups)}.")

    bursts = compute_burst_photos(photos)
    for burst in bursts:
        if len(burst) > 1:
            print(f"Keeping {burst[0].path}")
            for frame in burst[1:]:
                print(f"  burst frame: {frame.path}")

    photos = [burst[0] for burst in bursts]
    print(f"{len(photos)} photo(s) left after setting aside burst frames.")

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
    parser.add_argument(
        "-w",
        "--workers",
        type=_non_zero_int,
        help="number of workers for Organy to utilize",
        default=get_recommended_workers()
    )
    return parser


def _non_zero_int(value):
    iv = int(value)
    if iv == 0:
        raise argparse.ArgumentTypeError(f"value must be a non-zero integer, got {value}")

    return iv
