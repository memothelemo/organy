"""
Command line interface entrypoint for Organy.

The real work itself lives in Organy's library modules.
"""

import argparse
import io
import sys

from pathlib import Path

from loguru import logger
from organy import pipeline
from organy.utils.os import get_recommended_workers
from organy.utils.types import non_zero_uint


def main(argv: list[str] | None = None) -> int:
    """
    The main entrypoint of the Organy binary, where it accepts custom arguments
    otherwise it will default to `sys.argv` automatically by argparse.
    
    Ideally, there should no binary related functions like loading a logger
    because this function will be used for unit testing.
    """
    args = _build_parser().parse_args(argv)
    logger.debug("using organy with {} worker(s)", args.workers)

    try:
        imported = pipeline.import_photos(args.sources)
    except FileNotFoundError as error:
        print(f"organy: {error}", file=sys.stderr)
        return 1

    photos = pipeline.load_photos(imported, workers=args.workers)
    photos, _ = pipeline.remove_duplicates(photos)

    # TODO: Stage 4 (Fun part), as functions in `organy.pipeline`. Each step
    # only suggests. The user must review everything at the end.
    #
    # 1. Read each photo's capture time from EXIF (DateTimeOriginal, plus
    #    SubSecTimeOriginal and OffsetTimeOriginal when present). Flag photos
    #    without one instead of guessing.
    # 2. Compute an embedding for each photo: a fingerprint like phash, but
    #    with more data and more accurate. It describes what the photo shows,
    #    so two photos of the same scene get close embeddings even with
    #    different colour grading or viewpoints. It comes from a downloaded
    #    model (e.g. CLIP) run through `cv2.dnn`.
    # 3. Estimate each photographer's clock offset. Take the closest embedding
    #    pairs between different photographers, confirm them with ORB matches
    #    and a RANSAC homography, and use the median of their time
    #    differences. Let the user correct it later.
    # 4. Split the corrected timeline into events wherever there is a long gap
    #    between photos. Never merge events automatically (rule 6).
    # 5. Inside each event, group photos of the same scene or moment into
    #    stacks by embedding similarity, with ORB matches to confirm close
    #    calls. Faces add evidence: detect with `cv2.FaceDetectorYN` (YuNet)
    #    and compare with `cv2.FaceRecognizerSF` (SFace). Stacks never change
    #    a photo's event (rule 5).
    # 6. Tag each photo by comparing its CLIP embedding with text prompts
    #    ("stage", "group photo", "food", ...). The prompts are embedded once,
    #    so tags cost almost nothing per photo. Use them for search and stack
    #    names, not for grouping.
    # 7. Inside each stack, suggest the best photo by sharpness (Laplacian variance)
    #    and exposure. Show the raw values, never as "confidence".
    embeddings = pipeline.compute_embeddings(
        photos,
        Path("models/clip-vit-base-patch32-vision.onnx"),
        args.workers
    )

    return 0


def run() -> int:
    """Starts Organy as a program, as the `organy` command and `uv run organy` do."""
    # Windows writes piped output in the locale's code page (e.g. cp1252),
    # which cannot encode many non-ASCII paths. Always write UTF-8 instead.
    # backslashreplace keeps undecodable Linux filenames from crashing print.
    for stream in (sys.stdout, sys.stderr):
        # Streams can be replaced (e.g. by test runners) with ones lacking reconfigure().
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")

    import dotenv
    import organy.bin.logger as logger

    dotenv.load_dotenv()
    logger.setup_for_binary()

    # argparse automatically defaults to `sys.argv` if argv is None.
    return main()


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
        type=non_zero_uint,
        help="number of workers for Organy to utilize",
        default=get_recommended_workers()
    )
    return parser
