"""
Command line interface entrypoint for Organy.

The real work itself lives in Organy's library modules.
"""

import argparse
import io
import sys

from datetime import timedelta
from pathlib import Path

from loguru import logger
from organy import pipeline
from organy.bin.results import ResultGroup, Results, ResultStack, print_report
from organy.tasks.events import DEFAULT_EVENT_GAP
from organy.utils.os import get_recommended_workers
from organy.utils.types import non_zero_uint


# Model files Organy expects in the models folder (see `--models`), each
# downloaded once:
# - CLIP ViT-B/32, from the Xenova/clip-vit-base-patch32 repository on Hugging
#   Face: "onnx/vision_model.onnx", "onnx/text_model.onnx", "vocab.json" and
#   "merges.txt", renamed as below
# - YuNet and SFace, from the OpenCV model zoo, keeping their names
CLIP_VISION_MODEL = "clip-vit-base-patch32-vision.onnx"
CLIP_TEXT_MODEL = "clip-vit-base-patch32-text.onnx"
CLIP_VOCAB = "clip-vocab.json"
CLIP_MERGES = "clip-merges.txt"
FACE_DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
FACE_RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx"

FACE_MODEL_FILES = (FACE_DETECTOR_MODEL, FACE_RECOGNIZER_MODEL)
MODEL_FILES = (CLIP_VISION_MODEL, CLIP_TEXT_MODEL, CLIP_VOCAB, CLIP_MERGES, *FACE_MODEL_FILES)


def main(argv: list[str] | None = None) -> int:
    """
    The main entrypoint of the Organy binary, where it accepts custom arguments
    otherwise it will default to `sys.argv` automatically by argparse.
    
    Ideally, there should no binary related functions like loading a logger
    because this function will be used for unit testing.
    """
    args = _build_parser().parse_args(argv)
    logger.debug("using organy with {} worker(s)", args.workers)

    models: Path = args.models
    missing = [name for name in MODEL_FILES if not (models / name).is_file()]
    if args.no_faces:
        missing = [name for name in missing if name not in FACE_MODEL_FILES]
    if missing:
        print(f"organy: missing model file(s) in {models}: {', '.join(missing)}", file=sys.stderr)
        return 1

    sources: list[Path] = args.sources
    if not sources:
        if args.no_ui:
            print("organy: no photos given, pass photo files or folders when using --no-ui", file=sys.stderr)
            return 1

        from organy.bin.dialogs import ask_photo_sources
        sources = ask_photo_sources()
        if not sources:
            logger.info("no photos selected, nothing to do")
            return 0

    try:
        imported = pipeline.import_photos(sources)
    except FileNotFoundError as error:
        print(f"organy: {error}", file=sys.stderr)
        return 1

    photos = pipeline.load_photos(imported, workers=args.workers)
    photos, _ = pipeline.remove_duplicates(photos)

    # Stage 4: each step only suggests. The user reviews everything at the end.
    embeddings = pipeline.compute_embeddings(photos, models / CLIP_VISION_MODEL)
    times = pipeline.estimate_camera_times(embeddings)
    events, untimed = pipeline.split_events(times, timedelta(minutes=args.event_gap))

    # Photos without a shared time belong to no event, but similar ones among
    # them are still worth stacking, so they are stacked as one group of their own.
    groups = events + ([untimed] if untimed else [])
    face_models = None if args.no_faces else (models / FACE_DETECTOR_MODEL, models / FACE_RECOGNIZER_MODEL)
    stacks = pipeline.find_stacks(groups, embeddings, times, face_models)

    tags = pipeline.tag_photos(
        embeddings, models / CLIP_TEXT_MODEL, models / CLIP_VOCAB, models / CLIP_MERGES,
    )
    suggestions = iter(pipeline.suggest_best_photos(stack for group in stacks for stack in group))

    results = Results(
        groups=[
            ResultGroup(
                stacks=[ResultStack(stack, *next(suggestions)) for stack in group],
                is_event=index < len(events),
            )
            for index, group in enumerate(stacks)
        ],
        times=times,
        tags=tags,
    )

    if args.no_ui:
        print_report(results)
    else:
        # Imported here so the window's modules only load when it is shown.
        from organy.bin.results_window import show_results
        show_results(results)
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
        nargs="*",
        type=Path,
        help="photo files or folders to import, chosen in a window when none are given",
    )
    parser.add_argument(
        "-m",
        "--models",
        type=Path,
        default=Path("models"),
        help="folder holding the downloaded model files (default: models)",
    )
    parser.add_argument(
        "--event-gap",
        type=non_zero_uint,
        default=int(DEFAULT_EVENT_GAP.total_seconds() // 60),
        help="minutes without photos that start a new event (default: %(default)s)",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="print the results as text instead of showing them in a window",
    )
    parser.add_argument(
        "--no-faces",
        action="store_true",
        help="do not compare faces when stacking photos",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=non_zero_uint,
        help="number of workers for Organy to utilize",
        default=get_recommended_workers()
    )
    return parser

