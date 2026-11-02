import os
import sys

from loguru import logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loguru import Record

# Level used when the user does not set ORGANY_LOG.
DEFAULT_LEVEL = "INFO"

# Loguru's built-in levels, from the most to the least detailed, with one
# colour each, matching Rust's env_logger. Rust has no SUCCESS or CRITICAL, so
# those borrow the colour of INFO and ERROR. Loguru has no public way to list
# its levels, so they are written out here.
LEVEL_COLORS = {
    "TRACE": "cyan",
    "DEBUG": "blue",
    "INFO": "green",
    "SUCCESS": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red",
}

# A fixed template, in the style of Rust's env_logger, so Loguru parses it
# once instead of building a new one for every line.
LOG_FORMAT = (
    "<light-black>[</light-black>{time:YYYY-MM-DDTHH:mm:ss!UTC}Z "
    "<level>{level: <5}</level> {name}<light-black>]</light-black> {message}"
)

# The same template for lines logged by a worker thread, such as the ones
# `load_photos` starts, with the thread's ID after the module name.
THREAD_LOG_FORMAT = LOG_FORMAT.replace("{name}", "{name} thread={extra[thread_id]}")


def setup_for_binary() -> None:
    """
    Sets up the logger for the binary environment.

    Users can use `ORGANY_LOG` environment variable to adjust the minimum log level
    as long as it is a valid level by loguru.

    Lines look like Rust's env_logger, such as
    `[2026-11-02T12:08:39Z INFO  organy.tasks.imports] found 3 candidates`,
    with the time in UTC. Colours are only used when stderr is a terminal.
    """
    level = os.environ.get("ORGANY_LOG", "").strip().upper() or DEFAULT_LEVEL
    if level not in LEVEL_COLORS:
        raise SystemExit(
            f"organy: unknown ORGANY_LOG level {level!r}, expected one of: {', '.join(LEVEL_COLORS)}"
        )

    for name, color in LEVEL_COLORS.items():
        logger.level(name, color=f"<{color}>")

    # Loguru starts with its own stderr handler that shows every level, which
    # would ignore ORGANY_LOG and print each message twice.
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=_format,
    )


def _format(record: "Record") -> str:
    """
    Picks the line template for a log record. Records carrying a `thread_id`
    get the template that shows it. Loguru expects the newline and exception
    to be part of a template returned by a function.
    """
    template = THREAD_LOG_FORMAT if "thread_id" in record["extra"] else LOG_FORMAT
    return template + "\n{exception}"
