"""Per-rank logging.

Every rank writes its own file (logs/rank{r}.log) and mirrors INFO+ to stderr
with the rank in the prefix. When a collective deadlocks, the diagnostic is
`tail logs/rank*.log`: the rank whose last line differs from the others is the
one that skipped or added a collective call.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "minidist"


def setup_logging(rank: int, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # Idempotent: spawned children re-import modules, and tests call this repeatedly.
    logger.handlers.clear()

    fmt = logging.Formatter(
        f"%(asctime)s.%(msecs)03d [rank {rank}] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(log_dir / f"rank{rank}.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


def get_logger() -> logging.Logger:
    """The rank-tagged logger; valid after setup_logging() ran in this process."""
    return logging.getLogger(LOGGER_NAME)
