"""Structured logging helper.

Use ``get_logger(__name__)`` instead of ``print``. Handlers are attached once per logger
name so repeated calls (common across imports and notebooks) do not duplicate output.
"""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stderr with a single handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    return logger
