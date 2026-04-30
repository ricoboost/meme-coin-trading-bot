"""Logging configuration helpers."""

from __future__ import annotations

import logging
from typing import Optional


def configure_logging(
    level: int = logging.INFO,
    logger_name: Optional[str] = None,
    *,
    force: bool = True,
) -> logging.Logger:
    """Configure and return a module logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=force,
    )
    return logging.getLogger(logger_name)
