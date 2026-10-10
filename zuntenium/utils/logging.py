"""
Structured logging configuration for ZUNTENIUM.
"""
from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured ZUNTENIUM logger.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Log level (default INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(f"zuntenium.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s zuntenium.%(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def configure_logging(level: str = "INFO") -> None:
    """
    Configure root ZUNTENIUM logging level.

    Args:
        level: "DEBUG" | "INFO" | "WARNING" | "ERROR".
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger("zuntenium").setLevel(numeric)
