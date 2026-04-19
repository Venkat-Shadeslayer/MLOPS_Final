"""
Project-wide logging setup.

Every module gets its logger via `get_logger(__name__)`. Uniform format
across services means Grafana/Loki could parse them later without changes.
"""
from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def _configure_root() -> None:
    """Idempotent root logger setup."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(_LOG_LEVEL)
    for noisy in ("urllib3", "asyncio", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Use module's __name__."""
    _configure_root()
    return logging.getLogger(name)