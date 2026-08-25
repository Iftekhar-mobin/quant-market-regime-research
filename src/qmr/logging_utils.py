"""Logging helpers shared by the CLI, the pipeline and the research console."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Install a console handler, and optionally a file handler.

    Calling this twice is safe: existing handlers on the root logger are
    replaced rather than duplicated.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # These libraries are chatty at INFO and drown out the study log.
    for noisy in ("matplotlib", "numexpr", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class CallbackHandler(logging.Handler):
    """Forward formatted log records to an arbitrary callable.

    The research console uses this to stream a running study into the browser
    without waiting for the process to finish.
    """

    def __init__(self, callback: Callable[[str], None], level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self._callback = callback
        self.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._callback(self.format(record))
        except Exception:  # a broken sink must never abort the study
            self.handleError(record)
