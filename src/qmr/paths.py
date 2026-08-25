"""Filesystem layout of the project.

Every path is derived from the repository root so the package behaves the same
whether it is imported from a notebook, the CLI or the Streamlit console.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/qmr/paths.py -> src/qmr -> src -> <repository root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The root can be relocated (containers, shared storage) via an env var.
if os.environ.get("QMR_PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["QMR_PROJECT_ROOT"]).resolve()

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "samples"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
REPORT_DIR = PROJECT_ROOT / "reports"
LOG_DIR = PROJECT_ROOT / "logs"

DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"


def ensure_directories() -> None:
    """Create the writable directories the pipeline expects."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_DATA_DIR,
        EXPERIMENT_DIR,
        REPORT_DIR,
        LOG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def data_search_paths() -> list[Path]:
    """Directories scanned for market history, in priority order.

    `data/raw` holds the full local history; `data/samples` holds the trimmed
    files that ship with the repository so the console works on a fresh clone.
    """
    return [RAW_DATA_DIR, SAMPLE_DATA_DIR]
