"""Console tabs. Each module exposes ``render(selection, config)``."""

from tabs import (
    comparison,
    data_explorer,
    overview,
    regime_lab,
    results,
    signals,
    training,
)

__all__ = [
    "overview",
    "data_explorer",
    "regime_lab",
    "training",
    "results",
    "comparison",
    "signals",
]
