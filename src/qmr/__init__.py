"""Quantitative market-regime research framework.

The package is organised as a straight line through a research study:

    data -> features -> regimes -> labels -> models -> validation
         -> backtest -> evaluation

Each stage is an independent module that consumes and returns plain pandas
objects, so any stage can be inspected, replaced or unit-tested on its own.
"""

from qmr.version import __version__

__all__ = ["__version__"]
