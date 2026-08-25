"""Market regime identification and characterisation."""

from qmr.regimes.analysis import (
    regime_profile,
    regime_summary,
    regime_transition_matrix,
    run_lengths,
)
from qmr.regimes.detectors import RegimeDetector, build_detector

__all__ = [
    "RegimeDetector",
    "build_detector",
    "regime_profile",
    "regime_summary",
    "regime_transition_matrix",
    "run_lengths",
]
