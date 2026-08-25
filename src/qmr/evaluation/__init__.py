"""Statistical evaluation of study results."""

from qmr.evaluation.classification import classification_summary, confusion_frame
from qmr.evaluation.significance import (
    bootstrap_sharpe_interval,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    stability_across_folds,
)

__all__ = [
    "classification_summary",
    "confusion_frame",
    "bootstrap_sharpe_interval",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "stability_across_folds",
]
