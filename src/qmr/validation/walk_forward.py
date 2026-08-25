"""Purged walk-forward validation.

Why not k-fold
--------------
Shuffled cross-validation on a price series trains on Tuesday to predict Monday.
Every fold leaks, every score is optimistic, and the size of the optimism is
unknowable. Walk-forward is the only honest answer: always train on the past,
always test on the future that follows it.

Why the embargo
---------------
Even a strictly forward split leaks when the target is forward-looking. A label
at the last bar of the training window is a function of the twelve bars that
follow it, and those bars are the start of the test window. The fix is a purge:
drop ``embargo_bars`` from the end of each training window so no training label
can see into the period being tested. The embargo should be at least as long as
the label horizon.

Two schemes are available:

``expanding``
    Each fold trains on everything available so far. Uses the most data and
    matches how a model would actually be maintained in production.

``rolling``
    Each fold trains on a fixed-length window. Tests whether the edge survives
    without ancient history, which is a direct probe of regime dependence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qmr.config import ValidationConfig
from qmr.logging_utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Fold:
    """One train/test split, expressed as positional index arrays."""

    index: int
    train_start: int
    train_end: int  # exclusive, before the embargo is applied
    test_start: int
    test_end: int  # exclusive
    embargo_bars: int

    @property
    def train_slice(self) -> slice:
        # The embargo is carved out of the tail of the training window.
        return slice(self.train_start, max(self.train_start, self.train_end - self.embargo_bars))

    @property
    def test_slice(self) -> slice:
        return slice(self.test_start, self.test_end)

    @property
    def train_size(self) -> int:
        train = self.train_slice
        return max(0, train.stop - train.start)

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


def walk_forward_splits(n_samples: int, config: ValidationConfig | None = None) -> list[Fold]:
    """Generate the folds for a study of ``n_samples`` bars."""
    config = config or ValidationConfig()

    if config.scheme not in {"expanding", "rolling"}:
        raise ValueError(f"Unknown validation scheme {config.scheme!r}")
    if not 0 < config.test_size < 1:
        raise ValueError("validation.test_size must be a fraction strictly between 0 and 1")
    if not 0 < config.min_train_size < 1:
        raise ValueError("validation.min_train_size must be a fraction strictly between 0 and 1")

    test_bars = max(1, int(round(n_samples * config.test_size)))
    min_train_bars = max(1, int(round(n_samples * config.min_train_size)))

    available = n_samples - min_train_bars
    if available < test_bars:
        raise ValueError(
            f"Sample of {n_samples} bars is too short for {config.n_folds} folds of "
            f"{test_bars} bars after a {min_train_bars}-bar initial training window. "
            f"Lengthen the study window, or reduce validation.min_train_size / test_size."
        )

    n_folds = max(1, int(config.n_folds))
    # Space the test windows evenly across whatever room is left after the first
    # training window; with one fold this degenerates to a single holdout.
    step = available / n_folds if n_folds > 1 else available

    folds: list[Fold] = []
    for i in range(n_folds):
        test_start = int(round(min_train_bars + i * step))
        test_end = min(n_samples, test_start + test_bars)
        if test_end - test_start < max(1, test_bars // 2):
            # A stub final fold carries no information; drop it rather than
            # reporting a metric computed on a handful of bars.
            continue

        train_start = 0 if config.scheme == "expanding" else max(0, test_start - min_train_bars)

        fold = Fold(
            index=len(folds),
            train_start=train_start,
            train_end=test_start,
            test_start=test_start,
            test_end=test_end,
            embargo_bars=int(config.embargo_bars),
        )
        if fold.train_size < 100:
            log.warning("Skipping fold %d: only %d training bars", i, fold.train_size)
            continue
        folds.append(fold)

    if not folds:
        raise ValueError("Walk-forward produced no usable folds; widen the study window.")

    log.info(
        "Walk-forward (%s): %d folds, ~%d train / %d test bars each, %d-bar embargo",
        config.scheme,
        len(folds),
        int(np.mean([f.train_size for f in folds])),
        test_bars,
        config.embargo_bars,
    )
    return folds


def describe_folds(folds: list[Fold], index: pd.Index) -> pd.DataFrame:
    """Render the fold layout against real timestamps, for review and display."""
    rows = []
    for fold in folds:
        train = fold.train_slice
        rows.append(
            {
                "Fold": fold.index + 1,
                "Train start": index[train.start],
                "Train end": index[max(train.start, train.stop - 1)],
                "Train bars": fold.train_size,
                "Test start": index[fold.test_start],
                "Test end": index[fold.test_end - 1],
                "Test bars": fold.test_size,
                "Embargo bars": fold.embargo_bars,
            }
        )
    return pd.DataFrame(rows)
