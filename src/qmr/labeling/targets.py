"""Target construction for directional studies.

Two labelling schemes are provided.

``directional``
    Sign of the forward return over a fixed horizon, with a volatility-scaled
    dead band so that noise around zero is labelled flat rather than being
    split arbitrarily between long and short.

``triple_barrier``
    The path-aware alternative. From each bar a hypothetical position is opened
    and followed forward until it hits a profit barrier, a loss barrier, or the
    time limit; the label records which barrier came first. Both price barriers
    are set in ATR units, so the labels adapt to the volatility regime instead
    of applying one fixed pip distance to a market whose character changes.

Why this matters for the study
------------------------------
A fixed-horizon sign label rewards a model for calling a move that a real
position would never have survived: a 12-bar forward return of +30 pips is
labelled a win even if the path went 60 pips against the position first. The
triple-barrier label prices that path in, which is why it is the default.

Every label at bar ``t`` is by construction a function of bars after ``t``.
That is legitimate for a target and illegitimate for a feature; the walk-forward
embargo in :mod:`qmr.validation.walk_forward` is what keeps the two apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qmr.config import LabelConfig
from qmr.features import indicators as ind
from qmr.logging_utils import get_logger

log = get_logger(__name__)

LABEL_NAMES = {-1: "Short", 0: "Flat", 1: "Long"}


@dataclass
class LabelResult:
    """Targets plus the diagnostics needed to judge whether they are sane."""

    labels: pd.Series
    forward_return: pd.Series
    holding_bars: pd.Series
    barrier_hit: pd.Series
    method: str

    @property
    def distribution(self) -> pd.Series:
        counts = self.labels.value_counts().sort_index()
        counts.index = [LABEL_NAMES.get(int(i), str(i)) for i in counts.index]
        return counts

    def summary(self) -> dict[str, float | str]:
        share = self.labels.value_counts(normalize=True)
        return {
            "method": self.method,
            "samples": int(len(self.labels)),
            "long_share": float(share.get(1, 0.0)),
            "flat_share": float(share.get(0, 0.0)),
            "short_share": float(share.get(-1, 0.0)),
            "mean_holding_bars": float(self.holding_bars.mean()),
            "mean_forward_return_bps": float(self.forward_return.mean() * 1e4),
        }


def _directional_labels(frame: pd.DataFrame, config: LabelConfig) -> LabelResult:
    close = frame["close"]
    atr_series = ind.atr(frame["high"], frame["low"], close, config.atr_window)

    forward_return = close.shift(-config.horizon) / close - 1.0
    # Dead band scaled to the volatility actually present at entry.
    deadband = (config.deadband_atr * atr_series / close).reindex(close.index)

    labels = pd.Series(0, index=close.index, dtype="int64")
    labels[forward_return > deadband] = 1
    labels[forward_return < -deadband] = -1

    holding = pd.Series(float(config.horizon), index=close.index)
    barrier = pd.Series("time", index=close.index)

    valid = forward_return.notna() & deadband.notna()
    return LabelResult(
        labels=labels[valid],
        forward_return=forward_return[valid],
        holding_bars=holding[valid],
        barrier_hit=barrier[valid],
        method="directional",
    )


def _triple_barrier_labels(frame: pd.DataFrame, config: LabelConfig) -> LabelResult:
    """Walk each bar forward until a barrier is touched.

    The scan is written against numpy arrays and breaks out of the inner loop on
    the first touch, which keeps it near-linear in practice: most paths resolve
    long before the time barrier.
    """
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    atr_series = ind.atr(frame["high"], frame["low"], frame["close"], config.atr_window)
    atr_values = atr_series.to_numpy(dtype=float)

    n = len(close)
    horizon = int(config.horizon)

    labels = np.zeros(n, dtype=np.int64)
    forward_return = np.full(n, np.nan)
    holding = np.full(n, np.nan)
    barrier = np.full(n, "none", dtype=object)

    for i in range(n - horizon):
        entry = close[i]
        volatility = atr_values[i]
        if not np.isfinite(volatility) or volatility <= 0 or entry <= 0:
            continue

        upper = entry + config.take_profit_atr * volatility
        lower = entry - config.stop_loss_atr * volatility

        outcome = 0
        exit_price = close[i + horizon]
        bars_held = horizon
        touched = "time"

        for step in range(1, horizon + 1):
            j = i + step
            hit_upper = high[j] >= upper
            hit_lower = low[j] <= lower

            if hit_upper and hit_lower:
                # Both barriers inside one bar: the intrabar path is unknown, so
                # assume the adverse one to keep the labels conservative.
                outcome, exit_price, bars_held, touched = -1, lower, step, "both"
                break
            if hit_upper:
                outcome, exit_price, bars_held, touched = 1, upper, step, "upper"
                break
            if hit_lower:
                outcome, exit_price, bars_held, touched = -1, lower, step, "lower"
                break

        labels[i] = outcome
        forward_return[i] = exit_price / entry - 1.0
        holding[i] = bars_held
        barrier[i] = touched

    index = frame.index
    valid = np.isfinite(forward_return)

    return LabelResult(
        labels=pd.Series(labels, index=index)[valid],
        forward_return=pd.Series(forward_return, index=index)[valid],
        holding_bars=pd.Series(holding, index=index)[valid],
        barrier_hit=pd.Series(barrier, index=index)[valid],
        method="triple_barrier",
    )


def build_labels(frame: pd.DataFrame, config: LabelConfig | None = None) -> LabelResult:
    """Build targets for every bar in ``frame``."""
    config = config or LabelConfig()

    if config.method == "directional":
        result = _directional_labels(frame, config)
    elif config.method == "triple_barrier":
        result = _triple_barrier_labels(frame, config)
    else:
        raise ValueError(
            f"Unknown labelling method {config.method!r}. Use 'directional' or 'triple_barrier'."
        )

    summary = result.summary()
    log.info(
        "Labels (%s): %d samples | long %.1f%% flat %.1f%% short %.1f%% | mean hold %.1f bars",
        summary["method"],
        summary["samples"],
        summary["long_share"] * 100,
        summary["flat_share"] * 100,
        summary["short_share"] * 100,
        summary["mean_holding_bars"],
    )
    return result
