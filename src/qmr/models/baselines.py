"""Rule-based benchmark strategies.

A machine-learning result means nothing without something to beat. These are
the honest comparators: strategies a practitioner could have run in 1990, priced
through the same execution model and the same costs as the learned signals.

If a gradient-boosted ensemble on ninety engineered features cannot beat a
moving-average crossover after costs, the correct conclusion is that it has not
found an edge — not that the benchmark needs handicapping.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from qmr.features import indicators as ind


def buy_and_hold(frame: pd.DataFrame, **_: object) -> pd.Series:
    """Always long. The benchmark every directional strategy is measured against."""
    return pd.Series(1.0, index=frame.index, name="signal")


def ma_crossover(frame: pd.DataFrame, fast: int = 50, slow: int = 200, **_: object) -> pd.Series:
    """Classic trend following: long above the slow average, short below."""
    close = frame["close"]
    fast_ma = close.ewm(span=fast, adjust=False).mean()
    slow_ma = close.ewm(span=slow, adjust=False).mean()
    return pd.Series(np.sign(fast_ma - slow_ma), index=frame.index, name="signal").fillna(0.0)


def rsi_mean_reversion(
    frame: pd.DataFrame, window: int = 14, lower: float = 30.0, upper: float = 70.0, **_: object
) -> pd.Series:
    """Fade the extremes: long when oversold, short when overbought, flat between.

    Positions are held until the oscillator returns through the midpoint, so the
    strategy is a genuine mean-reversion rule rather than a one-bar blip.
    """
    rsi_series = ind.rsi(frame["close"], window)

    signal = pd.Series(np.nan, index=frame.index)
    signal[rsi_series < lower] = 1.0
    signal[rsi_series > upper] = -1.0
    signal[(rsi_series > 45) & (rsi_series < 55)] = 0.0
    return signal.ffill().fillna(0.0)


def donchian_breakout(frame: pd.DataFrame, window: int = 55, **_: object) -> pd.Series:
    """Turtle-style channel breakout, the canonical momentum benchmark."""
    channel = ind.donchian(frame["high"], frame["low"], window)
    close = frame["close"]

    signal = pd.Series(np.nan, index=frame.index)
    signal[close > channel["dc_upper"]] = 1.0
    signal[close < channel["dc_lower"]] = -1.0
    return signal.ffill().fillna(0.0)


def volatility_filtered_trend(
    frame: pd.DataFrame, fast: int = 50, slow: int = 200, adx_floor: float = 20.0, **_: object
) -> pd.Series:
    """Trend following that stands aside when the market is not trending.

    This is the rule-based sibling of the regime-conditioning hypothesis: the
    same trend signal, switched off in the states where it historically fails.
    It is the benchmark the learned regime models most need to beat.
    """
    trend = ma_crossover(frame, fast=fast, slow=slow)
    adx = ind.directional_index(frame["high"], frame["low"], frame["close"], 14)["adx"]
    return trend.where(adx >= adx_floor, 0.0).fillna(0.0)


BASELINES: dict[str, tuple[str, Callable[..., pd.Series], str]] = {
    "buy_and_hold": (
        "Buy and hold",
        buy_and_hold,
        "Always long. The return the market handed out for doing nothing.",
    ),
    "ma_crossover": (
        "MA crossover (50/200)",
        ma_crossover,
        "Long above the slow average, short below. The default trend benchmark.",
    ),
    "rsi_mean_reversion": (
        "RSI mean reversion",
        rsi_mean_reversion,
        "Fade oscillator extremes, exit through the midpoint.",
    ),
    "donchian_breakout": (
        "Donchian breakout (55)",
        donchian_breakout,
        "Trade the break of the 55-bar channel; the classic momentum rule.",
    ),
    "volatility_filtered_trend": (
        "ADX-filtered trend",
        volatility_filtered_trend,
        "Trend following switched off when ADX says there is no trend.",
    ),
}


def build_baseline_signals(name: str, frame: pd.DataFrame, **params: object) -> pd.Series:
    """Generate the signal series for one named baseline."""
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline {name!r}. Available: {sorted(BASELINES)}")
    _, function, _ = BASELINES[name]
    return function(frame, **params)


def baseline_catalogue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Key": key, "Strategy": label, "Rule": description}
            for key, (label, _, description) in BASELINES.items()
        ]
    )
