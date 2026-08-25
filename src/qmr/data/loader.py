"""Loading and normalisation of OHLCV history.

Broker exports are inconsistent: column casing varies, timestamps arrive either
as epoch seconds or as strings, and duplicate bars are common around weekends
and daylight-saving changes. Everything downstream assumes a clean frame, so
all of that is dealt with exactly once, here.

Canonical output
----------------
A ``DataFrame`` indexed by a sorted, unique, timezone-naive ``DatetimeIndex``
named ``timestamp``, with float columns ``open, high, low, close, volume``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qmr.data.catalog import DatasetInfo, find_dataset
from qmr.logging_utils import get_logger

log = get_logger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

_COLUMN_ALIASES = {
    "open": "open",
    "o": "open",
    "high": "high",
    "h": "high",
    "low": "low",
    "l": "low",
    "close": "close",
    "c": "close",
    "adj close": "close",
    "price": "close",
    "volume": "volume",
    "tick_volume": "volume",
    "tickvolume": "volume",
    "vol": "volume",
    "real_volume": "real_volume",
    "spread": "spread",
    "time": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "timestamp": "timestamp",
}


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        if key in _COLUMN_ALIASES:
            renamed[column] = _COLUMN_ALIASES[key]
    frame = frame.rename(columns=renamed)

    # Unnamed trailing columns are an artefact of exports with a trailing comma.
    junk = [c for c in frame.columns if str(c).lower().startswith("unnamed")]
    frame = frame.drop(columns=junk, errors="ignore")

    # Exports that carry both `time` and `Time` collapse onto the same
    # canonical name; keep the first and discard the rest so every canonical
    # column stays one-dimensional.
    return frame.loc[:, ~frame.columns.duplicated(keep="first")]


def _to_datetime(series: pd.Series) -> pd.Series:
    """Interpret a timestamp column as either epoch seconds or a date string."""
    if pd.api.types.is_numeric_dtype(series):
        # Epoch seconds for anything after ~1973; epoch ms above that scale.
        unit = "ms" if series.dropna().median() > 1e11 else "s"
        return pd.to_datetime(series, unit=unit, errors="coerce")
    return pd.to_datetime(series, errors="coerce", format="mixed")


def read_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Read one CSV of market history into the canonical frame."""
    path = Path(path)
    frame = pd.read_csv(path)
    frame = _normalise_columns(frame)

    if "timestamp" not in frame.columns:
        # Some exports write the timestamp as the unnamed index column.
        first = frame.columns[0]
        frame = frame.rename(columns={first: "timestamp"})

    frame["timestamp"] = _to_datetime(frame["timestamp"])

    missing = [c for c in ["open", "high", "low", "close"] if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {missing}")

    if "volume" not in frame.columns:
        # Spot FX from some sources carries no volume; a constant keeps the
        # volume-derived features defined (and flat, hence uninformative).
        frame["volume"] = 1.0

    frame = frame[["timestamp", *OHLCV_COLUMNS]]
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])

    for column in OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")

    frame = frame.set_index("timestamp").sort_index()

    duplicates = frame.index.duplicated(keep="last")
    if duplicates.any():
        log.info("Dropping %d duplicate timestamps from %s", int(duplicates.sum()), path.name)
        frame = frame[~duplicates]

    # A bar whose high is below its low, or with a non-positive price, is a
    # broker glitch rather than a market event.
    valid = (
        (frame["high"] >= frame["low"])
        & (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        & np.isfinite(frame[OHLCV_COLUMNS]).all(axis=1)
    )
    dropped = int((~valid).sum())
    if dropped:
        log.info("Dropping %d malformed bars from %s", dropped, path.name)

    return frame.loc[valid]


def load_ohlcv(
    symbol: str,
    timeframe: str = "H1",
    start: str | None = None,
    end: str | None = None,
    max_bars: int | None = None,
) -> pd.DataFrame:
    """Load one symbol/timeframe, optionally restricted to a study window.

    ``max_bars`` keeps the most recent N bars and exists so the research
    console stays responsive on multi-year minute data.
    """
    dataset: DatasetInfo = find_dataset(symbol, timeframe)
    frame = read_ohlcv_csv(dataset.path)

    if start:
        frame = frame.loc[frame.index >= pd.Timestamp(start)]
    if end:
        frame = frame.loc[frame.index <= pd.Timestamp(end)]
    if max_bars is not None and len(frame) > max_bars:
        frame = frame.iloc[-max_bars:]

    if frame.empty:
        raise ValueError(
            f"No {symbol} {timeframe} bars remain after applying the window "
            f"start={start!r} end={end!r}."
        )

    log.info(
        "Loaded %s %s: %d bars, %s to %s",
        dataset.symbol,
        dataset.timeframe,
        len(frame),
        frame.index[0].date(),
        frame.index[-1].date(),
    )
    return frame


_RESAMPLE_RULE = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
    "W1": "1W",
    "MN1": "1MS",
}


def resample_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate bars up to a slower timeframe (H1 -> H4, H1 -> D1, ...)."""
    rule = _RESAMPLE_RULE.get(timeframe.upper())
    if rule is None:
        raise ValueError(f"Cannot resample to unknown timeframe {timeframe!r}")

    aggregated = frame.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return aggregated.dropna(subset=["open", "high", "low", "close"])


def describe(frame: pd.DataFrame) -> dict[str, object]:
    """Summary statistics used by the data explorer."""
    returns = np.log(frame["close"]).diff().dropna()
    gaps = frame.index.to_series().diff().dropna()
    median_gap = gaps.median()

    return {
        "bars": int(len(frame)),
        "start": frame.index[0],
        "end": frame.index[-1],
        "median_bar_spacing": median_gap,
        "missing_bars_pct": float((gaps > median_gap * 1.5).mean() * 100),
        "mean_return_bps": float(returns.mean() * 1e4),
        "volatility_bps": float(returns.std() * 1e4),
        "skew": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
        "max_bar_range_pct": float(((frame["high"] - frame["low"]) / frame["close"]).max() * 100),
    }
