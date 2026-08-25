"""Assembly of the model feature matrix.

Design rules that apply to every feature produced here:

1. **Causality.** Nothing uses information from a bar later than the one it is
   reported on. See :mod:`qmr.features.indicators`.
2. **Scale invariance.** Features are ratios, z-scores or bounded oscillators
   rather than raw prices, so a model fitted on EURUSD at 1.10 transfers to
   GOLD at 2400 and, more importantly, does not silently learn the price level
   as a proxy for the calendar date.
3. **Stationarity by construction.** Anything that trends without bound (OBV,
   the accumulation/distribution line, raw MACD) enters as a rolling z-score or
   a slope, never as a level.

The pipeline also emits four *regime descriptors* that the regime detectors
consume: ``trend_strength``, ``vol_percentile``, ``momentum_score`` and
``mean_reversion_score``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qmr.config import FeatureConfig
from qmr.features import indicators as ind
from qmr.logging_utils import get_logger

log = get_logger(__name__)

FEATURE_BLOCKS = ["returns", "trend", "momentum", "volatility", "structure", "volume", "session"]

# Descriptors are always produced: the regime layer depends on them and they are
# cheap relative to the rest of the pipeline.
REGIME_DESCRIPTORS = [
    "trend_strength",
    "vol_percentile",
    "momentum_score",
    "mean_reversion_score",
]

_BLOCK_DESCRIPTIONS = {
    "returns": "Multi-horizon log returns, their trailing dispersion and higher moments.",
    "trend": "Moving-average geometry, slope in volatility units, and directional strength.",
    "momentum": "Bounded oscillators: RSI, stochastic, Williams %R, CCI, rate of change.",
    "volatility": "ATR, realised and range-based volatility, term structure and volatility of volatility.",
    "structure": "Position inside the price channel and distance to the last confirmed swing levels.",
    "volume": "Normalised flow measures: OBV slope, money flow, force index, participation.",
    "session": "Cyclical encodings of the trading hour and weekday.",
}


# ---------------------------------------------------------------------------
# Individual blocks
# ---------------------------------------------------------------------------
def _returns_block(frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    close = frame["close"]
    log_return = np.log(close).diff()

    out = {"ret_1": log_return}
    for horizon in horizons:
        if horizon == 1:
            continue
        out[f"ret_{horizon}"] = np.log(close).diff(horizon)
        # Return per unit of trailing risk: a 20-bar move means something very
        # different in a quiet market than in a violent one.
        out[f"ret_{horizon}_norm"] = out[f"ret_{horizon}"] / (
            log_return.rolling(horizon * 3).std(ddof=0).replace(0.0, np.nan) * np.sqrt(horizon)
        )

    out["ret_skew_50"] = log_return.rolling(50).skew()
    out["ret_kurt_50"] = log_return.rolling(50).kurt()
    out["ret_autocorr_50"] = log_return.rolling(50).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=True
    )
    out["up_bar_ratio_20"] = (log_return > 0).rolling(20).mean()
    return pd.DataFrame(out, index=frame.index)


def _trend_block(frame: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    close, high, low = frame["close"], frame["high"], frame["low"]
    atr14 = ind.atr(high, low, close, 14).replace(0.0, np.nan)

    out: dict[str, pd.Series] = {}
    emas = {}
    for window in windows:
        ema = close.ewm(span=window, adjust=False).mean()
        emas[window] = ema
        # Distance to the average, measured in ATR units rather than price.
        out[f"ema_{window}_dist_atr"] = (close - ema) / atr14
        out[f"ema_{window}_slope_atr"] = ema.diff(max(2, window // 4)) / atr14

    ordered = sorted(windows)
    for fast, slow in zip(ordered, ordered[1:], strict=False):
        out[f"ema_{fast}_{slow}_spread_atr"] = (emas[fast] - emas[slow]) / atr14

    dmi = ind.directional_index(high, low, close, 14)
    out["adx_14"] = dmi["adx"]
    out["di_balance"] = (dmi["plus_di"] - dmi["minus_di"]) / (
        dmi["plus_di"] + dmi["minus_di"]
    ).replace(0.0, np.nan)

    macd_frame = ind.macd(close)
    out["macd_hist_atr"] = macd_frame["macd_hist"] / atr14
    out["macd_cross"] = np.sign(macd_frame["macd"] - macd_frame["macd_signal"])

    out["hurst_100"] = ind.hurst_exponent(close, window=100)
    return pd.DataFrame(out, index=frame.index)


def _momentum_block(frame: pd.DataFrame) -> pd.DataFrame:
    close, high, low = frame["close"], frame["high"], frame["low"]
    stoch = ind.stochastic(high, low, close)

    out = {
        "rsi_14": ind.rsi(close, 14),
        "rsi_50": ind.rsi(close, 50),
        "rsi_14_delta": ind.rsi(close, 14).diff(5),
        "stoch_k": stoch["stoch_k"],
        "stoch_d": stoch["stoch_d"],
        "williams_r": ind.williams_r(high, low, close, 14),
        "cci_20": ind.cci(high, low, close, 20),
        "roc_12": close.pct_change(12) * 100,
        "roc_48": close.pct_change(48) * 100,
    }
    return pd.DataFrame(out, index=frame.index)


def _volatility_block(frame: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    close, high, low = frame["close"], frame["high"], frame["low"]
    out: dict[str, pd.Series] = {}

    for window in windows:
        atr_w = ind.atr(high, low, close, window)
        out[f"atr_{window}_rel"] = atr_w / close
        out[f"realised_vol_{window}"] = ind.realised_volatility(close, window)
        out[f"parkinson_vol_{window}"] = ind.parkinson_volatility(high, low, window)

    short_window, long_window = min(windows), max(windows)
    # Volatility term structure: above 1 means the market is heating up.
    out["vol_term_structure"] = out[f"realised_vol_{short_window}"] / out[
        f"realised_vol_{long_window}"
    ].replace(0.0, np.nan)
    out["vol_of_vol"] = out[f"realised_vol_{short_window}"].rolling(long_window).std(ddof=0)

    bands = ind.bollinger(close, 20, 2.0)
    out["bb_width"] = bands["bb_width"]
    out["bb_position"] = bands["bb_position"]
    out["bb_squeeze"] = ind.rolling_percentile(bands["bb_width"], long_window)

    out["gap_atr"] = (frame["open"] - close.shift(1)) / ind.atr(high, low, close, 14).replace(
        0.0, np.nan
    )
    out["bar_range_atr"] = (high - low) / ind.atr(high, low, close, 14).replace(0.0, np.nan)
    return pd.DataFrame(out, index=frame.index)


def _structure_block(frame: pd.DataFrame) -> pd.DataFrame:
    close, high, low = frame["close"], frame["high"], frame["low"]
    atr14 = ind.atr(high, low, close, 14).replace(0.0, np.nan)
    out: dict[str, pd.Series] = {}

    for window in (20, 55, 200):
        channel = ind.donchian(high, low, window)
        # 0 at the channel floor, 1 at the ceiling.
        out[f"dc_{window}_position"] = (close - channel["dc_lower"]) / channel["dc_span"]
        out[f"dc_{window}_width_atr"] = channel["dc_span"] / atr14

    swings = ind.swing_points(high, low, left=5, right=5)
    out["dist_swing_high_atr"] = (swings["swing_high_price"] - close) / atr14
    out["dist_swing_low_atr"] = (close - swings["swing_low_price"]) / atr14
    # Where price sits between the last confirmed swing low and swing high.
    swing_span = (swings["swing_high_price"] - swings["swing_low_price"]).replace(0.0, np.nan)
    out["swing_position"] = (close - swings["swing_low_price"]) / swing_span
    out["bars_since_swing_high"] = _bars_since(swings["swing_high"] > 0)
    out["bars_since_swing_low"] = _bars_since(swings["swing_low"] > 0)

    return pd.DataFrame(out, index=frame.index)


def _volume_block(frame: pd.DataFrame) -> pd.DataFrame:
    close, high, low, volume = frame["close"], frame["high"], frame["low"], frame["volume"]
    obv = ind.on_balance_volume(close, volume)
    adl = ind.accumulation_distribution(high, low, close, volume)

    out = {
        # Levels of OBV/ADL are unbounded random walks; their slopes are not.
        "obv_slope_20": ind.rolling_zscore(obv.diff(20), 200),
        "adl_slope_20": ind.rolling_zscore(adl.diff(20), 200),
        "mfi_14": ind.money_flow_index(high, low, close, volume, 14),
        "force_index_z": ind.rolling_zscore(close.diff() * volume, 100),
        "volume_z_50": ind.rolling_zscore(volume, 50),
        "volume_ratio_20": volume / volume.rolling(20).mean().replace(0.0, np.nan),
    }
    return pd.DataFrame(out, index=frame.index)


def _session_block(frame: pd.DataFrame) -> pd.DataFrame:
    index = frame.index
    hour = index.hour.to_numpy(dtype=float)
    weekday = index.dayofweek.to_numpy(dtype=float)

    # Cyclical encoding: hour 23 and hour 0 are adjacent, and the model should
    # be told so rather than left to infer it from a discontinuous integer.
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "weekday_sin": np.sin(2 * np.pi * weekday / 5),
            "weekday_cos": np.cos(2 * np.pi * weekday / 5),
            "is_london_ny_overlap": ((hour >= 13) & (hour < 17)).astype(float),
        },
        index=index,
    )


def _bars_since(flags: pd.Series) -> pd.Series:
    """Number of bars elapsed since the last time ``flags`` was true."""
    positions = pd.Series(np.arange(len(flags)), index=flags.index, dtype=float)
    marked = positions.where(flags.to_numpy()).ffill()
    return positions - marked


# ---------------------------------------------------------------------------
# Regime descriptors
# ---------------------------------------------------------------------------
def _regime_descriptors(frame: pd.DataFrame) -> pd.DataFrame:
    """Four interpretable coordinates describing the state of the market.

    The regime detectors work in this low-dimensional space rather than on the
    full feature matrix, so the clusters they find can actually be read and
    named ("quiet uptrend", "high-volatility reversal") instead of being an
    opaque partition of a 90-dimensional space.
    """
    close, high, low = frame["close"], frame["high"], frame["low"]
    atr14 = ind.atr(high, low, close, 14).replace(0.0, np.nan)

    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=100, adjust=False).mean()

    # Signed trend strength in volatility units: direction and conviction in one
    # number, comparable across symbols and across time.
    trend_strength = (ema_fast - ema_slow) / atr14

    realised = ind.realised_volatility(close, 20)
    vol_percentile = ind.rolling_percentile(realised, 500)

    # Momentum centred on zero, bounded to roughly [-1, 1].
    momentum_score = (ind.rsi(close, 14) - 50) / 50

    # Below 0.5 the series mean-reverts, above it trends; centred so that zero
    # means "no memory".
    mean_reversion_score = 0.5 - ind.hurst_exponent(close, window=100)

    return pd.DataFrame(
        {
            "trend_strength": trend_strength,
            "vol_percentile": vol_percentile,
            "momentum_score": momentum_score,
            "mean_reversion_score": mean_reversion_score,
        },
        index=frame.index,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_features(
    frame: pd.DataFrame,
    config: FeatureConfig | None = None,
    warmup_bars: int = 300,
) -> pd.DataFrame:
    """Build the full feature matrix for an OHLCV frame.

    Returns the original OHLCV columns alongside the features and the regime
    descriptors, so a single frame can drive the model, the regime layer, the
    backtest and the charts without re-deriving anything.
    """
    config = config or FeatureConfig()
    unknown = set(config.blocks) - set(FEATURE_BLOCKS)
    if unknown:
        raise ValueError(f"Unknown feature block(s): {sorted(unknown)}. Valid: {FEATURE_BLOCKS}")

    parts: list[pd.DataFrame] = [frame.copy()]

    if "returns" in config.blocks:
        parts.append(_returns_block(frame, config.return_horizons))
    if "trend" in config.blocks:
        parts.append(_trend_block(frame, config.trend_windows))
    if "momentum" in config.blocks:
        parts.append(_momentum_block(frame))
    if "volatility" in config.blocks:
        parts.append(_volatility_block(frame, config.volatility_windows))
    if "structure" in config.blocks:
        parts.append(_structure_block(frame))
    if "volume" in config.blocks:
        parts.append(_volume_block(frame))
    if "session" in config.blocks:
        parts.append(_session_block(frame))

    parts.append(_regime_descriptors(frame))

    features = pd.concat(parts, axis=1)
    features = features.loc[:, ~features.columns.duplicated(keep="first")]
    features = features.replace([np.inf, -np.inf], np.nan)

    # Indicator warm-up produces a leading block of NaN. Discard it outright
    # rather than imputing: imputed warm-up values are not market observations.
    before = len(features)
    if warmup_bars > 0:
        features = features.iloc[warmup_bars:]

    # A handful of interior gaps survive (weekend boundaries, zero-range bars).
    # Forward filling those is legitimate — it repeats the last observed value —
    # while a backward fill would import the future, so it is not used.
    features = features.ffill().dropna()

    log.info(
        "Built %d features over %d bars (%d dropped to indicator warm-up and gaps)",
        len(feature_columns(features)),
        len(features),
        before - len(features),
    )
    return features


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Model-input columns: everything except raw OHLCV and any target column."""
    excluded = {"open", "high", "low", "close", "volume", "label", "regime", "forward_return"}
    return [c for c in features.columns if c not in excluded]


def feature_catalogue(config: FeatureConfig | None = None) -> pd.DataFrame:
    """Human-readable description of the enabled feature blocks."""
    config = config or FeatureConfig()
    rows = [
        {"Block": block, "Enabled": block in config.blocks, "Description": description}
        for block, description in _BLOCK_DESCRIPTIONS.items()
    ]
    return pd.DataFrame(rows)
