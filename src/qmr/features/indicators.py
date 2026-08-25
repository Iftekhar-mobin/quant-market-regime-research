"""Technical indicator primitives.

Every function here is *causal*: the value at bar ``t`` uses information from
bars ``<= t`` only. That property is the whole point of this module. A single
centred window or an unshifted extremum is enough to turn an out-of-sample
Sharpe ratio into fiction, so the indicators are implemented directly rather
than pulled from a library whose look-ahead behaviour would have to be audited.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    return pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average true range, Wilder-smoothed."""
    return (
        true_range(high, low, close)
        .ewm(alpha=1 / window, adjust=False, min_periods=window)
        .mean()
    )


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line})


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, smooth: int = 3
) -> pd.DataFrame:
    highest = high.rolling(window).max()
    lowest = low.rolling(window).min()
    span = (highest - lowest).replace(0.0, np.nan)
    k = 100 * (close - lowest) / span
    return pd.DataFrame({"stoch_k": k, "stoch_d": k.rolling(smooth).mean()})


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    highest = high.rolling(window).max()
    lowest = low.rolling(window).min()
    span = (highest - lowest).replace(0.0, np.nan)
    return -100 * (highest - close) / span


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    typical = (high + low + close) / 3
    rolling_mean = typical.rolling(window).mean()
    mean_deviation = typical.rolling(window).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (typical - rolling_mean) / (0.015 * mean_deviation.replace(0.0, np.nan))


def directional_index(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.DataFrame:
    """Directional movement indicators: +DI, -DI and ADX (Wilder smoothing)."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    smoothing = {"alpha": 1 / window, "adjust": False, "min_periods": window}
    atr_series = true_range(high, low, close).ewm(**smoothing).mean().replace(0.0, np.nan)

    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(**smoothing).mean() / atr_series
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(**smoothing).mean() / atr_series

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(**smoothing).mean()

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx})


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    middle = close.rolling(window).mean()
    deviation = close.rolling(window).std(ddof=0)
    upper = middle + n_std * deviation
    lower = middle - n_std * deviation
    width = (upper - lower) / middle.replace(0.0, np.nan)
    position = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": width,
            "bb_position": position,
        }
    )


def keltner(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20, multiplier: float = 1.5
) -> pd.DataFrame:
    middle = close.ewm(span=window, adjust=False).mean()
    band = multiplier * atr(high, low, close, window)
    return pd.DataFrame(
        {"kc_middle": middle, "kc_upper": middle + band, "kc_lower": middle - band}
    )


def donchian(high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    """Rolling price channel.

    Shifted by one bar so the current bar cannot set its own channel and then
    trivially sit at the 0 or 1 extreme of it.
    """
    upper = high.rolling(window).max().shift(1)
    lower = low.rolling(window).min().shift(1)
    span = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame({"dc_upper": upper, "dc_lower": lower, "dc_span": span})


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def money_flow_index(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14
) -> pd.Series:
    typical = (high + low + close) / 3
    raw_flow = typical * volume
    direction = np.sign(typical.diff()).fillna(0.0)
    positive = raw_flow.where(direction > 0, 0.0).rolling(window).sum()
    negative = raw_flow.where(direction < 0, 0.0).rolling(window).sum()
    ratio = positive / negative.replace(0.0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50.0)


def accumulation_distribution(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    span = (high - low).replace(0.0, np.nan)
    multiplier = ((close - low) - (high - close)) / span
    return (multiplier.fillna(0.0) * volume).cumsum()


def realised_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    return np.log(close).diff().rolling(window).std(ddof=0)


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Range-based volatility estimator.

    Roughly five times more efficient than the close-to-close estimator on the
    same sample, which matters when the study window is short.
    """
    log_range = np.log(high / low) ** 2
    return np.sqrt(log_range.rolling(window).mean() / (4 * np.log(2)))


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return (series - mean) / std


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Where the current value sits in its own trailing distribution, in [0, 1]."""
    return series.rolling(window).rank(pct=True)


def hurst_exponent(close: pd.Series, window: int = 100, max_lag: int = 20) -> pd.Series:
    """Rolling Hurst exponent, variance-of-lagged-differences estimator.

    H above 0.5 indicates trend persistence, H below 0.5 mean reversion. It is
    computed on log prices over a coarse grid of lags to keep the rolling apply
    affordable on multi-year intraday samples.
    """
    log_price = np.log(close)
    lags = np.arange(2, max_lag + 1)
    log_lags = np.log(lags)

    # The estimator is a least-squares slope of log(tau) on log(lag), and a
    # least-squares slope is a fixed linear combination of its inputs. Building
    # that combination once turns what would be a per-bar polyfit into a
    # handful of rolling means, which is what makes this affordable on years of
    # intraday data.
    centred_log_lags = log_lags - log_lags.mean()
    weights = centred_log_lags / np.sum(centred_log_lags**2)

    slope = pd.Series(0.0, index=close.index)
    valid = pd.Series(True, index=close.index)
    for lag, weight in zip(lags, weights, strict=True):
        squared_diff = log_price.diff(lag) ** 2
        tau = np.sqrt(squared_diff.rolling(window).mean())
        log_tau = np.log(tau.where(tau > 0))
        valid &= log_tau.notna()
        slope = slope + weight * log_tau.fillna(0.0)

    return slope.where(valid)


def swing_points(high: pd.Series, low: pd.Series, left: int = 5, right: int = 5) -> pd.DataFrame:
    """Confirmed swing highs and lows.

    A pivot at bar ``t`` is only knowable ``right`` bars later, so the flags are
    reported on the confirming bar rather than on the pivot itself. This is the
    causal counterpart of the usual ``argrelextrema`` pivot detection, which
    marks the pivot in place and thereby leaks the future into the feature set.
    """
    window = left + right + 1
    rolling_high = high.rolling(window).max()
    rolling_low = low.rolling(window).min()

    # Bar (t - right) is the centre of the window that ends at bar t.
    centre_high = high.shift(right)
    centre_low = low.shift(right)

    is_swing_high = (centre_high >= rolling_high).astype(float)
    is_swing_low = (centre_low <= rolling_low).astype(float)

    return pd.DataFrame(
        {
            "swing_high": is_swing_high,
            "swing_low": is_swing_low,
            "swing_high_price": centre_high.where(is_swing_high > 0).ffill(),
            "swing_low_price": centre_low.where(is_swing_low > 0).ffill(),
        }
    )
