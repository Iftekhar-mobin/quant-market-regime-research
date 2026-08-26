"""Vectorised position-level backtester.

Execution model
---------------
A signal produced on the close of bar ``t`` is executed at the open of bar
``t + execution_lag`` (default: the very next bar). The position then earns the
open-to-open return of the bars it is held for. This is the conservative
convention: the alternative — marking the signal against the close of the same
bar that produced it — quietly assumes the trade was filled at a price that was
not knowable when the decision was made, and it is the single most common way a
backtest ends up unreproducible in a live account.

Costs
-----
Every change in position pays ``cost_bps + slippage_bps`` on the traded notional.
Flipping from long to short therefore pays twice, which is correct: it is two
transactions. Costs are charged on the bar the position changes.

Minimum holding period
----------------------
A model trained on a twelve-bar-ahead target expresses one opinion about the
next twelve bars, not twelve independent opinions. Acting on its output every
bar pays the spread twelve times to hold what is economically a single
position, and on intraday FX that alone is enough to turn a real edge into a
loss. ``min_holding_bars`` therefore holds a new position for at least that
many bars before a fresh signal can move it. Set it to the label horizon.

The engine is intentionally simple — no partial fills, no margin model, no
intrabar stops. It exists to price a *signal*, and every extra mechanism inside
it is one more place for an assumption to hide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qmr.backtest.metrics import drawdown_series, performance_metrics
from qmr.config import BacktestConfig
from qmr.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class BacktestResult:
    """Everything produced by one backtest run."""

    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    benchmark_equity: pd.Series
    benchmark_returns: pd.Series
    metrics: dict[str, float] = field(default_factory=dict)
    benchmark_metrics: dict[str, float] = field(default_factory=dict)
    costs_paid: float = 0.0

    @property
    def drawdown(self) -> pd.Series:
        return drawdown_series(self.equity)

    @property
    def benchmark_drawdown(self) -> pd.Series:
        return drawdown_series(self.benchmark_equity)

    def frame(self) -> pd.DataFrame:
        """Bar-level record of the run, for export and charting."""
        return pd.DataFrame(
            {
                "position": self.positions,
                "strategy_return": self.returns,
                "equity": self.equity,
                "drawdown": self.drawdown,
                "benchmark_return": self.benchmark_returns,
                "benchmark_equity": self.benchmark_equity,
            }
        )


def _extract_trades(
    positions: pd.Series, prices: pd.Series, returns: pd.Series
) -> pd.DataFrame:
    """Collapse the per-bar position series into discrete round trips."""
    records = []
    active_side = 0
    entry_index: pd.Timestamp | None = None
    entry_position = 0

    values = positions.to_numpy()
    for i, side in enumerate(values):
        if side != active_side:
            if active_side != 0 and entry_index is not None:
                window = returns.iloc[entry_position : i]
                records.append(
                    {
                        "entry_time": entry_index,
                        "exit_time": positions.index[i],
                        "side": "long" if active_side > 0 else "short",
                        "entry_price": float(prices.iloc[entry_position]),
                        "exit_price": float(prices.iloc[i]),
                        "bars": int(i - entry_position),
                        "return": float((1.0 + window).prod() - 1.0),
                    }
                )
            if side != 0:
                entry_index = positions.index[i]
                entry_position = i
            active_side = side

    # A position still open on the final bar is closed at the last price.
    if active_side != 0 and entry_index is not None:
        window = returns.iloc[entry_position:]
        records.append(
            {
                "entry_time": entry_index,
                "exit_time": positions.index[-1],
                "side": "long" if active_side > 0 else "short",
                "entry_price": float(prices.iloc[entry_position]),
                "exit_price": float(prices.iloc[-1]),
                "bars": int(len(positions) - entry_position),
                "return": float((1.0 + window).prod() - 1.0),
            }
        )

    return pd.DataFrame(records)


def enforce_min_holding(signals: pd.Series, min_bars: int) -> pd.Series:
    """Hold a newly opened position for at least ``min_bars`` bars.

    While the holding period is running the incoming signal is ignored, so the
    strategy commits to a position for as long as the forecast that opened it
    is meant to apply. Once the period elapses the target signal takes over
    again, including a flip straight to the opposite side.
    """
    if min_bars <= 1:
        return signals

    target = signals.to_numpy(dtype=float)
    held = np.zeros_like(target)

    current = 0.0
    bars_in_position = 0

    for i, desired in enumerate(target):
        if current != 0.0 and bars_in_position < min_bars:
            bars_in_position += 1
        else:
            if desired != current:
                current = desired
                bars_in_position = 1 if desired != 0.0 else 0
            elif current != 0.0:
                bars_in_position += 1
        held[i] = current

    return pd.Series(held, index=signals.index, name=signals.name)


def apply_session_filter(signals: pd.Series, hours: list[int]) -> pd.Series:
    """Allow new entries only during the given hours of the day.

    Spreads on FX are widest in the thin hours, and a constant cost assumption
    understates that. Restricting entries to liquid sessions is the cheap way to
    stop a strategy paying its worst prices; an existing position is left alone,
    because closing it early is a different decision.
    """
    if not hours:
        return signals

    allowed = signals.index.hour.isin(hours)
    filtered = signals.where(allowed, other=np.nan)
    # Outside the session, hold whatever was already on rather than flattening.
    return filtered.ffill().fillna(0.0)


def volatility_scalar(
    frame: pd.DataFrame,
    target: float,
    window: int,
    bars_per_year: int,
    max_leverage: float,
) -> pd.Series:
    """Position multiplier that targets a constant annualised volatility.

    Sharpe is return divided by volatility. A strategy that holds one unit
    through both calm and violent markets earns most of its variance in the
    violent ones without earning proportionally more return, so its Sharpe is
    dragged down by periods it was never compensated for. Sizing inversely to
    trailing volatility removes that drag.

    This is not a way to manufacture return — it does not change the sign of
    anything — but it is one of the few genuinely free improvements in
    portfolio construction, and it is standard practice at every systematic
    fund.

    The estimate uses returns up to bar ``t`` and is then shifted a further bar,
    so the multiplier applied at execution was knowable beforehand.
    """
    log_return = np.log(frame["close"]).diff()
    realised = log_return.rolling(window).std(ddof=0) * np.sqrt(bars_per_year)
    scalar = target / realised.replace(0.0, np.nan)
    return scalar.shift(1).clip(upper=max_leverage).fillna(0.0)


def _size_at_entry(side: pd.Series, scalar: pd.Series) -> pd.Series:
    """Fix the position multiplier at entry and hold it for the trade.

    Rebalancing to the volatility target on every bar would charge transaction
    cost for each drift in the estimate. A real system sizes when it opens and
    leaves the position alone, so that is what is modelled here.
    """
    entries = side.ne(side.shift()) & side.ne(0)
    held = scalar.where(entries).ffill().fillna(0.0)
    return held.where(side != 0, 0.0)


def run_backtest(
    frame: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig | None = None,
    bars_per_year: int | None = None,
) -> BacktestResult:
    """Backtest a signal series against OHLCV data.

    Parameters
    ----------
    frame
        OHLCV history; only ``open`` and ``close`` are used.
    signals
        Desired exposure per bar in ``[-1, 1]``. Integer signals (-1, 0, 1) are
        the usual case, but fractional conviction sizing works unchanged.
    """
    config = config or BacktestConfig()
    bars_per_year = bars_per_year or config.bars_per_year

    aligned = signals.reindex(frame.index).fillna(0.0).astype(float).clip(-1.0, 1.0)
    aligned = apply_session_filter(aligned, config.session_hours)
    aligned = enforce_min_holding(aligned, config.min_holding_bars)

    # The signal on bar t is acted on at the open of bar t + lag.
    side = aligned.shift(config.execution_lag).fillna(0.0)

    size = pd.Series(config.position_size, index=frame.index)
    if config.volatility_target:
        size = config.position_size * _size_at_entry(
            side,
            volatility_scalar(
                frame,
                config.volatility_target,
                config.volatility_window,
                bars_per_year,
                config.max_leverage,
            ),
        )

    positions = side * size

    # Open-to-open returns match the execution convention above.
    execution_price = frame["open"]
    bar_return = execution_price.pct_change().shift(-1).fillna(0.0)

    gross_return = positions * bar_return

    cost_rate = (config.cost_bps + config.slippage_bps) / 1e4
    traded_notional = positions.diff().abs().fillna(positions.abs())
    costs = traded_notional * cost_rate

    net_return = gross_return - costs
    equity = config.initial_capital * (1.0 + net_return).cumprod()

    # Buy and hold, charged one entry at the start and nothing thereafter — a
    # position held for the whole window genuinely transacts once. Over a
    # multi-year study that single charge is a rounding error, but leaving it
    # out entirely would hand the benchmark an advantage the strategy does not
    # get.
    benchmark_return = bar_return.copy()
    benchmark_return.iloc[0] -= cost_rate
    benchmark_equity = config.initial_capital * (1.0 + benchmark_return).cumprod()

    # Trades are delimited by the discrete side, not the sized position: a
    # change in size is not a new trade, and counting it as one would inflate
    # the trade count and every per-trade statistic derived from it.
    trades = _extract_trades(side, execution_price, net_return)

    metrics = performance_metrics(
        net_return, equity, positions, trades, bars_per_year=bars_per_year
    )
    benchmark_metrics = performance_metrics(
        benchmark_return, benchmark_equity, bars_per_year=bars_per_year
    )

    log.info(
        "Backtest: %d bars, %d trades, net return %.2f%%, Sharpe %.2f, max drawdown %.2f%%",
        len(net_return),
        len(trades),
        metrics["total_return"] * 100,
        metrics["sharpe"] if np.isfinite(metrics["sharpe"]) else float("nan"),
        metrics["max_drawdown"] * 100,
    )

    return BacktestResult(
        equity=equity,
        returns=net_return,
        positions=positions,
        trades=trades,
        benchmark_equity=benchmark_equity,
        benchmark_returns=benchmark_return,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        costs_paid=float(costs.sum()),
    )


def signals_from_probabilities(
    probabilities: pd.DataFrame,
    threshold: float = 0.55,
) -> pd.Series:
    """Convert class probabilities into a -1 / 0 / +1 exposure series.

    A position is taken only when the winning class clears ``threshold``. The
    threshold is the strategy's main risk dial: raising it trades fewer, higher
    conviction positions, and the sweep in the results view shows what that does
    to risk-adjusted return.
    """
    if probabilities.empty:
        return pd.Series(dtype=float)

    long_probability = probabilities.get("long", pd.Series(0.0, index=probabilities.index))
    short_probability = probabilities.get("short", pd.Series(0.0, index=probabilities.index))

    signal = pd.Series(0.0, index=probabilities.index)
    signal[(long_probability >= threshold) & (long_probability >= short_probability)] = 1.0
    signal[(short_probability >= threshold) & (short_probability > long_probability)] = -1.0
    return signal
