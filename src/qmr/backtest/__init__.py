"""Vectorised backtesting and performance measurement."""

from qmr.backtest.engine import BacktestResult, run_backtest
from qmr.backtest.metrics import performance_metrics, summarise_metrics

__all__ = ["BacktestResult", "run_backtest", "performance_metrics", "summarise_metrics"]
