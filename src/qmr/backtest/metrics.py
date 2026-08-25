"""Risk and performance statistics.

The metric set is deliberately weighted towards risk rather than return.
Accuracy and total return are both easy to inflate; drawdown, the stability of
returns across sub-periods, and the cost of turnover are what separate a result
that survives contact with a live account from one that does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualisation_factor(bars_per_year: int) -> float:
    return float(np.sqrt(bars_per_year))


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """Deepest peak-to-trough decline, with the peak and trough timestamps."""
    if equity.empty:
        return 0.0, None, None
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    trough = drawdown.idxmin()
    peak = equity.loc[:trough].idxmax() if trough is not None else None
    return float(drawdown.min()), peak, trough


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def longest_drawdown_bars(equity: pd.Series) -> int:
    """Longest stretch spent below a previous equity peak."""
    underwater = equity < equity.cummax()
    if not underwater.any():
        return 0
    groups = (~underwater).cumsum()
    return int(underwater.groupby(groups).sum().max())


def performance_metrics(
    returns: pd.Series,
    equity: pd.Series | None = None,
    positions: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
    bars_per_year: int = 6240,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Full performance profile of a per-bar return stream."""
    returns = returns.dropna().astype(float)
    if returns.empty:
        return {key: float("nan") for key in METRIC_ORDER}

    if equity is None:
        equity = (1.0 + returns).cumprod()

    factor = annualisation_factor(bars_per_year)
    mean_return = returns.mean()
    volatility = returns.std(ddof=0)

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max(len(returns) / bars_per_year, 1e-9)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0) if equity.iloc[0] > 0 else np.nan

    excess = returns - risk_free_rate / bars_per_year
    sharpe = float(excess.mean() / volatility * factor) if volatility > 0 else np.nan

    downside = returns[returns < 0]
    downside_deviation = downside.std(ddof=0) if len(downside) > 1 else np.nan
    sortino = (
        float(mean_return / downside_deviation * factor)
        if downside_deviation and downside_deviation > 0
        else np.nan
    )

    worst_drawdown, _, _ = max_drawdown(equity)
    calmar = float(cagr / abs(worst_drawdown)) if worst_drawdown < 0 else np.nan

    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else np.nan

    metrics = {
        "total_return": total_return,
        "cagr": cagr,
        "annualised_volatility": float(volatility * factor),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": float(worst_drawdown),
        "longest_drawdown_bars": float(longest_drawdown_bars(equity)),
        "profit_factor": profit_factor,
        "hit_rate": float((returns > 0).mean()),
        "skew": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
        "var_95": float(returns.quantile(0.05)),
        # Expected shortfall: the average loss on the worst 5% of bars.
        "cvar_95": float(returns[returns <= returns.quantile(0.05)].mean()),
        "bars": float(len(returns)),
    }

    if positions is not None:
        positions = positions.reindex(returns.index).fillna(0.0)
        metrics["exposure"] = float((positions != 0).mean())
        # Turnover in units of notional traded per bar, annualised.
        metrics["annual_turnover"] = float(positions.diff().abs().mean() * bars_per_year)
    else:
        metrics["exposure"] = np.nan
        metrics["annual_turnover"] = np.nan

    if trades is not None and not trades.empty:
        wins = trades[trades["return"] > 0]["return"]
        losses_ = trades[trades["return"] <= 0]["return"]
        metrics["trades"] = float(len(trades))
        metrics["trade_win_rate"] = float(len(wins) / len(trades))
        metrics["avg_win"] = float(wins.mean()) if len(wins) else 0.0
        metrics["avg_loss"] = float(losses_.mean()) if len(losses_) else 0.0
        metrics["avg_trade_bars"] = float(trades["bars"].mean())
        metrics["payoff_ratio"] = (
            float(abs(wins.mean() / losses_.mean())) if len(wins) and len(losses_) and losses_.mean() != 0 else np.nan
        )
    else:
        for key in ("trades", "trade_win_rate", "avg_win", "avg_loss", "avg_trade_bars", "payoff_ratio"):
            metrics[key] = np.nan

    return metrics


METRIC_ORDER = [
    "total_return",
    "cagr",
    "annualised_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "longest_drawdown_bars",
    "profit_factor",
    "hit_rate",
    "exposure",
    "annual_turnover",
    "trades",
    "trade_win_rate",
    "avg_win",
    "avg_loss",
    "avg_trade_bars",
    "payoff_ratio",
    "skew",
    "kurtosis",
    "var_95",
    "cvar_95",
    "bars",
]

# Metrics that read naturally as percentages in the console.
PERCENT_METRICS = {
    "total_return",
    "cagr",
    "annualised_volatility",
    "max_drawdown",
    "hit_rate",
    "exposure",
    "trade_win_rate",
    "avg_win",
    "avg_loss",
    "var_95",
    "cvar_95",
}

METRIC_LABELS = {
    "total_return": "Total return",
    "cagr": "CAGR",
    "annualised_volatility": "Annualised volatility",
    "sharpe": "Sharpe ratio",
    "sortino": "Sortino ratio",
    "calmar": "Calmar ratio",
    "max_drawdown": "Maximum drawdown",
    "longest_drawdown_bars": "Longest drawdown (bars)",
    "profit_factor": "Profit factor",
    "hit_rate": "Positive bars",
    "exposure": "Time in market",
    "annual_turnover": "Annual turnover",
    "trades": "Trades",
    "trade_win_rate": "Winning trades",
    "avg_win": "Average win",
    "avg_loss": "Average loss",
    "avg_trade_bars": "Average holding (bars)",
    "payoff_ratio": "Payoff ratio",
    "skew": "Return skew",
    "kurtosis": "Return kurtosis",
    "var_95": "Value at risk (95%)",
    "cvar_95": "Expected shortfall (95%)",
    "bars": "Bars evaluated",
}


def summarise_metrics(metrics: dict[str, float]) -> pd.DataFrame:
    """Format a metric dictionary as a two-column display table."""
    rows = []
    for key in METRIC_ORDER:
        if key not in metrics:
            continue
        value = metrics[key]
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            formatted = "n/a"
        elif key in PERCENT_METRICS:
            formatted = f"{value * 100:.2f}%"
        elif key in {"bars", "trades", "longest_drawdown_bars"}:
            formatted = f"{value:,.0f}"
        else:
            formatted = f"{value:.3f}"
        rows.append({"Metric": METRIC_LABELS.get(key, key), "Value": formatted})
    return pd.DataFrame(rows)


def metrics_frame(named_metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Stack several metric dictionaries into one comparison table."""
    frame = pd.DataFrame(named_metrics).T
    ordered = [c for c in METRIC_ORDER if c in frame.columns]
    return frame[ordered]
