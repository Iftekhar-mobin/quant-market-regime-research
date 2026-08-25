"""Is the result real, or is it the search?

A Sharpe ratio computed once, on one configuration, is a point estimate with a
wide standard error. A Sharpe ratio selected as the best of forty configurations
is not an estimate of anything at all unless the selection is priced in. This
module provides the three checks the study reports alongside every headline
number:

* a **bootstrap confidence interval**, which resamples the return stream in
  blocks so that autocorrelation is preserved;
* the **probabilistic Sharpe ratio**, which corrects the naive standard error
  for the skew and fat tails of financial returns;
* the **deflated Sharpe ratio**, which additionally discounts for the number of
  configurations tried before this one was chosen.

References: Bailey and Lopez de Prado, *The Deflated Sharpe Ratio* (2014);
Politis and Romano, *The Stationary Bootstrap* (1994).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _sharpe(returns: np.ndarray, bars_per_year: int) -> float:
    volatility = returns.std(ddof=0)
    if volatility <= 0:
        return float("nan")
    return float(returns.mean() / volatility * np.sqrt(bars_per_year))


def stationary_bootstrap_indices(
    n: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Index sample for the stationary bootstrap of Politis and Romano.

    Blocks of geometrically distributed length are stitched together, which
    preserves short-range dependence in the return series. Resampling
    individual bars independently would destroy the autocorrelation and produce
    confidence intervals that are far too narrow.
    """
    p = 1.0 / max(mean_block, 1.0)
    indices = np.empty(n, dtype=np.int64)
    position = rng.integers(0, n)

    for i in range(n):
        indices[i] = position
        if rng.random() < p:
            position = rng.integers(0, n)
        else:
            position = (position + 1) % n
    return indices


def bootstrap_sharpe_interval(
    returns: pd.Series,
    bars_per_year: int = 6240,
    n_samples: int = 1000,
    confidence_level: float = 0.95,
    mean_block: float | None = None,
    seed: int = 7,
) -> dict[str, float]:
    """Bootstrap confidence interval for the annualised Sharpe ratio.

    ``p_value_positive`` is the bootstrap share of replications with a Sharpe at
    or below zero — the one-sided probability that the observed edge is noise.
    """
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < 50:
        return {"sharpe": float("nan"), "lower": float("nan"), "upper": float("nan")}

    rng = np.random.default_rng(seed)
    # A block roughly one trading month long at the study frequency.
    mean_block = mean_block or max(5.0, bars_per_year / 260 * 20)

    replications = np.empty(n_samples)
    for i in range(n_samples):
        sample = values[stationary_bootstrap_indices(len(values), mean_block, rng)]
        replications[i] = _sharpe(sample, bars_per_year)

    replications = replications[np.isfinite(replications)]
    alpha = 1.0 - confidence_level

    return {
        "sharpe": _sharpe(values, bars_per_year),
        "lower": float(np.quantile(replications, alpha / 2)),
        "upper": float(np.quantile(replications, 1 - alpha / 2)),
        "bootstrap_mean": float(replications.mean()),
        "bootstrap_std": float(replications.std(ddof=1)),
        "p_value_positive": float((replications <= 0).mean()),
        "confidence_level": confidence_level,
        "samples": int(len(replications)),
    }


def probabilistic_sharpe_ratio(
    returns: pd.Series, benchmark_sharpe: float = 0.0, bars_per_year: int = 6240
) -> float:
    """Probability that the true Sharpe ratio exceeds ``benchmark_sharpe``.

    Corrects the standard error of the Sharpe estimator for skew and excess
    kurtosis. Financial returns have both, and ignoring them overstates
    significance in exactly the direction that flatters the strategy.
    """
    values = returns.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 30:
        return float("nan")

    volatility = values.std(ddof=1)
    if volatility <= 0:
        return float("nan")

    # Per-bar Sharpe; the benchmark arrives annualised.
    observed = values.mean() / volatility
    target = benchmark_sharpe / np.sqrt(bars_per_year)

    skew = stats.skew(values)
    excess_kurtosis = stats.kurtosis(values, fisher=True)

    denominator = np.sqrt(
        1.0 - skew * observed + (excess_kurtosis / 4.0) * observed**2
    )
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")

    z = (observed - target) * np.sqrt(n - 1) / denominator
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    trial_sharpe_std: float | None = None,
    bars_per_year: int = 6240,
) -> float:
    """Probability the Sharpe survives after correcting for selection bias.

    ``n_trials`` is the number of configurations evaluated before this one was
    reported. Searching a hundred variants and reporting the best of them
    produces an impressive Sharpe from pure noise; this discounts for that.
    """
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < 30 or n_trials < 1:
        return float("nan")

    if trial_sharpe_std is None or not np.isfinite(trial_sharpe_std) or trial_sharpe_std <= 0:
        # Without an observed spread across trials, fall back to the standard
        # error of a single Sharpe estimate as the scale of the search.
        trial_sharpe_std = 1.0 / np.sqrt(len(values))

    euler_mascheroni = 0.5772156649015329
    if n_trials == 1:
        expected_max = 0.0
    else:
        # Expected maximum of n_trials draws from a standard normal.
        expected_max = trial_sharpe_std * (
            (1 - euler_mascheroni) * stats.norm.ppf(1 - 1.0 / n_trials)
            + euler_mascheroni * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        )

    return probabilistic_sharpe_ratio(
        returns, benchmark_sharpe=expected_max * np.sqrt(bars_per_year), bars_per_year=bars_per_year
    )


def stability_across_folds(fold_metrics: pd.DataFrame, metric: str = "sharpe") -> dict[str, float]:
    """Consistency of a metric across the walk-forward folds.

    A strategy that earns its entire Sharpe in one fold and loses money in the
    other five has not found a persistent edge; it has found one good year. The
    fraction of positive folds is the blunt version of that test, and the t
    statistic on the fold means is the formal one.
    """
    if metric not in fold_metrics.columns:
        return {}

    values = fold_metrics[metric].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {}

    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0

    result = {
        "mean": mean,
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
        "positive_fold_share": float((values > 0).mean()),
        "folds": int(n),
    }
    if n > 1 and std > 0:
        t_statistic = mean / (std / np.sqrt(n))
        result["t_statistic"] = float(t_statistic)
        result["p_value"] = float(1.0 - stats.t.cdf(t_statistic, df=n - 1))
    return result


def significance_report(
    returns: pd.Series,
    fold_metrics: pd.DataFrame | None = None,
    n_trials: int = 1,
    bars_per_year: int = 6240,
    n_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 7,
) -> dict[str, object]:
    """Every significance check the study reports, in one call."""
    interval = bootstrap_sharpe_interval(
        returns,
        bars_per_year=bars_per_year,
        n_samples=n_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    trial_std = None
    if fold_metrics is not None and "sharpe" in fold_metrics.columns and len(fold_metrics) > 1:
        trial_std = float(fold_metrics["sharpe"].std(ddof=1))

    report: dict[str, object] = dict(interval)
    report["probabilistic_sharpe"] = probabilistic_sharpe_ratio(
        returns, benchmark_sharpe=0.0, bars_per_year=bars_per_year
    )
    report["deflated_sharpe"] = deflated_sharpe_ratio(
        returns, n_trials=n_trials, trial_sharpe_std=trial_std, bars_per_year=bars_per_year
    )
    report["n_trials"] = int(n_trials)
    if fold_metrics is not None and not fold_metrics.empty:
        report["fold_stability"] = stability_across_folds(fold_metrics, "sharpe")
    return report
