"""Characterisation of detected regimes.

Detecting states is only half the work. A regime is only useful if it is
*persistent* (it lasts long enough to trade), *distinct* (forward returns and
risk differ across states) and *stable* (the same states keep reappearing out of
sample). The functions here measure exactly those three properties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qmr.backtest.metrics import annualisation_factor


def run_lengths(regimes: pd.Series) -> pd.DataFrame:
    """Contiguous runs of a single regime, with their start and end timestamps."""
    changed = regimes.ne(regimes.shift())
    run_id = changed.cumsum()

    grouped = regimes.groupby(run_id)
    return pd.DataFrame(
        {
            "regime": grouped.first().to_numpy(),
            "start": grouped.apply(lambda s: s.index[0]).to_numpy(),
            "end": grouped.apply(lambda s: s.index[-1]).to_numpy(),
            "bars": grouped.size().to_numpy(),
        }
    )


def regime_transition_matrix(regimes: pd.Series, normalise: bool = True) -> pd.DataFrame:
    """Bar-to-bar transition probabilities between regimes.

    The diagonal is the persistence of each state: a value near 1 means the
    market rarely leaves it from one bar to the next, which is the property a
    regime-conditioned strategy relies on.
    """
    current = regimes.iloc[:-1].to_numpy()
    following = regimes.iloc[1:].to_numpy()

    states = sorted(pd.unique(regimes))
    matrix = pd.DataFrame(0.0, index=states, columns=states)
    for source, target in zip(current, following, strict=True):
        matrix.loc[source, target] += 1.0

    if normalise:
        row_totals = matrix.sum(axis=1).replace(0.0, np.nan)
        matrix = matrix.div(row_totals, axis=0).fillna(0.0)

    matrix.index.name = "from"
    matrix.columns.name = "to"
    return matrix


def regime_profile(
    frame: pd.DataFrame,
    regimes: pd.Series,
    descriptors: list[str] | None = None,
    bars_per_year: int = 6240,
) -> pd.DataFrame:
    """One row per regime: how often it occurs, how long it lasts, and what the
    market does inside it.

    The return statistics are for buy-and-hold *within* the regime. They answer
    "does this state carry a directional edge on its own", which is the baseline
    any regime-conditioned model has to improve on.
    """
    descriptors = descriptors or [
        c
        for c in ("trend_strength", "vol_percentile", "momentum_score", "mean_reversion_score")
        if c in frame.columns
    ]

    aligned = regimes.reindex(frame.index).dropna().astype(int)
    prices = frame.loc[aligned.index, "close"]
    bar_return = np.log(prices).diff().fillna(0.0)

    runs = run_lengths(aligned)
    mean_run = runs.groupby("regime")["bars"].mean()

    rows = []
    for state, group in bar_return.groupby(aligned):
        share = len(group) / len(aligned)
        mean_bar = group.mean()
        volatility = group.std(ddof=0)
        annual_factor = annualisation_factor(bars_per_year)

        row = {
            "regime": int(state),
            "bars": int(len(group)),
            "share": share,
            "mean_run_bars": float(mean_run.get(state, np.nan)),
            "mean_return_bps": float(mean_bar * 1e4),
            "volatility_bps": float(volatility * 1e4),
            "annualised_return": float(mean_bar * bars_per_year),
            "annualised_volatility": float(volatility * annual_factor),
            "return_per_unit_risk": float(mean_bar / volatility * annual_factor)
            if volatility > 0
            else np.nan,
            "positive_bar_share": float((group > 0).mean()),
        }
        for descriptor in descriptors:
            row[descriptor] = float(frame.loc[aligned.index, descriptor][aligned == state].mean())
        rows.append(row)

    return pd.DataFrame(rows).set_index("regime")


def regime_summary(
    frame: pd.DataFrame,
    regimes: pd.Series,
    names: dict[int, str] | None = None,
    bars_per_year: int = 6240,
) -> pd.DataFrame:
    """Display-ready regime table for the research console."""
    profile = regime_profile(frame, regimes, bars_per_year=bars_per_year)
    names = names or {}

    table = pd.DataFrame(
        {
            "Regime": [names.get(int(i), f"Regime {i}") for i in profile.index],
            "Bars": profile["bars"],
            "Share": (profile["share"] * 100).round(1),
            "Avg run (bars)": profile["mean_run_bars"].round(1),
            "Return/bar (bps)": profile["mean_return_bps"].round(2),
            "Vol/bar (bps)": profile["volatility_bps"].round(2),
            "Ann. return": (profile["annualised_return"] * 100).round(1),
            "Ann. volatility": (profile["annualised_volatility"] * 100).round(1),
            "Return/risk": profile["return_per_unit_risk"].round(2),
        }
    )
    table.index.name = "id"
    return table


def regime_stability(train_regimes: pd.Series, test_regimes: pd.Series) -> pd.DataFrame:
    """Compare regime frequencies between two windows.

    A detector whose state frequencies shift wildly from the training window to
    the test window has found an artefact of the training period, not a
    recurring market state.
    """
    train_share = train_regimes.value_counts(normalize=True).sort_index()
    test_share = test_regimes.value_counts(normalize=True).sort_index()

    combined = pd.DataFrame({"train_share": train_share, "test_share": test_share}).fillna(0.0)
    combined["drift"] = (combined["test_share"] - combined["train_share"]).abs()
    return combined


def label_regime_series(regimes: pd.Series, names: dict[int, str]) -> pd.Series:
    """Map integer regime codes onto their readable names."""
    return regimes.map(lambda code: names.get(int(code), f"Regime {code}"))
