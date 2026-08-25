"""The study, end to end.

One call to :func:`run_experiment` executes the whole research pipeline for a
single configuration and returns everything needed to judge the result:

    load -> features -> labels -> walk-forward
        (per fold: fit regimes -> fit model -> predict out of sample)
    -> stitch out-of-sample predictions -> backtest -> metrics
    -> regime breakdown -> significance -> baselines

The critical property is that *nothing* is fitted on data the fold is being
tested on. The regime detector, the feature scaler and the classifier are all
refitted inside each fold, on the training window alone, with an embargo
separating it from the test window. Every number the study reports is therefore
out of sample by construction, and the code path that would let it not be does
not exist.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from qmr.backtest.engine import BacktestResult, run_backtest, signals_from_probabilities
from qmr.backtest.metrics import performance_metrics
from qmr.config import Config
from qmr.data.catalog import bars_per_year as timeframe_bars_per_year
from qmr.data.loader import load_ohlcv
from qmr.evaluation.classification import (
    classification_summary,
    confusion_frame,
    threshold_sweep,
)
from qmr.evaluation.significance import significance_report
from qmr.features import build_features
from qmr.features.pipeline import feature_columns
from qmr.labeling import build_labels
from qmr.logging_utils import get_logger
from qmr.models.baselines import BASELINES, build_baseline_signals
from qmr.models.zoo import PROBABILITY_COLUMNS, build_model
from qmr.regimes import build_detector, regime_summary, regime_transition_matrix
from qmr.validation import describe_folds, walk_forward_splits

log = get_logger(__name__)

ProgressCallback = Callable[[float, str], None]


@dataclass
class ExperimentResult:
    """The complete record of one study."""

    run_id: str
    config: Config
    predictions: pd.DataFrame
    backtest: BacktestResult
    fold_metrics: pd.DataFrame
    fold_layout: pd.DataFrame
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    classification: dict[str, float]
    confusion: pd.DataFrame
    threshold_curve: pd.DataFrame
    feature_importance: pd.Series | None
    regime_table: pd.DataFrame
    regime_transitions: pd.DataFrame
    regime_performance: pd.DataFrame
    significance: dict[str, Any]
    baselines: dict[str, dict[str, float]]
    price: pd.DataFrame
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_seconds: float = 0.0

    @property
    def label(self) -> str:
        cfg = self.config
        regime = "no regimes" if cfg.regime.method == "none" else f"{cfg.regime.method} regimes"
        return f"{cfg.data.symbol} {cfg.data.timeframe} | {cfg.model.name} | {regime}"

    def headline(self) -> dict[str, float]:
        """The four numbers a reviewer looks at first."""
        return {
            "sharpe": self.metrics.get("sharpe", float("nan")),
            "cagr": self.metrics.get("cagr", float("nan")),
            "max_drawdown": self.metrics.get("max_drawdown", float("nan")),
            "directional_precision": self.classification.get(
                "directional_precision", float("nan")
            ),
        }


def _one_hot_regimes(regimes: pd.Series, n_regimes: int) -> pd.DataFrame:
    """Regime membership as indicator columns, with a stable column set.

    The columns are fixed by ``n_regimes`` rather than by what appeared in the
    fold, so the feature matrix has the same shape in every fold even when a
    regime is absent from one training window.
    """
    frame = pd.DataFrame(
        0.0,
        index=regimes.index,
        columns=[f"regime_{i}" for i in range(n_regimes)],
    )
    for state in range(n_regimes):
        frame.loc[regimes == state, f"regime_{state}"] = 1.0
    return frame


def _fit_fold_models(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    train_regimes: pd.Series,
    config: Config,
) -> dict[int, Any]:
    """Fit either one pooled model, or one specialised model per regime.

    Specialised models let each state learn its own relationship between
    features and outcome — the strong form of the regime hypothesis. The cost is
    that each model sees a fraction of the data, so a regime without enough
    training bars falls back to the pooled model rather than fitting noise.
    """
    pooled = build_model(config.model, seed=config.experiment.seed).fit(
        train_features, train_labels
    )
    if not config.regime.specialised_models:
        return {-1: pooled}

    models: dict[int, Any] = {-1: pooled}
    minimum_bars = max(500, len(train_features) // (config.regime.n_regimes * 4))

    for state in sorted(train_regimes.unique()):
        mask = train_regimes == state
        subset_features = train_features[mask]
        subset_labels = train_labels[mask]

        if len(subset_features) < minimum_bars or subset_labels.nunique() < 2:
            log.info(
                "Regime %d has only %d training bars; falling back to the pooled model",
                state,
                len(subset_features),
            )
            continue
        models[int(state)] = build_model(config.model, seed=config.experiment.seed).fit(
            subset_features, subset_labels
        )
    return models


def _predict_fold(
    models: dict[int, Any],
    test_features: pd.DataFrame,
    test_regimes: pd.Series,
) -> pd.DataFrame:
    """Score the test window with the pooled or the regime-specialised models."""
    if set(models) == {-1}:
        return models[-1].predict_proba(test_features)

    probabilities = pd.DataFrame(
        np.nan, index=test_features.index, columns=PROBABILITY_COLUMNS
    )
    for state in sorted(test_regimes.unique()):
        mask = (test_regimes == state).to_numpy()
        if not mask.any():
            continue
        model = models.get(int(state), models[-1])
        probabilities.loc[mask] = model.predict_proba(test_features[mask]).to_numpy()

    return probabilities.fillna(1.0 / len(PROBABILITY_COLUMNS))


def _regime_performance(
    predictions: pd.DataFrame,
    returns: pd.Series,
    regime_names: dict[int, str],
    bars_per_year: int,
) -> pd.DataFrame:
    """Strategy performance broken down by the regime in force at the time.

    This is the table the research question actually turns on: it shows whether
    the edge is spread evenly across market conditions or concentrated in one or
    two states — and therefore whether conditioning on the regime is buying
    anything.
    """
    aligned = returns.reindex(predictions.index).fillna(0.0)
    rows = []

    for state, group in aligned.groupby(predictions["regime"]):
        metrics = performance_metrics(group, bars_per_year=bars_per_year)
        subset = predictions.loc[group.index]
        traded = subset["prediction"] != 0
        rows.append(
            {
                "Regime": regime_names.get(int(state), f"Regime {int(state)}"),
                "Bars": len(group),
                "Share": len(group) / len(aligned),
                "Sharpe": metrics["sharpe"],
                "Return/bar (bps)": group.mean() * 1e4,
                "Max drawdown": metrics["max_drawdown"],
                "Hit rate": metrics["hit_rate"],
                "Signal rate": float(traded.mean()),
                "Directional precision": float(
                    (subset.loc[traded, "label"] == subset.loc[traded, "prediction"]).mean()
                )
                if traded.any()
                else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("Bars", ascending=False).reset_index(drop=True)


def run_experiment(
    config: Config,
    progress: ProgressCallback | None = None,
    run_id: str | None = None,
) -> ExperimentResult:
    """Execute one full study and return its results."""
    started = time.perf_counter()
    run_id = run_id or (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.data.symbol}_"
        f"{config.model.name}_{config.regime.method}"
    )

    def report(fraction: float, message: str) -> None:
        log.info(message)
        if progress is not None:
            progress(min(max(fraction, 0.0), 1.0), message)

    np.random.seed(config.experiment.seed)
    bars_per_year = timeframe_bars_per_year(config.data.timeframe)

    # -- 1. data ----------------------------------------------------------
    report(0.02, f"Loading {config.data.symbol} {config.data.timeframe} history")
    price = load_ohlcv(
        config.data.symbol,
        config.data.timeframe,
        start=config.data.start,
        end=config.data.end,
    )

    # -- 2. features ------------------------------------------------------
    report(0.10, "Building causal feature matrix")
    features = build_features(price, config.features, warmup_bars=config.data.warmup_bars)

    # -- 3. labels --------------------------------------------------------
    report(0.20, f"Labelling targets ({config.labeling.method})")
    label_result = build_labels(features, config.labeling)

    shared_index = features.index.intersection(label_result.labels.index)
    features = features.loc[shared_index]
    labels = label_result.labels.loc[shared_index]

    model_features = feature_columns(features)
    if not model_features:
        raise ValueError("No model features were produced; check features.blocks in the config.")

    log.info(
        "Study sample: %d bars, %d features, %s to %s",
        len(features),
        len(model_features),
        features.index[0].date(),
        features.index[-1].date(),
    )

    # -- 4. walk-forward --------------------------------------------------
    folds = walk_forward_splits(len(features), config.validation)
    fold_layout = describe_folds(folds, features.index)

    oos_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, float]] = []
    importance_frames: list[pd.Series] = []
    regime_names: dict[int, str] = {}
    detector_for_display = None

    for fold in folds:
        step = 0.25 + 0.55 * (fold.index / max(1, len(folds)))
        report(step, f"Fold {fold.index + 1} of {len(folds)}: fitting on {fold.train_size} bars")

        train_features_all = features.iloc[fold.train_slice]
        test_features_all = features.iloc[fold.test_slice]
        train_labels = labels.iloc[fold.train_slice]
        test_labels = labels.iloc[fold.test_slice]

        # 4a. regimes: fitted on the training window, applied to both.
        detector = build_detector(config.regime, random_state=config.experiment.seed)
        detector.fit(train_features_all)
        train_regimes = detector.predict(train_features_all)
        test_regimes = detector.predict(test_features_all)
        regime_names.update(detector.labels_)
        detector_for_display = detector

        # 4b. model inputs, optionally carrying the regime as a feature.
        train_X = train_features_all[model_features].copy()
        test_X = test_features_all[model_features].copy()

        if config.regime.as_model_feature and config.regime.method != "none":
            n_regimes = detector.n_regimes
            train_X = pd.concat([train_X, _one_hot_regimes(train_regimes, n_regimes)], axis=1)
            test_X = pd.concat([test_X, _one_hot_regimes(test_regimes, n_regimes)], axis=1)

        # 4c. fit and score.
        models = _fit_fold_models(train_X, train_labels, train_regimes, config)
        probabilities = _predict_fold(models, test_X, test_regimes)

        predicted = signals_from_probabilities(
            probabilities, config.model.decision_threshold
        ).astype(int)

        fold_frame = pd.DataFrame(
            {
                "fold": fold.index + 1,
                "label": test_labels.to_numpy(),
                "prediction": predicted.to_numpy(),
                "regime": test_regimes.to_numpy(),
                "close": test_features_all["close"].to_numpy(),
            },
            index=test_features_all.index,
        )
        for column in PROBABILITY_COLUMNS:
            fold_frame[f"p_{column}"] = probabilities[column].to_numpy()
        oos_frames.append(fold_frame)

        pooled = models[-1]
        importance = pooled.feature_importance()
        if importance is not None:
            importance_frames.append(importance)

        # Per-fold economics, on the fold's own slice of price history.
        fold_prices = price.loc[test_features_all.index]
        fold_backtest = run_backtest(
            fold_prices, predicted, config.backtest, bars_per_year=bars_per_year
        )
        fold_rows.append(
            {
                "fold": fold.index + 1,
                "train_bars": fold.train_size,
                "test_bars": fold.test_size,
                "start": test_features_all.index[0],
                "end": test_features_all.index[-1],
                "sharpe": fold_backtest.metrics["sharpe"],
                "total_return": fold_backtest.metrics["total_return"],
                "max_drawdown": fold_backtest.metrics["max_drawdown"],
                "trades": fold_backtest.metrics["trades"],
                "signal_rate": float((predicted != 0).mean()),
                "accuracy": float((predicted == test_labels.to_numpy()).mean()),
                "benchmark_sharpe": fold_backtest.benchmark_metrics["sharpe"],
            }
        )

    # -- 5. stitch the out-of-sample record -------------------------------
    report(0.82, "Assembling the out-of-sample record")
    predictions = pd.concat(oos_frames).sort_index()
    # Overlapping test windows would double-count bars; keep the first fold that
    # covered each timestamp so every bar is scored exactly once.
    predictions = predictions[~predictions.index.duplicated(keep="first")]

    fold_metrics = pd.DataFrame(fold_rows)

    # -- 6. economics -----------------------------------------------------
    report(0.86, "Backtesting the out-of-sample signal")
    oos_price = price.loc[predictions.index]
    backtest = run_backtest(
        oos_price, predictions["prediction"], config.backtest, bars_per_year=bars_per_year
    )

    # -- 7. classification and regime diagnostics -------------------------
    report(0.90, "Scoring predictions and regime breakdown")
    classification = classification_summary(predictions["label"], predictions["prediction"])
    confusion = confusion_frame(predictions["label"], predictions["prediction"])
    sweep = threshold_sweep(
        predictions[[f"p_{c}" for c in PROBABILITY_COLUMNS]].rename(
            columns={f"p_{c}": c for c in PROBABILITY_COLUMNS}
        ),
        predictions["label"],
    )

    oos_features = features.loc[predictions.index]
    regime_table = regime_summary(
        oos_features, predictions["regime"], regime_names, bars_per_year=bars_per_year
    )
    regime_transitions = regime_transition_matrix(predictions["regime"])
    regime_transitions.index = [regime_names.get(int(i), f"Regime {i}") for i in regime_transitions.index]
    regime_transitions.columns = [
        regime_names.get(int(i), f"Regime {i}") for i in regime_transitions.columns
    ]
    regime_performance = _regime_performance(
        predictions, backtest.returns, regime_names, bars_per_year
    )

    feature_importance = None
    if importance_frames:
        feature_importance = (
            pd.concat(importance_frames, axis=1).mean(axis=1).sort_values(ascending=False)
        )

    # -- 8. significance --------------------------------------------------
    report(0.94, "Running significance checks")
    significance = significance_report(
        backtest.returns,
        fold_metrics=fold_metrics,
        n_trials=max(1, len(folds)),
        bars_per_year=bars_per_year,
        n_samples=config.evaluation.bootstrap_samples,
        confidence_level=config.evaluation.confidence_level,
        seed=config.experiment.seed,
    )

    # -- 9. rule-based benchmarks over the same window --------------------
    report(0.97, "Pricing the rule-based benchmarks")
    baselines: dict[str, dict[str, float]] = {}
    for key in BASELINES:
        try:
            # Baselines are computed on the full history so their own warm-up
            # does not eat into the evaluation window, then sliced to it.
            signal = build_baseline_signals(key, price).reindex(predictions.index).fillna(0.0)
            baseline_result = run_backtest(
                oos_price, signal, config.backtest, bars_per_year=bars_per_year
            )
            baselines[key] = baseline_result.metrics
        except Exception as exc:  # a failing benchmark must not sink the study
            log.warning("Benchmark %s failed: %s", key, exc)

    duration = time.perf_counter() - started
    report(1.0, f"Study complete in {duration:.1f}s")

    if detector_for_display is not None:
        regime_names.update(detector_for_display.labels_)

    return ExperimentResult(
        run_id=run_id,
        config=config,
        predictions=predictions,
        backtest=backtest,
        fold_metrics=fold_metrics,
        fold_layout=fold_layout,
        metrics=backtest.metrics,
        benchmark_metrics=backtest.benchmark_metrics,
        classification=classification,
        confusion=confusion,
        threshold_curve=sweep,
        feature_importance=feature_importance,
        regime_table=regime_table,
        regime_transitions=regime_transitions,
        regime_performance=regime_performance,
        significance=significance,
        baselines=baselines,
        price=oos_price,
        duration_seconds=duration,
    )
