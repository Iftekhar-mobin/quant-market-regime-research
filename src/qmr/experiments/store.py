"""Persistence of experiment runs.

Each run gets its own directory under ``experiments/<run_id>/`` holding the
resolved configuration and every table needed to rebuild the report:

    config.yaml            the exact configuration that produced this run
    summary.json           headline metrics, significance, benchmark comparison
    predictions.parquet    per-bar out-of-sample record
    equity.parquet         equity curve, drawdown and benchmark
    fold_metrics.csv       per-fold economics
    fold_layout.csv        the walk-forward schedule with real timestamps
    trades.csv             round trips
    regime_table.csv       regime characterisation
    regime_performance.csv strategy performance by regime
    feature_importance.csv averaged model-native importance
    threshold_curve.csv    precision/coverage trade-off
    confusion.csv          classification confusion matrix

Plain formats on purpose: a study that can only be read back by the code that
wrote it is not reproducible in any useful sense.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qmr.config import Config
from qmr.experiments.runner import ExperimentResult
from qmr.logging_utils import get_logger
from qmr.paths import EXPERIMENT_DIR

log = get_logger(__name__)


@dataclass
class ExperimentRecord:
    """Lightweight index entry, read without loading the full run."""

    run_id: str
    path: Path
    created_at: str
    summary: dict[str, Any]

    @property
    def symbol(self) -> str:
        return str(self.summary.get("symbol", "?"))

    @property
    def model(self) -> str:
        return str(self.summary.get("model", "?"))

    @property
    def regime_method(self) -> str:
        return str(self.summary.get("regime_method", "?"))


def _json_safe(value: Any) -> Any:
    """Convert numpy and pandas scalars into something json can hold."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def _write_table(frame: pd.DataFrame | pd.Series | None, path: Path) -> None:
    if frame is None or len(frame) == 0:
        return
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name="value")
    if path.suffix == ".parquet":
        frame.to_parquet(path)
    else:
        frame.to_csv(path)


def save_experiment(result: ExperimentResult, root: Path | None = None) -> Path:
    """Write one run to disk and return its directory."""
    root = root or EXPERIMENT_DIR
    directory = root / result.run_id
    directory.mkdir(parents=True, exist_ok=True)

    result.config.save(directory / "config.yaml")

    summary = {
        "run_id": result.run_id,
        "created_at": result.created_at,
        "duration_seconds": result.duration_seconds,
        "label": result.label,
        "symbol": result.config.data.symbol,
        "timeframe": result.config.data.timeframe,
        "model": result.config.model.name,
        "regime_method": result.config.regime.method,
        "n_regimes": result.config.regime.n_regimes,
        "specialised_models": result.config.regime.specialised_models,
        "labeling": result.config.labeling.method,
        "decision_threshold": result.config.model.decision_threshold,
        "oos_bars": int(len(result.predictions)),
        "oos_start": result.predictions.index[0],
        "oos_end": result.predictions.index[-1],
        "metrics": result.metrics,
        "benchmark_metrics": result.benchmark_metrics,
        "classification": result.classification,
        "significance": result.significance,
        "baselines": result.baselines,
    }
    (directory / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )

    _write_table(result.predictions, directory / "predictions.parquet")
    _write_table(result.backtest.frame(), directory / "equity.parquet")
    _write_table(result.backtest.trades, directory / "trades.csv")
    _write_table(result.fold_metrics, directory / "fold_metrics.csv")
    _write_table(result.fold_layout, directory / "fold_layout.csv")
    _write_table(result.regime_table, directory / "regime_table.csv")
    _write_table(result.regime_performance, directory / "regime_performance.csv")
    _write_table(result.regime_transitions, directory / "regime_transitions.csv")
    _write_table(result.confusion, directory / "confusion.csv")
    _write_table(result.threshold_curve, directory / "threshold_curve.csv")
    _write_table(result.feature_importance, directory / "feature_importance.csv")
    _write_table(result.price, directory / "price.parquet")

    log.info("Saved experiment to %s", directory)
    return directory


def list_experiments(root: Path | None = None) -> list[ExperimentRecord]:
    """Every stored run, most recent first."""
    root = root or EXPERIMENT_DIR
    if not root.exists():
        return []

    records: list[ExperimentRecord] = []
    for directory in root.iterdir():
        summary_path = directory / "summary.json"
        if not directory.is_dir() or not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Skipping unreadable summary in %s", directory)
            continue
        records.append(
            ExperimentRecord(
                run_id=summary.get("run_id", directory.name),
                path=directory,
                created_at=summary.get("created_at", ""),
                summary=summary,
            )
        )

    return sorted(records, key=lambda r: r.created_at, reverse=True)


def experiments_frame(root: Path | None = None) -> pd.DataFrame:
    """The stored runs as a comparison table."""
    rows = []
    for record in list_experiments(root):
        metrics = record.summary.get("metrics", {}) or {}
        classification = record.summary.get("classification", {}) or {}
        significance = record.summary.get("significance", {}) or {}
        rows.append(
            {
                "Run": record.run_id,
                "Symbol": record.summary.get("symbol"),
                "Timeframe": record.summary.get("timeframe"),
                "Model": record.summary.get("model"),
                "Regimes": record.summary.get("regime_method"),
                "Sharpe": metrics.get("sharpe"),
                "CAGR": metrics.get("cagr"),
                "Max drawdown": metrics.get("max_drawdown"),
                "Profit factor": metrics.get("profit_factor"),
                "Trades": metrics.get("trades"),
                "Precision": classification.get("directional_precision"),
                "Deflated Sharpe": significance.get("deflated_sharpe"),
                "OOS bars": record.summary.get("oos_bars"),
                "Created": record.created_at[:19].replace("T", " "),
            }
        )
    return pd.DataFrame(rows)


def _read_table(path: Path, index_col: int | None = 0) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col=index_col)


def load_experiment(run_id: str, root: Path | None = None) -> dict[str, Any]:
    """Read one stored run back as plain tables.

    Returns a dictionary rather than an :class:`ExperimentResult` because the
    fitted models are deliberately not persisted: a study is reproduced by
    rerunning its configuration, not by unpickling an estimator whose library
    version has since moved on.
    """
    root = root or EXPERIMENT_DIR
    directory = root / run_id
    if not directory.exists():
        raise FileNotFoundError(f"No experiment named {run_id!r} under {root}")

    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))

    return {
        "run_id": run_id,
        "path": directory,
        "summary": summary,
        "config": Config.load(directory / "config.yaml"),
        "predictions": _read_table(directory / "predictions.parquet"),
        "equity": _read_table(directory / "equity.parquet"),
        "trades": _read_table(directory / "trades.csv", index_col=0),
        "fold_metrics": _read_table(directory / "fold_metrics.csv", index_col=0),
        "fold_layout": _read_table(directory / "fold_layout.csv", index_col=0),
        "regime_table": _read_table(directory / "regime_table.csv"),
        "regime_performance": _read_table(directory / "regime_performance.csv", index_col=0),
        "regime_transitions": _read_table(directory / "regime_transitions.csv"),
        "confusion": _read_table(directory / "confusion.csv"),
        "threshold_curve": _read_table(directory / "threshold_curve.csv", index_col=0),
        "feature_importance": _read_table(directory / "feature_importance.csv"),
        "price": _read_table(directory / "price.parquet"),
    }


def delete_experiment(run_id: str, root: Path | None = None) -> None:
    """Remove one stored run."""
    root = root or EXPERIMENT_DIR
    directory = root / run_id
    if directory.exists():
        shutil.rmtree(directory)
        log.info("Deleted experiment %s", run_id)
