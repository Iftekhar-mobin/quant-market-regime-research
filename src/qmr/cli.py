"""Command-line entry point.

    qmr datasets                          list the discovered market history
    qmr run --set model.name=xgboost      run one study
    qmr compare --models xgboost,lightgbm run several and tabulate them
    qmr results                           list stored runs
    qmr show <run_id>                     print the report for one run
    qmr export-mt5 --symbols EURUSD       pull fresh history from MetaTrader 5
    qmr console                           launch the research console

Every command that produces a study writes it to ``experiments/<run_id>/`` in
plain formats, so the CLI and the console read exactly the same artefacts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from qmr.config import Config, parse_override
from qmr.logging_utils import configure_logging, get_logger
from qmr.paths import DEFAULT_CONFIG_PATH, LOG_DIR, PROJECT_ROOT, ensure_directories
from qmr.version import __version__

log = get_logger("qmr.cli")


def _print_frame(frame: pd.DataFrame, title: str | None = None) -> None:
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    if frame is None or frame.empty:
        print("(nothing to show)")
        return
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(frame.to_string(index=False))


def _load_config(args: argparse.Namespace) -> Config:
    config = Config.load(args.config)
    overrides = dict(parse_override(item) for item in (args.set or []))
    if overrides:
        config = config.with_overrides(overrides)
        log.info("Applied %d override(s): %s", len(overrides), overrides)
    return config


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_datasets(args: argparse.Namespace) -> int:
    from qmr.data.catalog import catalog_frame

    frame = catalog_frame()
    if frame.empty:
        print(
            "No market history found.\n"
            "Put CSV files named <SYMBOL>_<TIMEFRAME>_<YYYYMMDD>_<YYYYMMDD>.csv into "
            f"{PROJECT_ROOT / 'data' / 'raw'},\n"
            "or run `qmr export-mt5` with a MetaTrader 5 terminal installed."
        )
        return 1
    _print_frame(frame, f"{len(frame)} dataset(s)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from qmr.backtest.metrics import summarise_metrics
    from qmr.experiments import run_experiment, save_experiment

    config = _load_config(args)
    result = run_experiment(config)

    print(f"\n=== {result.label} ===")
    _print_frame(summarise_metrics(result.metrics), "Out-of-sample performance")
    _print_frame(
        result.fold_metrics[
            ["fold", "start", "end", "sharpe", "total_return", "max_drawdown", "trades"]
        ].round(4),
        "Per-fold results",
    )
    _print_frame(result.regime_performance.round(4), "Performance by regime")

    baselines = pd.DataFrame(result.baselines).T
    if not baselines.empty:
        _print_frame(
            baselines[["sharpe", "total_return", "max_drawdown", "trades"]]
            .round(4)
            .reset_index()
            .rename(columns={"index": "benchmark"}),
            "Rule-based benchmarks, same window",
        )

    significance = result.significance
    print("\nSignificance")
    print("-" * 12)
    print(
        f"Sharpe {significance.get('sharpe', float('nan')):.3f} "
        f"[{significance.get('lower', float('nan')):.3f}, "
        f"{significance.get('upper', float('nan')):.3f}] "
        f"at {significance.get('confidence_level', 0.95):.0%} confidence"
    )
    print(f"Probabilistic Sharpe : {significance.get('probabilistic_sharpe', float('nan')):.3f}")
    print(f"Deflated Sharpe      : {significance.get('deflated_sharpe', float('nan')):.3f}")

    if not args.no_save:
        directory = save_experiment(result)
        print(f"\nSaved to {directory}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from qmr.experiments import run_experiment, save_experiment

    config = _load_config(args)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]

    rows = []
    for model_name in models:
        for regime_method in regimes:
            label = f"{model_name} / {regime_method}"
            log.info("Running %s", label)
            try:
                variant = config.with_overrides(
                    {"model.name": model_name, "regime.method": regime_method}
                )
                result = run_experiment(variant)
            except Exception as exc:  # one bad arm must not lose the whole sweep
                log.error("%s failed: %s", label, exc)
                continue

            if not args.no_save:
                save_experiment(result)
            rows.append(
                {
                    "Model": model_name,
                    "Regimes": regime_method,
                    "Sharpe": round(result.metrics["sharpe"], 3),
                    "CAGR": round(result.metrics["cagr"], 4),
                    "Max drawdown": round(result.metrics["max_drawdown"], 4),
                    "Profit factor": round(result.metrics["profit_factor"], 3),
                    "Trades": int(result.metrics["trades"]),
                    "Precision": round(result.classification["directional_precision"], 4),
                    "Deflated Sharpe": round(result.significance.get("deflated_sharpe", float("nan")), 3),
                }
            )

    if not rows:
        print("Every arm of the comparison failed; see the log above.")
        return 1

    frame = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    _print_frame(frame, "Model comparison")

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"\nWritten to {path}")
    return 0


def cmd_results(args: argparse.Namespace) -> int:
    from qmr.experiments.store import experiments_frame

    frame = experiments_frame()
    if frame.empty:
        print("No stored experiments. Run `qmr run` first.")
        return 1
    _print_frame(frame.round(4), f"{len(frame)} stored experiment(s)")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    from qmr.backtest.metrics import summarise_metrics
    from qmr.experiments.store import load_experiment

    try:
        record = load_experiment(args.run_id)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    summary = record["summary"]
    print(f"\n=== {summary.get('label', args.run_id)} ===")
    print(f"Run       : {summary['run_id']}")
    print(f"Created   : {summary.get('created_at', '?')}")
    print(f"OOS window: {summary.get('oos_start')} to {summary.get('oos_end')}")

    _print_frame(summarise_metrics(summary.get("metrics", {})), "Out-of-sample performance")
    _print_frame(record.get("fold_metrics"), "Per-fold results")
    _print_frame(record.get("regime_performance"), "Performance by regime")
    return 0


def cmd_export_mt5(args: argparse.Namespace) -> int:
    from qmr.data.mt5_export import export_history

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]

    try:
        written = export_history(symbols, timeframes, bars=args.bars)
    except ImportError as exc:
        print(exc)
        return 1

    for path in written:
        print(f"wrote {path}")
    return 0 if written else 1


def cmd_console(args: argparse.Namespace) -> int:
    """Launch the Streamlit research console."""
    import subprocess

    app = PROJECT_ROOT / "app" / "main.py"
    if not app.exists():
        print(f"Console entry point not found at {app}")
        return 1

    command = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port)]
    print(f"Starting the research console on http://localhost:{args.port}")
    return subprocess.call(command)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qmr",
        description="Market-regime research framework for FX and commodity time series.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"qmr {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    parser.add_argument("--log-file", type=Path, default=None, help="also write the log here")

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_arguments(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="configuration file"
        )
        sub.add_argument(
            "--set",
            action="append",
            metavar="KEY=VALUE",
            help="override a configuration value, e.g. --set model.name=xgboost (repeatable)",
        )
        sub.add_argument("--no-save", action="store_true", help="do not persist the run")

    datasets = subparsers.add_parser("datasets", help="list the discovered market history")
    datasets.set_defaults(func=cmd_datasets)

    run = subparsers.add_parser("run", help="run one study")
    add_config_arguments(run)
    run.set_defaults(func=cmd_run)

    compare = subparsers.add_parser("compare", help="run several studies and tabulate them")
    add_config_arguments(compare)
    compare.add_argument(
        "--models", default="logistic,random_forest,xgboost", help="comma-separated model keys"
    )
    compare.add_argument(
        "--regimes", default="none,kmeans", help="comma-separated regime methods"
    )
    compare.add_argument("--output", type=Path, default=None, help="write the table to CSV")
    compare.set_defaults(func=cmd_compare)

    results = subparsers.add_parser("results", help="list stored experiments")
    results.set_defaults(func=cmd_results)

    show = subparsers.add_parser("show", help="print the report for one stored experiment")
    show.add_argument("run_id")
    show.set_defaults(func=cmd_show)

    export = subparsers.add_parser("export-mt5", help="pull history from a MetaTrader 5 terminal")
    export.add_argument("--symbols", default="EURUSD,GBPUSD,USDJPY,GOLD")
    export.add_argument("--timeframes", default="H1,H4,D1")
    export.add_argument("--bars", type=int, default=200_000)
    export.set_defaults(func=cmd_export_mt5)

    console = subparsers.add_parser("console", help="launch the research console")
    console.add_argument("--port", type=int, default=8501)
    console.set_defaults(func=cmd_console)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    ensure_directories()
    configure_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=args.log_file or (LOG_DIR / "qmr.log"),
    )

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        log.error("%s", exc, exc_info=args.verbose)
        if not args.verbose:
            print("\nRe-run with -v for the full traceback.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
