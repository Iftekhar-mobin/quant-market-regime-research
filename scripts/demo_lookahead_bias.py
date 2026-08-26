"""Demonstrate look-ahead bias by measuring it.

Look-ahead bias is the most expensive mistake in quantitative research: a
feature that quietly uses information from after the bar it is reported on. The
resulting backtest is not merely optimistic, it is meaningless, and nothing in
the output announces the problem.

This script introduces two leaks on purpose and prices them, so the damage is a
number rather than a warning.

    honest   the pipeline as shipped
    swing    swing pivots reported on the bar they occurred, not the bar that
             confirms them - a 5-bar leak through 5 of ~84 features
    future   the price move over the next 5 bars handed to the model directly

Usage
-----
    python scripts/demo_lookahead_bias.py                 # all three arms
    python scripts/demo_lookahead_bias.py --arm swing     # just one
    python scripts/demo_lookahead_bias.py --symbol GOLD --folds 3

Nothing under ``src/`` is modified. The leaks are installed at runtime by
rebinding a module attribute and removed again afterwards, so the repository is
in exactly the state it started in when the script exits.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

import qmr.experiments.runner as runner_module  # noqa: E402
import qmr.features.indicators as indicators_module  # noqa: E402
import qmr.features.pipeline as pipeline_module  # noqa: E402
from qmr.config import Config  # noqa: E402
from qmr.experiments.runner import run_experiment  # noqa: E402
from qmr.logging_utils import configure_logging  # noqa: E402

ARMS = ("honest", "swing", "future")


# ---------------------------------------------------------------------------
# The two leaks
# ---------------------------------------------------------------------------
def leaky_swing_points(
    high: pd.Series, low: pd.Series, left: int = 5, right: int = 5
) -> pd.DataFrame:
    """`swing_points` with the confirmation delay removed.

    The shipped version reports a pivot on the bar that *confirms* it, because a
    local top is not knowable until the bars after it have printed. This version
    marks the pivot in place - the conventional ``argrelextrema`` approach, and
    the reason so many published backtests cannot be reproduced live.

    Compare against ``qmr.features.indicators.swing_points``: the only change is
    that ``centre_high``/``centre_low`` are no longer shifted.
    """
    window = left + right + 1
    rolling_high = high.rolling(window).max()
    rolling_low = low.rolling(window).min()

    centre_high = high  # shipped version: high.shift(right)
    centre_low = low  # shipped version: low.shift(right)

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


def make_future_leaking_features(original_build_features):
    """Wrap `build_features` so it also hands the model the answer.

    ``close.shift(-5)`` is the price five bars from now. Dividing by today's
    close turns it into the forward return - the thing the model is being asked
    to predict, supplied as an input.
    """

    def build_features_with_leak(frame, config=None, warmup_bars=300):
        features = original_build_features(frame, config, warmup_bars)
        features["tomorrow_close"] = features["close"].shift(-5) / features["close"] - 1.0
        # shift(-5) leaves five empty rows at the end; drop them.
        return features.dropna()

    return build_features_with_leak


# ---------------------------------------------------------------------------
# Running one arm
# ---------------------------------------------------------------------------
def run_arm(arm: str, config: Config) -> dict[str, float]:
    """Install the named leak, run one study, then restore the pipeline."""
    original_swing = indicators_module.swing_points
    original_build = pipeline_module.build_features
    original_runner_build = runner_module.build_features

    try:
        if arm == "swing":
            # `_structure_block` calls `ind.swing_points(...)`, so rebinding the
            # attribute on the module is enough for the pipeline to pick it up.
            indicators_module.swing_points = leaky_swing_points
        elif arm == "future":
            # The runner did `from qmr.features import build_features`, which
            # bound its own name, so both references have to be replaced.
            leaked = make_future_leaking_features(original_build)
            pipeline_module.build_features = leaked
            runner_module.build_features = leaked

        result = run_experiment(config)

        importance = result.feature_importance
        top_feature = str(importance.index[0]) if importance is not None and len(importance) else "n/a"

        return {
            "arm": arm,
            "precision": result.classification["directional_precision"],
            "sharpe": result.metrics["sharpe"],
            "cagr": result.metrics["cagr"],
            "max_drawdown": result.metrics["max_drawdown"],
            "trades": result.metrics["trades"],
            "top_feature": top_feature,
        }
    finally:
        # Always restore, even if the study raised.
        indicators_module.swing_points = original_swing
        pipeline_module.build_features = original_build
        runner_module.build_features = original_runner_build


DESCRIPTIONS = {
    "honest": "the pipeline as shipped",
    "swing": "5-bar leak through 5 of ~84 features",
    "future": "next 5 bars handed to the model directly",
}


def report(rows: list[dict[str, float]]) -> None:
    print("\n" + "=" * 84)
    print("LOOK-AHEAD BIAS: what each leak is worth")
    print("=" * 84)
    print(
        f"\n{'Arm':<10}{'Precision':>11}{'Sharpe':>10}{'CAGR':>10}"
        f"{'Max DD':>10}{'Trades':>9}   Top feature"
    )
    print("-" * 84)
    for row in rows:
        print(
            f"{row['arm']:<10}"
            f"{row['precision'] * 100:>10.1f}%"
            f"{row['sharpe']:>+10.2f}"
            f"{row['cagr'] * 100:>9.1f}%"
            f"{row['max_drawdown'] * 100:>9.1f}%"
            f"{row['trades']:>9,.0f}"
            f"   {row['top_feature']}"
        )
    print("-" * 84)
    for arm, text in DESCRIPTIONS.items():
        if any(r["arm"] == arm for r in rows):
            print(f"  {arm:<8} {text}")

    by_arm = {row["arm"]: row for row in rows}
    if "honest" in by_arm and "swing" in by_arm:
        gain = by_arm["swing"]["sharpe"] - by_arm["honest"]["sharpe"]
        print(
            f"\nThe subtle leak is worth {gain:+.2f} Sharpe. Notice how ordinary that\n"
            f"looks. It does not produce an absurd equity curve - it just makes the\n"
            f"results a little better, which is exactly why you would believe it.\n"
            f"Only 5 of roughly 84 features were affected, and only by 5 bars."
        )
    if "honest" in by_arm and "future" in by_arm:
        gain = by_arm["future"]["sharpe"] - by_arm["honest"]["sharpe"]
        print(
            f"\nThe blatant leak is worth {gain:+.2f} Sharpe. If a retail backtest ever\n"
            f"shows you a Sharpe above 3 with a single-digit drawdown, this is very\n"
            f"often the reason. Note which feature the model leaned on hardest."
        )
    leaks_run = [arm for arm in ("swing", "future") if arm in by_arm]
    if leaks_run:
        opening = (
            "Neither leak announced itself."
            if len(leaks_run) > 1
            else "The leak did not announce itself."
        )
        print(
            f"\n{opening} No error, no warning, no obviously\n"
            "broken number - only better results. That is what makes it the most\n"
            "expensive mistake in this field, and why every feature in\n"
            "src/qmr/features is written to use bar t and earlier only.\n"
        )
    else:
        print(
            "\nThat is the control arm on its own. Run without --arm to add the two\n"
            "leaked arms and see what each one is worth.\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm", choices=[*ARMS, "all"], default="all", help="which arm(s) to run"
    )
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--start", default=None, help="study start date, e.g. 2020-01-01")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--verbose", action="store_true", help="show the full study log")
    args = parser.parse_args()

    configure_logging(logging.INFO if args.verbose else logging.ERROR)

    config = Config.load().with_overrides(
        {
            "data.symbol": args.symbol,
            "data.timeframe": args.timeframe,
            "data.start": args.start,
            "model.name": args.model,
            # The control arm has to be free of regimes, or the regime layer
            # becomes a second thing changing between rows.
            "regime.method": "none",
            "validation.n_folds": args.folds,
            "evaluation.bootstrap_samples": 100,
        }
    )

    arms = list(ARMS) if args.arm == "all" else [args.arm]
    print(
        f"\n{args.symbol} {args.timeframe} | {args.model} | {args.folds} folds"
        f" | {len(arms)} arm(s)\nEach arm is a full walk-forward study; expect a"
        f" minute or so per arm.\n"
    )

    rows = []
    for arm in arms:
        print(f"  running '{arm}' ...", flush=True)
        try:
            rows.append(run_arm(arm, config))
        except Exception as exc:  # one bad arm should not lose the others
            print(f"  arm '{arm}' failed: {exc}")

    if not rows:
        print("No arm completed. Re-run with --verbose to see why.")
        return 1

    report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
