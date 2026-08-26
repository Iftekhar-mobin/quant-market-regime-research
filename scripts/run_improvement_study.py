"""Search for a configuration that clears zero, and log every attempt.

The point of this script is not to find a good number. It is to find a good
number *and keep an honest record of how many were tried*, because those two
facts together are what a result means.

Searching many configurations and reporting the best one is how most published
retail strategies are produced, and it is why so few survive. The mechanism is
simple: with enough arms, the best of them looks good on noise alone. The
defences here are:

* **Every arm is written to `reports/improvement_ablation.csv`**, including the
  ones that failed. The trial count is therefore auditable rather than implied.
* **The deflated Sharpe ratio** discounts each result by the number of arms
  actually run, not by the number the author chose to mention.
* **A holdout stage** re-runs the winner on instruments the search never
  touched. An edge that only exists on the instrument it was tuned on is not an
  edge.

Usage
-----
    python scripts/run_improvement_study.py                # the full programme
    python scripts/run_improvement_study.py --stage levers # one stage only
    python scripts/run_improvement_study.py --holdout      # validate the winner
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from qmr.config import Config  # noqa: E402
from qmr.experiments.runner import run_experiment  # noqa: E402
from qmr.logging_utils import configure_logging, get_logger  # noqa: E402
from qmr.paths import REPORT_DIR  # noqa: E402

log = get_logger("improvement_study")

LEDGER = REPORT_DIR / "improvement_ablation.csv"

# The search instrument. Everything else is held out.
SEARCH = {"data.symbol": "EURUSD", "data.timeframe": "H1", "data.start": "2020-01-01"}

COMMON = {
    "validation.n_folds": 3,
    "regime.method": "none",
    "model.name": "random_forest",
    "evaluation.bootstrap_samples": 300,
}

LIQUID_HOURS = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def _arms() -> dict[str, list[tuple[str, dict]]]:
    """Every configuration tried, grouped into stages.

    Each stage answers one question, and each arm changes as little as possible
    from its own reference so the attribution is clean.
    """
    return {
        # Stage 1: does any single execution lever move the needle?
        "levers": [
            ("baseline", {}),
            ("session filter", {"backtest.session_hours": LIQUID_HOURS}),
            ("top-20 features", {"model.top_k_features": 20}),
            ("volatility target 10%", {"backtest.volatility_target": 0.10}),
            ("threshold 0.55", {"model.decision_threshold": 0.55}),
            ("horizon 48", {"labeling.horizon": 48, "backtest.min_holding_bars": 48,
                            "validation.embargo_bars": 96}),
        ],
        # Stage 2: change the question the learner is asked.
        "meta": [
            ("meta (MA crossover)", {"labeling.method": "meta"}),
            ("meta + session", {"labeling.method": "meta",
                                "backtest.session_hours": LIQUID_HOURS}),
            ("meta + top-20", {"labeling.method": "meta", "model.top_k_features": 20}),
            ("meta + session + top-20", {"labeling.method": "meta",
                                         "backtest.session_hours": LIQUID_HOURS,
                                         "model.top_k_features": 20}),
            ("meta, Donchian primary", {"labeling.method": "meta",
                                        "labeling.primary": "donchian_breakout"}),
            ("meta, ADX-filtered primary", {"labeling.method": "meta",
                                            "labeling.primary": "volatility_filtered_trend"}),
            ("meta, RSI primary", {"labeling.method": "meta",
                                   "labeling.primary": "rsi_mean_reversion"}),
        ],
        # Stage 3: tune the most promising structure.
        "tune": [
            ("meta best + threshold 0.55", {"labeling.method": "meta",
                                            "backtest.session_hours": LIQUID_HOURS,
                                            "model.decision_threshold": 0.55}),
            ("meta best + threshold 0.60", {"labeling.method": "meta",
                                            "backtest.session_hours": LIQUID_HOURS,
                                            "model.decision_threshold": 0.60}),
            ("meta best + xgboost", {"labeling.method": "meta",
                                     "backtest.session_hours": LIQUID_HOURS,
                                     "model.name": "xgboost"}),
            ("meta best + lightgbm", {"labeling.method": "meta",
                                      "backtest.session_hours": LIQUID_HOURS,
                                      "model.name": "lightgbm"}),
            ("meta best + logistic", {"labeling.method": "meta",
                                      "backtest.session_hours": LIQUID_HOURS,
                                      "model.name": "logistic"}),
            ("meta best + kmeans regimes", {"labeling.method": "meta",
                                            "backtest.session_hours": LIQUID_HOURS,
                                            "regime.method": "kmeans"}),
            ("meta best + horizon 48", {"labeling.method": "meta",
                                        "backtest.session_hours": LIQUID_HOURS,
                                        "labeling.horizon": 48,
                                        "backtest.min_holding_bars": 48,
                                        "validation.embargo_bars": 96}),
        ],
        # Stage 5: the leading structure, measured better rather than tuned more.
        # Adding arms deepens the multiple-testing problem; these mostly reduce
        # the variance of the estimate instead of searching for a new one.
        "consolidate": [
            ("meta + top-20, 5 seeds", {"labeling.method": "meta",
                                        "model.top_k_features": 20,
                                        "model.n_seeds": 5}),
            ("meta + top-20, 6 folds", {"labeling.method": "meta",
                                        "model.top_k_features": 20,
                                        "validation.n_folds": 6}),
            ("meta + top-20, from 2016", {"labeling.method": "meta",
                                          "model.top_k_features": 20,
                                          "data.start": "2016-01-01",
                                          "validation.n_folds": 6}),
            ("meta + top-20 + vol target", {"labeling.method": "meta",
                                            "model.top_k_features": 20,
                                            "backtest.volatility_target": 0.10}),
            ("meta + top-20, 5 seeds, 6 folds, 2016", {"labeling.method": "meta",
                                                       "model.top_k_features": 20,
                                                       "model.n_seeds": 5,
                                                       "data.start": "2016-01-01",
                                                       "validation.n_folds": 6}),
        ],
        # Stage 4: slower timeframes, where the cost per unit of move is lower.
        "timeframe": [
            ("meta on H4", {"labeling.method": "meta", "data.timeframe": "H4",
                            "data.start": "2010-01-01", "labeling.horizon": 12,
                            "backtest.min_holding_bars": 12,
                            "validation.embargo_bars": 24}),
            ("meta on D1", {"labeling.method": "meta", "data.timeframe": "D1",
                            "data.start": "2005-01-01", "labeling.horizon": 10,
                            "backtest.min_holding_bars": 10,
                            "validation.embargo_bars": 20}),
            ("meta on H4 + session", {"labeling.method": "meta", "data.timeframe": "H4",
                                      "data.start": "2010-01-01", "labeling.horizon": 12,
                                      "backtest.min_holding_bars": 12,
                                      "validation.embargo_bars": 24,
                                      "backtest.session_hours": LIQUID_HOURS}),
        ],
    }


def run_one(name: str, stage: str, overrides: dict) -> dict:
    """Run one arm and return a ledger row (whether or not it succeeded)."""
    settings = {**SEARCH, **COMMON, **overrides}
    row = {
        "stage": stage,
        "arm": name,
        "symbol": settings["data.symbol"],
        "timeframe": settings["data.timeframe"],
        "changes": "; ".join(f"{k}={v}" for k, v in overrides.items()) or "(reference)",
    }
    started = time.perf_counter()
    try:
        result = run_experiment(Config.load().with_overrides(settings))
        row.update(
            {
                "sharpe": round(result.metrics["sharpe"], 4),
                "cagr": round(result.metrics["cagr"], 5),
                "max_drawdown": round(result.metrics["max_drawdown"], 4),
                "profit_factor": round(result.metrics["profit_factor"], 4),
                "trades": int(result.metrics["trades"]),
                "exposure": round(result.metrics["exposure"], 4),
                "precision": round(result.classification["directional_precision"], 4),
                "oos_bars": int(len(result.predictions)),
                "sharpe_ci_low": round(result.significance.get("lower", float("nan")), 4),
                "sharpe_ci_high": round(result.significance.get("upper", float("nan")), 4),
                "deflated_sharpe": round(result.significance.get("deflated_sharpe", float("nan")), 4),
                "benchmark_sharpe": round(result.benchmark_metrics["sharpe"], 4),
                "status": "ok",
            }
        )
    except Exception as exc:  # a failed arm is still a trial and still gets logged
        row.update({"status": f"failed: {exc}"[:160]})
        log.warning("%s failed: %s", name, exc)

    row["seconds"] = round(time.perf_counter() - started, 1)
    return row


def append_to_ledger(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if LEDGER.exists():
        frame = pd.concat([pd.read_csv(LEDGER), frame], ignore_index=True)
    frame.to_csv(LEDGER, index=False)
    return frame


def show(frame: pd.DataFrame) -> None:
    ok = frame[frame["status"] == "ok"].sort_values("sharpe", ascending=False)
    if ok.empty:
        print("\nNo arm completed.")
        return

    print("\n" + "=" * 108)
    print(f"IMPROVEMENT LEDGER  -  {len(frame)} arms attempted, {len(ok)} completed")
    print("=" * 108)
    print(
        f"{'Arm':<32}{'Sharpe':>9}{'95% CI':>18}{'Deflated':>10}"
        f"{'CAGR':>9}{'MaxDD':>9}{'Trades':>8}{'Prec':>8}"
    )
    print("-" * 108)
    for _, r in ok.iterrows():
        ci = f"[{r['sharpe_ci_low']:+.2f},{r['sharpe_ci_high']:+.2f}]"
        print(
            f"{r['arm'][:31]:<32}{r['sharpe']:>+9.2f}{ci:>18}{r['deflated_sharpe']:>10.3f}"
            f"{r['cagr'] * 100:>8.1f}%{r['max_drawdown'] * 100:>8.1f}%"
            f"{r['trades']:>8,}{r['precision'] * 100:>7.1f}%"
        )
    print("-" * 108)

    positive = ok[ok["sharpe"] > 0]
    expected = len(ok) * 0.5
    print(f"\n{len(positive)} of {len(ok)} arms have a positive Sharpe.")
    if len(positive) < expected:
        print(
            f"If every arm had a true Sharpe of zero, about {expected:.0f} would have landed\n"
            f"above zero by chance. Only {len(positive)} did. The population of configurations\n"
            f"is therefore centred below zero - these are not noisy draws around a\n"
            f"break-even strategy, they are draws from a losing one."
        )
    else:
        print(
            f"About {expected:.0f} would land above zero by chance alone if every arm had a\n"
            f"true Sharpe of zero, so the count proves nothing on its own. The\n"
            f"confidence intervals are what matter."
        )
    survivors = ok[(ok["sharpe_ci_low"] > 0)]
    if survivors.empty:
        print(
            "No arm's 95% confidence interval excludes zero. Nothing here is yet\n"
            "distinguishable from luck, however good the point estimate looks."
        )
    else:
        print("Arms whose confidence interval excludes zero:")
        for _, r in survivors.iterrows():
            print(f"  {r['arm']}  Sharpe {r['sharpe']:+.2f}  deflated {r['deflated_sharpe']:.3f}")
    print(f"\nFull ledger: {LEDGER}\n")


# Instruments the search never touches. An edge that exists only on the
# instrument it was tuned on is a property of that sample, not of the market.
HOLDOUT = [
    ("GBPUSD", "H1"), ("GOLD", "H1"), ("USDJPY", "H1"), ("AUDUSD", "H1"),
    ("USDCAD", "H1"), ("USDCHF", "H1"), ("EURGBP", "H1"), ("GBPJPY", "H1"),
    ("EURJPY", "H1"), ("CADCHF", "H1"),
]


def run_holdout() -> int:
    """Re-run the best arm from the ledger on instruments the search never saw.

    This is the only test in the programme that the search cannot game. Every
    arm above was chosen by looking at EURUSD; if the winner is real it should
    survive on data that had no say in picking it, and if it was noise it will
    not.
    """
    if not LEDGER.exists():
        print("No ledger yet. Run the search first.")
        return 1

    ledger = pd.read_csv(LEDGER)
    completed = ledger[(ledger["status"] == "ok") & (ledger["stage"] != "holdout")]
    if completed.empty:
        print("No completed arms in the ledger.")
        return 1

    best = completed.sort_values("sharpe", ascending=False).iloc[0]
    print(f"\nBest arm on the search instrument: {best['arm']}  (Sharpe {best['sharpe']:+.3f})")
    print(f"Changes: {best['changes']}")
    print(f"Chosen from {len(ledger)} attempted arms.\n")

    overrides = {}
    if best["changes"] != "(reference)":
        for item in str(best["changes"]).split("; "):
            key, _, raw = item.partition("=")
            if raw in {"True", "False"}:
                value: object = raw == "True"
            elif raw.startswith("["):
                value = [int(v) for v in raw.strip("[]").split(", ") if v.strip()]
            else:
                try:
                    value = int(raw)
                except ValueError:
                    try:
                        value = float(raw)
                    except ValueError:
                        value = raw
            overrides[key] = value

    rows = []
    for symbol, timeframe in HOLDOUT:
        settings = {**SEARCH, **COMMON, **overrides}
        settings["data.symbol"] = symbol
        settings["data.timeframe"] = timeframe
        print(f"  holdout: {symbol} {timeframe} ...", flush=True)
        rows.append(run_one(f"HOLDOUT {symbol} {timeframe}", "holdout", settings))
        print(f"        -> {rows[-1].get('sharpe', 'failed')}", flush=True)

    append_to_ledger(rows)

    print("\n" + "=" * 72)
    print("HOLDOUT RESULT")
    print("=" * 72)
    for row in rows:
        if row["status"] != "ok":
            print(f"  {row['arm']}: {row['status']}")
            continue
        print(
            f"  {row['arm']:<26} Sharpe {row['sharpe']:+.3f}  "
            f"CI [{row['sharpe_ci_low']:+.2f}, {row['sharpe_ci_high']:+.2f}]  "
            f"trades {row['trades']:,}"
        )

    from scipy import stats

    completed_rows = [r for r in rows if r["status"] == "ok"]
    if not completed_rows:
        print("No holdout instrument completed.")
        return 1

    sharpes = pd.Series([r["sharpe"] for r in completed_rows])
    n = len(sharpes)
    k = int((sharpes > 0).sum())

    print(
        f"  {k} of {n} held-out instruments positive."
        f"  Median Sharpe {sharpes.median():+.3f},"
        f"  mean {sharpes.mean():+.3f}."
    )

    # Under the null that the configuration is worthless, each instrument is an
    # independent coin flip. This is the one test the search cannot influence.
    sign_p = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
    print(f"  Sign test vs a coin flip:   p = {sign_p:.4f}")

    if n > 1 and sharpes.std(ddof=1) > 0:
        t_stat = float(sharpes.mean() / (sharpes.std(ddof=1) / (n**0.5)))
        t_p = float(1 - stats.t.cdf(t_stat, df=n - 1))
        print(f"  Cross-sectional t-test:     t = {t_stat:.2f}, p = {t_p:.4f}")

    print()
    if sign_p < 0.05:
        print(
            "The configuration is positive on significantly more held-out "
            "instruments than chance would produce. It was selected on EURUSD "
            "alone and none of these had any say in choosing it, so this is "
            "genuine out-of-sample evidence rather than a restatement of the "
            "search."
        )
        print(
            "Caveats that still apply: these instruments are correlated with one "
            "another, so the effective sample is smaller than the count suggests; "
            "and it is one strategy family, one asset class, one broker's prices "
            "and one cost assumption. A result worth pursuing, not a result worth "
            "trading."
        )
    elif k > n / 2:
        print(
            "More held-out instruments are positive than negative, but not by "
            "enough to rule out chance. Suggestive, not conclusive."
        )
    else:
        print(
            "The winner did not survive out of sample. The honest reading is that "
            "it was the best of many draws on EURUSD rather than a property of the "
            "market, which is exactly what the holdout is for."
        )
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=[*_arms(), "all"], default="all", help="which stage to run"
    )
    parser.add_argument(
        "--holdout", action="store_true", help="validate the ledger's best arm on held-out data"
    )
    parser.add_argument("--fresh", action="store_true", help="start a new ledger")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.INFO if args.verbose else logging.ERROR)

    if args.holdout:
        return run_holdout()

    if args.fresh and LEDGER.exists():
        LEDGER.unlink()

    stages = _arms()
    selected = stages if args.stage == "all" else {args.stage: stages[args.stage]}
    total = sum(len(v) for v in selected.values())
    print(f"Running {total} arms across {len(selected)} stage(s). Roughly a minute each.\n")

    rows = []
    done = 0
    for stage, arms in selected.items():
        print(f"--- stage: {stage} ---")
        for name, overrides in arms:
            done += 1
            print(f"  [{done}/{total}] {name} ...", flush=True)
            rows.append(run_one(name, stage, overrides))
            print(
                f"        -> {rows[-1].get('sharpe', 'failed')}"
                f"  ({rows[-1]['seconds']}s)",
                flush=True,
            )

    show(append_to_ledger(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
