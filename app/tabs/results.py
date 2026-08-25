"""Results: the full report for one stored study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import theme


def _pick_run() -> str | None:
    from qmr.experiments.store import list_experiments

    records = list_experiments()
    if not records:
        return None

    options = [record.run_id for record in records]
    labels = {
        record.run_id: (
            f"{record.summary.get('label', record.run_id)}  ·  "
            f"Sharpe {record.summary.get('metrics', {}).get('sharpe', float('nan')):+.2f}  ·  "
            f"{record.created_at[:16].replace('T', ' ')}"
        )
        for record in records
    }

    default = st.session_state.get("last_run_id")
    index = options.index(default) if default in options else 0
    return st.selectbox(
        "Study", options, index=index, format_func=lambda k: labels[k], key="res_run"
    )


def _significance_verdict(significance: dict) -> tuple[str, str]:
    """A plain-English reading of the significance block."""
    lower = significance.get("lower")
    deflated = significance.get("deflated_sharpe")
    sharpe = significance.get("sharpe")

    if sharpe is None or not np.isfinite(sharpe):
        return "critical", "No usable return stream."
    if lower is not None and np.isfinite(lower) and lower > 0:
        if deflated is not None and np.isfinite(deflated) and deflated > 0.95:
            return "good", (
                "The confidence interval excludes zero and the Sharpe survives the "
                "correction for how many configurations were tried. This is as "
                "close to a real result as a single study gets."
            )
        return "warning", (
            "The confidence interval excludes zero, but the deflated Sharpe is "
            "not conclusive: some of the edge may be the search rather than the "
            "market. Confirm on another instrument before believing it."
        )
    if sharpe > 0:
        return "warning", (
            "Positive, but the confidence interval includes zero. On this sample "
            "the result is not distinguishable from luck."
        )
    return "critical", (
        "Negative out of sample. The honest conclusion is that this "
        "configuration has no edge — which is a finding, not a failure."
    )


def render(selection: dict, config) -> None:
    from qmr.backtest.metrics import summarise_metrics
    from qmr.experiments.store import load_experiment

    st.markdown(
        theme.lede(
            "The full out-of-sample report for one study: economics, per-fold "
            "stability, what the model learned, how it behaved in each regime, "
            "and whether any of it is statistically distinguishable from luck."
        ),
        unsafe_allow_html=True,
    )

    run_id = _pick_run()
    if run_id is None:
        st.info(
            "No studies have been run yet. Configure one on the **Run a study** "
            "tab, or run `qmr run` from the command line.",
        )
        return

    try:
        record = load_experiment(run_id)
    except Exception as exc:
        st.error(f"Could not load {run_id}: {exc}")
        return

    summary = record["summary"]
    metrics = summary.get("metrics", {}) or {}
    benchmark = summary.get("benchmark_metrics", {}) or {}
    classification = summary.get("classification", {}) or {}
    significance = summary.get("significance", {}) or {}
    baselines = summary.get("baselines", {}) or {}

    st.caption(
        f"`{run_id}` · out of sample {str(summary.get('oos_start'))[:10]} to "
        f"{str(summary.get('oos_end'))[:10]} · {summary.get('oos_bars', 0):,} bars · "
        f"{summary.get('model')} · {summary.get('regime_method')} regimes"
    )

    # -- headline ---------------------------------------------------------
    sharpe = metrics.get("sharpe", float("nan"))
    excess = sharpe - benchmark.get("sharpe", 0.0)
    st.markdown(
        theme.stat_tiles(
            [
                ("Sharpe ratio", f"{sharpe:+.2f}", "annualised, after costs"),
                (
                    "Against buy-and-hold",
                    f"{excess:+.2f}",
                    f"benchmark {benchmark.get('sharpe', float('nan')):+.2f}",
                ),
                ("CAGR", theme.format_metric("cagr", metrics.get("cagr")), "compound annual"),
                (
                    "Maximum drawdown",
                    theme.format_metric("max_drawdown", metrics.get("max_drawdown")),
                    "deepest peak-to-trough",
                ),
                (
                    "Directional precision",
                    theme.format_metric(
                        "directional_precision", classification.get("directional_precision")
                    ),
                    "correct calls, bars traded",
                ),
                (
                    "Time in market",
                    theme.format_metric("exposure", metrics.get("exposure")),
                    "share of bars holding a position",
                ),
            ]
        ),
        unsafe_allow_html=True,
    )

    tone, verdict = _significance_verdict(significance)
    {"good": st.success, "warning": st.warning, "critical": st.error}[tone](verdict)

    # -- equity and drawdown ----------------------------------------------
    equity = record.get("equity")
    if equity is not None and not equity.empty:
        st.plotly_chart(
            theme.equity_chart(
                equity["equity"],
                equity["benchmark_equity"],
                title="Out-of-sample equity, indexed to 100 at the start",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            theme.drawdown_chart(equity["drawdown"], title="Underwater curve"),
            use_container_width=True,
        )

    st.divider()

    # -- fold stability ----------------------------------------------------
    st.markdown("#### Stability across folds")
    st.markdown(
        theme.lede(
            "A strategy that earns its whole Sharpe in one fold and loses money in "
            "the others has not found a persistent edge; it has found one good "
            "year. Consistency across folds matters more than the pooled number."
        ),
        unsafe_allow_html=True,
    )

    fold_metrics = record.get("fold_metrics")
    if fold_metrics is not None and not fold_metrics.empty:
        left, right = st.columns([1.1, 1], gap="large")
        with left:
            st.plotly_chart(
                theme.bar_chart(
                    [f"Fold {int(f)}" for f in fold_metrics["fold"]],
                    fold_metrics["sharpe"].tolist(),
                    title="Sharpe ratio by fold",
                    diverging=True,
                    axis_title="Sharpe",
                    height=320,
                ),
                use_container_width=True,
            )
        with right:
            comparison = fold_metrics.set_index(
                fold_metrics["fold"].map(lambda f: f"Fold {int(f)}")
            )[["sharpe", "benchmark_sharpe"]].rename(
                columns={"sharpe": "Strategy", "benchmark_sharpe": "Buy and hold"}
            )
            st.plotly_chart(
                theme.grouped_bar_chart(
                    comparison,
                    title="Strategy against the benchmark, fold by fold",
                    axis_title="Sharpe",
                    height=320,
                ),
                use_container_width=True,
            )

        stability = significance.get("fold_stability", {}) or {}
        if stability:
            st.markdown(
                theme.stat_tiles(
                    [
                        (
                            "Profitable folds",
                            f"{stability.get('positive_fold_share', 0) * 100:.0f}%",
                            f"of {stability.get('folds', 0)} folds",
                        ),
                        (
                            "Mean fold Sharpe",
                            f"{stability.get('mean', float('nan')):+.2f}",
                            f"spread {stability.get('std', float('nan')):.2f}",
                        ),
                        (
                            "Worst fold",
                            f"{stability.get('min', float('nan')):+.2f}",
                            "the fold that hurt most",
                        ),
                        (
                            "p-value on the fold mean",
                            f"{stability.get('p_value', float('nan')):.3f}",
                            "one-sided t-test, folds as observations",
                        ),
                    ]
                ),
                unsafe_allow_html=True,
            )

        with st.expander("Per-fold detail", expanded=False):
            st.dataframe(fold_metrics.round(4), use_container_width=True, hide_index=True)
            layout = record.get("fold_layout")
            if layout is not None:
                st.caption("Walk-forward schedule")
                st.dataframe(layout, use_container_width=True, hide_index=True)

    st.divider()

    # -- regime breakdown --------------------------------------------------
    st.markdown("#### Performance by regime")
    st.markdown(
        theme.lede(
            "This is the table the research question turns on. If the edge is "
            "concentrated in one or two states, conditioning on the regime is "
            "earning its place; if it is spread evenly, the regime layer is "
            "complexity without benefit."
        ),
        unsafe_allow_html=True,
    )

    regime_performance = record.get("regime_performance")
    if regime_performance is not None and not regime_performance.empty:
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.plotly_chart(
                theme.bar_chart(
                    regime_performance["Regime"].tolist(),
                    regime_performance["Sharpe"].tolist(),
                    title="Sharpe ratio within each regime",
                    diverging=True,
                    orientation="h",
                    axis_title="Sharpe",
                    height=300,
                ),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                theme.bar_chart(
                    regime_performance["Regime"].tolist(),
                    (regime_performance["Directional precision"] * 100).tolist(),
                    title="Directional precision within each regime (%)",
                    orientation="h",
                    axis_title="Precision (%)",
                    value_format=".1f",
                    height=300,
                ),
                use_container_width=True,
            )
        st.dataframe(regime_performance.round(4), use_container_width=True, hide_index=True)

        transitions = record.get("regime_transitions")
        if transitions is not None and not transitions.empty:
            with st.expander("Regime transition matrix, out of sample", expanded=False):
                matrix = transitions.set_index(transitions.columns[0])
                st.plotly_chart(
                    theme.heatmap(
                        matrix, title="Transition probability", colorbar_title="p", height=340
                    ),
                    use_container_width=True,
                )

    st.divider()

    # -- what the model learned -------------------------------------------
    st.markdown("#### What the model learned")
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        importance = record.get("feature_importance")
        if importance is not None and not importance.empty:
            top = importance.head(18)
            names = top.iloc[:, 0].tolist() if top.shape[1] > 1 else top.index.tolist()
            values = top["value"].tolist() if "value" in top.columns else top.iloc[:, -1].tolist()
            st.plotly_chart(
                theme.bar_chart(
                    [str(n) for n in names],
                    [float(v) for v in values],
                    title="Feature importance, averaged across folds",
                    orientation="h",
                    axis_title="Share of total importance",
                    value_format=".3f",
                    height=460,
                ),
                use_container_width=True,
            )
            st.caption(
                "Model-native importance. It says what the model leaned on, not "
                "what causes returns — a feature can dominate an ensemble and "
                "still carry no economic signal."
            )
        else:
            st.info(
                "This learner does not expose a feature importance. Recurrent "
                "models are reported without one rather than with a fabricated "
                "surrogate.",
            )

    with right:
        confusion = record.get("confusion")
        if confusion is not None and not confusion.empty:
            matrix = confusion.set_index(confusion.columns[0])
            st.plotly_chart(
                theme.heatmap(
                    matrix,
                    title="Confusion matrix (rows: actual, columns: predicted)",
                    value_format=".0f",
                    colorbar_title="bars",
                    height=330,
                ),
                use_container_width=True,
            )

        st.markdown(
            theme.stat_tiles(
                [
                    (
                        "Accuracy",
                        theme.format_metric("accuracy", classification.get("accuracy")),
                        "misleading on its own",
                    ),
                    (
                        "Matthews correlation",
                        f"{classification.get('matthews_corrcoef', float('nan')):+.3f}",
                        "honest under class imbalance",
                    ),
                    (
                        "Signal rate",
                        theme.format_metric("signal_rate", classification.get("signal_rate")),
                        "bars where a position was taken",
                    ),
                ]
            ),
            unsafe_allow_html=True,
        )

    threshold_curve = record.get("threshold_curve")
    if threshold_curve is not None and not threshold_curve.empty:
        with st.expander("Decision threshold: precision against coverage", expanded=False):
            indexed = threshold_curve.set_index("threshold")
            st.plotly_chart(
                theme.line_chart(
                    {
                        "Directional precision": indexed["directional_precision"] * 100,
                        "Share of bars traded": indexed["signal_rate"] * 100,
                    },
                    title="Raising the threshold buys precision with coverage",
                    axis_title="Per cent",
                    x_title="Decision threshold",
                    value_format=".1f",
                ),
                use_container_width=True,
            )
            st.caption(
                "Both series are percentages, so they share one axis honestly. "
                "The threshold used by this study was "
                f"{summary.get('decision_threshold', float('nan')):.2f}."
            )

    st.divider()

    # -- benchmarks --------------------------------------------------------
    st.markdown("#### Against the rule-based benchmarks")
    st.markdown(
        theme.lede(
            "Strategies a practitioner could have run in 1990, priced through the "
            "same execution model and the same costs. If the learned signal cannot "
            "beat them, it has not found an edge."
        ),
        unsafe_allow_html=True,
    )

    if baselines:
        from qmr.models.baselines import BASELINES

        rows = [
            {
                "Strategy": BASELINES.get(key, (key, None, ""))[0],
                "Sharpe": values.get("sharpe"),
                "CAGR": values.get("cagr"),
                "Max drawdown": values.get("max_drawdown"),
                "Profit factor": values.get("profit_factor"),
                "Trades": values.get("trades"),
            }
            for key, values in baselines.items()
        ]
        rows.append(
            {
                "Strategy": f"This study ({summary.get('model')})",
                "Sharpe": metrics.get("sharpe"),
                "CAGR": metrics.get("cagr"),
                "Max drawdown": metrics.get("max_drawdown"),
                "Profit factor": metrics.get("profit_factor"),
                "Trades": metrics.get("trades"),
            }
        )
        table = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)

        st.plotly_chart(
            theme.bar_chart(
                table["Strategy"].tolist(),
                table["Sharpe"].tolist(),
                title="Sharpe ratio, same window and same costs",
                diverging=True,
                orientation="h",
                axis_title="Sharpe",
                height=320,
            ),
            use_container_width=True,
        )
        st.dataframe(table.round(4), use_container_width=True, hide_index=True)

    st.divider()

    # -- significance and full metrics --------------------------------------
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Is it real?")
        st.markdown(
            theme.stat_tiles(
                [
                    (
                        "Sharpe",
                        f"{significance.get('sharpe', float('nan')):+.2f}",
                        f"{significance.get('confidence_level', 0.95):.0%} interval "
                        f"[{significance.get('lower', float('nan')):+.2f}, "
                        f"{significance.get('upper', float('nan')):+.2f}]",
                    ),
                    (
                        "Probabilistic Sharpe",
                        f"{significance.get('probabilistic_sharpe', float('nan')):.3f}",
                        "P(true Sharpe > 0), skew and kurtosis corrected",
                    ),
                    (
                        "Deflated Sharpe",
                        f"{significance.get('deflated_sharpe', float('nan')):.3f}",
                        f"after correcting for {significance.get('n_trials', 1)} trials",
                    ),
                ]
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            "The interval comes from a stationary bootstrap, which resamples in "
            "blocks so the autocorrelation of the return stream survives. "
            "Resampling individual bars would give an interval far too narrow."
        )

    with right:
        st.markdown("#### Full performance profile")
        st.dataframe(
            summarise_metrics(metrics), use_container_width=True, hide_index=True, height=420
        )

    with st.expander("Trades", expanded=False):
        trades = record.get("trades")
        if trades is not None and not trades.empty:
            st.dataframe(trades.round(6), use_container_width=True, hide_index=True)
            st.plotly_chart(
                theme.histogram(
                    trades["return"] * 100,
                    title="Trade return distribution",
                    axis_title="Return per trade (%)",
                    bins=50,
                ),
                use_container_width=True,
            )
        else:
            st.caption("No trades were recorded for this run.")

    st.caption(f"Artefacts: `{record['path']}`")
