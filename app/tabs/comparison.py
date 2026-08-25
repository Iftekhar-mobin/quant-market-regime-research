"""Model comparison: every stored study, side by side."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import theme

METRIC_CHOICES = {
    "Sharpe": ("sharpe", "Sharpe ratio", True),
    "CAGR": ("cagr", "Compound annual growth (%)", True),
    "Max drawdown": ("max_drawdown", "Maximum drawdown (%)", True),
    "Profit factor": ("profit_factor", "Profit factor", False),
    "Precision": ("directional_precision", "Directional precision (%)", False),
    "Deflated Sharpe": ("deflated_sharpe", "Deflated Sharpe", False),
}


def _comparison_frame(records) -> pd.DataFrame:
    rows = []
    for record in records:
        summary = record.summary
        metrics = summary.get("metrics", {}) or {}
        classification = summary.get("classification", {}) or {}
        significance = summary.get("significance", {}) or {}
        benchmark = summary.get("benchmark_metrics", {}) or {}
        rows.append(
            {
                "Run": record.run_id,
                "Symbol": summary.get("symbol"),
                "Timeframe": summary.get("timeframe"),
                "Model": summary.get("model"),
                "Regimes": summary.get("regime_method"),
                "Per-regime models": summary.get("specialised_models", False),
                "sharpe": metrics.get("sharpe"),
                "benchmark_sharpe": benchmark.get("sharpe"),
                "cagr": metrics.get("cagr"),
                "max_drawdown": metrics.get("max_drawdown"),
                "profit_factor": metrics.get("profit_factor"),
                "trades": metrics.get("trades"),
                "directional_precision": classification.get("directional_precision"),
                "deflated_sharpe": significance.get("deflated_sharpe"),
                "oos_bars": summary.get("oos_bars"),
                "Created": record.created_at[:16].replace("T", " "),
            }
        )
    return pd.DataFrame(rows)


def render(selection: dict, config) -> None:
    from qmr.experiments.store import delete_experiment, list_experiments

    st.markdown(
        theme.lede(
            "Every stored study, side by side. The comparison that matters is not "
            "which learner scored highest — it is whether the regime-conditioned "
            "arms beat their own control arm on the same instrument, the same "
            "window and the same costs."
        ),
        unsafe_allow_html=True,
    )

    records = list_experiments()
    if not records:
        st.info(
            "No studies stored yet. Run several from the **Run a study** tab, or "
            "sweep them from the command line:\n\n"
            "`qmr compare --models logistic,xgboost,lightgbm --regimes none,kmeans`",
        )
        return

    frame = _comparison_frame(records)

    # -- filters, in one row above the charts ------------------------------
    filters = st.columns([1, 1, 1, 1])
    with filters[0]:
        symbols = ["All"] + sorted(frame["Symbol"].dropna().unique().tolist())
        symbol_filter = st.selectbox("Instrument", symbols, key="cmp_symbol")
    with filters[1]:
        models = ["All"] + sorted(frame["Model"].dropna().unique().tolist())
        model_filter = st.selectbox("Learner", models, key="cmp_model")
    with filters[2]:
        regimes = ["All"] + sorted(frame["Regimes"].dropna().unique().tolist())
        regime_filter = st.selectbox("Regime method", regimes, key="cmp_regime")
    with filters[3]:
        metric_label = st.selectbox("Rank by", list(METRIC_CHOICES), key="cmp_metric")

    view = frame.copy()
    if symbol_filter != "All":
        view = view[view["Symbol"] == symbol_filter]
    if model_filter != "All":
        view = view[view["Model"] == model_filter]
    if regime_filter != "All":
        view = view[view["Regimes"] == regime_filter]

    if view.empty:
        st.warning("No studies match those filters.")
        return

    metric_key, axis_title, as_percent = METRIC_CHOICES[metric_label]
    view = view.sort_values(metric_key, ascending=False)

    # -- ranking -----------------------------------------------------------
    labels = [
        f"{row['Model']} · {row['Regimes']}"
        + ("  (per-regime)" if row["Per-regime models"] else "")
        for _, row in view.iterrows()
    ]
    values = view[metric_key].astype(float)
    if as_percent and metric_key in {"cagr", "max_drawdown"}:
        values = values * 100

    st.plotly_chart(
        theme.bar_chart(
            labels,
            values.tolist(),
            title=f"{metric_label} across {len(view)} stored studies",
            orientation="h",
            diverging=metric_key in {"sharpe", "cagr", "deflated_sharpe"},
            axis_title=axis_title,
            height=max(300, 34 * len(view) + 110),
        ),
        use_container_width=True,
    )

    # -- the research question ---------------------------------------------
    st.divider()
    st.markdown("#### Does the regime layer earn its place?")
    st.markdown(
        theme.lede(
            "Studies are paired by instrument, timeframe and learner, then the "
            "regime-conditioned arm is compared against its own <code>none</code> "
            "control. This is the only comparison that isolates the effect of the "
            "regime layer from every other difference between two runs."
        ),
        unsafe_allow_html=True,
    )

    control = view[view["Regimes"] == "none"]
    treated = view[view["Regimes"] != "none"]

    if control.empty or treated.empty:
        st.info(
            "The comparison needs at least one study with `regime.method = none` "
            "and one with a regime detector, on the same instrument and learner.",
        )
    else:
        keys = ["Symbol", "Timeframe", "Model"]
        control_best = control.groupby(keys)["sharpe"].max().rename("Control (no regimes)")
        paired_rows = []
        for _, row in treated.iterrows():
            key = (row["Symbol"], row["Timeframe"], row["Model"])
            if key not in control_best.index:
                continue
            paired_rows.append(
                {
                    "Study": f"{row['Model']} · {row['Regimes']}"
                    + (" (per-regime)" if row["Per-regime models"] else ""),
                    "Control (no regimes)": float(control_best.loc[key]),
                    "With regimes": float(row["sharpe"]),
                }
            )

        if paired_rows:
            paired = pd.DataFrame(paired_rows).set_index("Study")
            st.plotly_chart(
                theme.grouped_bar_chart(
                    paired,
                    title="Sharpe ratio: regime-conditioned against its own control",
                    axis_title="Sharpe",
                    height=max(320, 46 * len(paired) + 120),
                ),
                use_container_width=True,
            )

            improvement = paired["With regimes"] - paired["Control (no regimes)"]
            wins = int((improvement > 0).sum())
            st.markdown(
                theme.stat_tiles(
                    [
                        ("Pairs compared", f"{len(paired)}", "matched on instrument and learner"),
                        (
                            "Regimes helped",
                            f"{wins} of {len(paired)}",
                            "higher Sharpe than the control",
                        ),
                        (
                            "Mean change in Sharpe",
                            f"{improvement.mean():+.3f}",
                            "positive means the regime layer added value",
                        ),
                        (
                            "Best improvement",
                            f"{improvement.max():+.3f}",
                            str(improvement.idxmax()),
                        ),
                    ]
                ),
                unsafe_allow_html=True,
            )

            if improvement.mean() <= 0:
                st.markdown(
                    theme.note(
                        "On this evidence the regime layer is not adding value. "
                        "That is a legitimate answer to the research question — "
                        "reporting it is what separates a study from a sales "
                        "pitch. Before concluding, check that the detected states "
                        "were persistent enough to trade (Regimes tab) and that "
                        "the control arm is not simply trading less."
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.info("No control/treatment pairs share an instrument and learner yet.")

    # -- strategy against its benchmark ------------------------------------
    st.divider()
    st.markdown("#### Every study against buy-and-hold")

    excess = (view["sharpe"] - view["benchmark_sharpe"]).astype(float)
    st.plotly_chart(
        theme.bar_chart(
            labels,
            excess.tolist(),
            title="Sharpe ratio above buy-and-hold, same window",
            orientation="h",
            diverging=True,
            axis_title="Excess Sharpe",
            height=max(300, 34 * len(view) + 110),
        ),
        use_container_width=True,
    )

    # -- table --------------------------------------------------------------
    st.divider()
    st.markdown("#### All studies")

    display = view.rename(
        columns={
            "sharpe": "Sharpe",
            "benchmark_sharpe": "Benchmark Sharpe",
            "cagr": "CAGR",
            "max_drawdown": "Max drawdown",
            "profit_factor": "Profit factor",
            "trades": "Trades",
            "directional_precision": "Precision",
            "deflated_sharpe": "Deflated Sharpe",
            "oos_bars": "OOS bars",
        }
    )
    st.dataframe(
        display.drop(columns=["Run"]).round(4), use_container_width=True, hide_index=True
    )

    csv = display.to_csv(index=False).encode("utf-8")
    download, remove = st.columns([1, 1])
    with download:
        st.download_button(
            "Download comparison as CSV",
            csv,
            file_name="model_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with remove:
        with st.popover("Delete a study", use_container_width=True):
            target = st.selectbox("Study to delete", view["Run"].tolist(), key="cmp_delete")
            if st.button("Delete permanently", type="secondary", key="cmp_delete_go"):
                delete_experiment(target)
                st.cache_data.clear()
                st.rerun()
