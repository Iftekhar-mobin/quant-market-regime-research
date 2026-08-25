"""Regime lab: fit a detector and characterise the states it finds.

This view is exploratory and fits on the whole selected window, which is fine
for *describing* the market and wrong for *scoring* a strategy. The distinction
is stated on screen, because a regime chart fitted on the full sample is the
most persuasive-looking way to fool yourself in this entire field.
"""

from __future__ import annotations

import streamlit as st
import theme
from state import cached_features

DETECTOR_HELP = {
    "kmeans": "Hard partition of the descriptor space. Fast, and the states are "
    "easy to read, but every bar is assigned with full confidence even at a "
    "transition.",
    "gmm": "Soft partition. Each bar carries a posterior over the states, which "
    "is closer to how markets actually change — a range does not become a trend "
    "between one bar and the next.",
    "rule": "Trend direction crossed with volatility level, thresholded at the "
    "training median. The transparent benchmark the learned detectors have to beat.",
    "none": "A single state. The control arm: everything else is measured "
    "against this.",
}


def render(selection: dict, config) -> None:
    st.markdown(
        theme.lede(
            "A regime is a persistent market state — trending or ranging, calm or "
            "stressed — that changes what a strategy should expect from its own "
            "signals. A regime is only useful if it is <b>persistent</b> enough to "
            "trade, <b>distinct</b> in its risk and return, and <b>stable</b> "
            "enough to reappear out of sample. This view measures all three."
        ),
        unsafe_allow_html=True,
    )

    st.warning(
        "This view fits the detector on the whole selected window so the states "
        "can be described. That is exploration, not evaluation — in a study the "
        "detector is refitted inside every walk-forward fold. Never read "
        "performance off this page.",
    )

    from qmr.config import RegimeConfig
    from qmr.data.catalog import bars_per_year
    from qmr.regimes import (
        build_detector,
        regime_summary,
        regime_transition_matrix,
        run_lengths,
    )

    symbol, timeframe = selection["symbol"], selection["timeframe"]
    blocks = tuple(config.features.blocks)

    try:
        features = cached_features(
            symbol, timeframe, selection["start"], selection["end"], blocks, config.data.warmup_bars
        )
    except Exception as exc:
        st.error(f"Could not build features: {exc}")
        return

    controls = st.columns([1, 1, 2])
    with controls[0]:
        method = st.selectbox(
            "Detector",
            ["kmeans", "gmm", "rule", "none"],
            index=0,
            key="rl_method",
        )
    with controls[1]:
        n_regimes = st.slider(
            "Number of regimes",
            2,
            8,
            int(config.regime.n_regimes),
            disabled=method in {"rule", "none"},
            key="rl_n",
            help="The rule-based detector always defines exactly four states.",
        )
    with controls[2]:
        st.caption(DETECTOR_HELP[method])

    detector = build_detector(
        RegimeConfig(method=method, n_regimes=n_regimes), random_state=config.experiment.seed
    )
    with st.spinner("Fitting the regime detector..."):
        regimes = detector.fit_predict(features)
    names = detector.labels_

    annualisation = bars_per_year(timeframe)
    summary = regime_summary(features, regimes, names, bars_per_year=annualisation)
    runs = run_lengths(regimes)

    # -- headline ---------------------------------------------------------
    persistence = float(regimes.eq(regimes.shift()).mean())
    st.markdown(
        theme.stat_tiles(
            [
                ("States found", f"{regimes.nunique()}", "distinct market regimes"),
                (
                    "Bar-to-bar persistence",
                    f"{persistence * 100:.1f}%",
                    "chance the state is unchanged next bar",
                ),
                (
                    "Median run",
                    f"{runs['bars'].median():.0f} bars",
                    "typical time spent in one state",
                ),
                (
                    "Regime switches",
                    f"{len(runs):,}",
                    "over the selected window",
                ),
            ]
        ),
        unsafe_allow_html=True,
    )

    if persistence < 0.9:
        st.markdown(
            theme.note(
                "Persistence below about 90% means the detector is switching state "
                "faster than a position could be held. A state that lasts three "
                "bars cannot be traded after costs, however well it separates the "
                "data. Try fewer regimes, or a slower timeframe."
            ),
            unsafe_allow_html=True,
        )

    # -- price with regime shading ----------------------------------------
    st.divider()
    window = st.select_slider(
        "Chart window (most recent bars)",
        options=[500, 1000, 2500, 5000, 10000],
        value=2500,
        key="rl_window",
    )
    view = features.iloc[-window:]

    st.plotly_chart(
        theme.price_chart(
            view[["open", "high", "low", "close"]],
            regimes=regimes.reindex(view.index),
            regime_names=names,
            title=f"{symbol} {timeframe} — price, shaded by detected regime",
            height=440,
        ),
        use_container_width=True,
    )

    # -- characterisation --------------------------------------------------
    st.divider()
    st.markdown("#### What each regime actually is")
    st.markdown(
        theme.lede(
            "The return columns are buy-and-hold <i>within</i> each state. They "
            "answer whether a regime carries a directional edge on its own, which "
            "is the baseline any regime-conditioned model has to improve on."
        ),
        unsafe_allow_html=True,
    )
    st.dataframe(summary, use_container_width=True)

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        shares = regimes.value_counts(normalize=True).sort_index()
        st.plotly_chart(
            theme.regime_share_chart(shares, names, title="Share of the sample"),
            use_container_width=True,
        )

    with right:
        transitions = regime_transition_matrix(regimes)
        transitions.index = [names.get(int(i), f"Regime {i}") for i in transitions.index]
        transitions.columns = [names.get(int(i), f"Regime {i}") for i in transitions.columns]
        st.plotly_chart(
            theme.heatmap(
                transitions,
                title="Transition probability, bar to bar",
                colorbar_title="p",
                height=340,
            ),
            use_container_width=True,
        )
        st.caption(
            "The diagonal is persistence. Values near 1 mean the market rarely "
            "leaves that state from one bar to the next."
        )

    # -- descriptor space --------------------------------------------------
    st.divider()
    st.markdown("#### The descriptor space")
    st.markdown(
        theme.lede(
            "Detectors work in four interpretable coordinates rather than on the "
            "full feature matrix, which is what makes the states nameable instead "
            "of an opaque partition of an eighty-dimensional space."
        ),
        unsafe_allow_html=True,
    )

    descriptors = ["trend_strength", "vol_percentile", "momentum_score", "mean_reversion_score"]
    axes = st.columns([1, 1, 2])
    with axes[0]:
        x_axis = st.selectbox("Horizontal", descriptors, index=0, key="rl_x")
    with axes[1]:
        y_axis = st.selectbox("Vertical", descriptors, index=1, key="rl_y")

    sample = features.sample(min(6000, len(features)), random_state=config.experiment.seed)
    st.plotly_chart(
        theme.scatter_chart(
            sample[x_axis],
            sample[y_axis],
            colour_by=regimes.reindex(sample.index),
            names=names,
            title=f"{y_axis} against {x_axis}",
            x_title=x_axis,
            y_title=y_axis,
        ),
        use_container_width=True,
    )
    st.caption(
        "Scatter plots cap the categorical palette at three hues: past three, "
        "adjacent colours stop being separable for colour-vision-deficient "
        "readers, so the remaining states fold into a neutral group. The table "
        "above carries the full breakdown."
    )

    # -- run lengths -------------------------------------------------------
    with st.expander("Regime run lengths", expanded=False):
        run_summary = (
            runs.groupby("regime")["bars"]
            .agg(["count", "mean", "median", "max"])
            .rename(
                columns={
                    "count": "Runs",
                    "mean": "Mean bars",
                    "median": "Median bars",
                    "max": "Longest run",
                }
            )
            .round(1)
        )
        run_summary.index = [names.get(int(i), f"Regime {i}") for i in run_summary.index]
        st.dataframe(run_summary, use_container_width=True)

        longest = runs.nlargest(12, "bars").copy()
        longest["regime"] = [names.get(int(i), f"Regime {i}") for i in longest["regime"]]
        st.caption("Longest individual episodes")
        st.dataframe(
            longest.rename(
                columns={"regime": "Regime", "start": "Start", "end": "End", "bars": "Bars"}
            ),
            use_container_width=True,
            hide_index=True,
        )
