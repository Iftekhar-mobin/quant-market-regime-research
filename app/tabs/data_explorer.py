"""Data explorer: what the raw series and the engineered features look like."""

from __future__ import annotations

import numpy as np
import streamlit as st
import theme
from state import cached_features, cached_labels, cached_price


def render(selection: dict, config) -> None:
    st.markdown(
        theme.lede(
            "Before any model is fitted, look at the series. This view shows the "
            "price and its engineered features, the distribution of returns that "
            "sets the ceiling on what any strategy can earn, and the target the "
            "models are asked to predict."
        ),
        unsafe_allow_html=True,
    )

    symbol, timeframe = selection["symbol"], selection["timeframe"]
    try:
        price = cached_price(symbol, timeframe, selection["start"], selection["end"])
    except Exception as exc:
        st.error(f"Could not load {symbol} {timeframe}: {exc}")
        return

    # -- price ------------------------------------------------------------
    controls = st.columns([1, 1, 2])
    with controls[0]:
        window = st.selectbox(
            "Chart window",
            ["Last 500 bars", "Last 2,000 bars", "Last 10,000 bars", "Full window"],
            index=1,
            key="de_window",
        )
    with controls[1]:
        show_ma = st.multiselect(
            "Overlays", ["EMA 20", "EMA 50", "EMA 200"], default=["EMA 50", "EMA 200"], key="de_ma"
        )

    limits = {"Last 500 bars": 500, "Last 2,000 bars": 2000, "Last 10,000 bars": 10000}
    view = price.iloc[-limits[window] :] if window in limits else price

    overlays = {}
    for label in show_ma:
        span = int(label.split()[-1])
        overlays[label] = price["close"].ewm(span=span, adjust=False).mean().reindex(view.index)

    st.plotly_chart(
        theme.price_chart(
            view,
            overlays=overlays,
            title=f"{symbol} {timeframe} — close",
        ),
        use_container_width=True,
    )

    # -- return distribution ----------------------------------------------
    st.divider()
    st.markdown("#### Return distribution")
    st.markdown(
        theme.lede(
            "The shape of this distribution is what makes financial forecasting "
            "hard. Most bars are noise around zero; the returns that matter live "
            "in tails far heavier than a normal distribution would produce."
        ),
        unsafe_allow_html=True,
    )

    log_return = np.log(price["close"]).diff().dropna()
    left, right = st.columns(2, gap="large")

    with left:
        st.plotly_chart(
            theme.histogram(
                log_return * 1e4,
                title="Log return per bar (basis points)",
                axis_title="Return (bps)",
            ),
            use_container_width=True,
        )

    with right:
        rolling_volatility = log_return.rolling(200).std(ddof=0) * 1e4
        st.plotly_chart(
            theme.line_chart(
                {"200-bar realised volatility": rolling_volatility.dropna()},
                title="Volatility is not constant — this is the regime hypothesis",
                axis_title="Volatility (bps per bar)",
                value_format=".1f",
            ),
            use_container_width=True,
        )

    quantiles = log_return.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) * 1e4
    st.markdown(
        theme.stat_tiles(
            [
                ("Mean", f"{log_return.mean() * 1e4:+.3f} bps", "per bar"),
                ("Std deviation", f"{log_return.std() * 1e4:.2f} bps", "per bar"),
                ("Skew", f"{log_return.skew():+.2f}", "0 would be symmetric"),
                ("Excess kurtosis", f"{log_return.kurtosis():.1f}", "0 would be normal"),
                ("1st percentile", f"{quantiles.loc[0.01]:.1f} bps", "worst 1% of bars"),
                ("99th percentile", f"{quantiles.loc[0.99]:+.1f} bps", "best 1% of bars"),
            ]
        ),
        unsafe_allow_html=True,
    )

    # -- features ----------------------------------------------------------
    st.divider()
    st.markdown("#### Engineered features")

    blocks = tuple(config.features.blocks)
    try:
        features = cached_features(
            symbol, timeframe, selection["start"], selection["end"], blocks, config.data.warmup_bars
        )
    except Exception as exc:
        st.error(f"Feature construction failed: {exc}")
        return

    from qmr.features.pipeline import feature_catalogue, feature_columns

    columns = feature_columns(features)

    st.markdown(
        theme.stat_tiles(
            [
                ("Features", f"{len(columns)}", "all causal and scale-invariant"),
                ("Usable bars", f"{len(features):,}", "after indicator warm-up"),
                ("Blocks enabled", f"{len(blocks)}", "of 7 available"),
            ]
        ),
        unsafe_allow_html=True,
    )

    st.dataframe(feature_catalogue(config.features), use_container_width=True, hide_index=True)

    chosen = st.multiselect(
        "Inspect features",
        columns,
        default=[c for c in ("trend_strength", "vol_percentile", "rsi_14") if c in columns],
        key="de_features",
        help="Features are plotted on one shared axis, so pick ones on comparable scales.",
    )
    if chosen:
        tail = features.iloc[-2000:]
        st.plotly_chart(
            theme.line_chart(
                {name: tail[name] for name in chosen},
                title="Feature values, last 2,000 bars",
                axis_title="Value",
            ),
            use_container_width=True,
        )
        st.markdown(
            theme.note(
                "All series share a single y-axis. Two measures on different "
                "scales belong in two charts, never on two axes of one chart — "
                "the alignment of two scales is arbitrary and invents a "
                "correlation the data does not contain."
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Feature summary statistics", expanded=False):
        summary = features[columns].describe().T[["mean", "std", "min", "50%", "max"]]
        st.dataframe(summary.round(4), use_container_width=True)

    # -- labels ------------------------------------------------------------
    st.divider()
    st.markdown("#### The prediction target")

    label_controls = st.columns([1, 1, 1, 1])
    with label_controls[0]:
        method = st.selectbox(
            "Labelling", ["triple_barrier", "directional"], index=0, key="de_label_method"
        )
    with label_controls[1]:
        horizon = st.number_input(
            "Horizon (bars)", 4, 200, int(config.labeling.horizon), 4, key="de_horizon"
        )
    with label_controls[2]:
        take_profit = st.number_input(
            "Upper barrier (ATR)", 0.25, 5.0, float(config.labeling.take_profit_atr), 0.25,
            key="de_tp",
        )
    with label_controls[3]:
        stop_loss = st.number_input(
            "Lower barrier (ATR)", 0.25, 5.0, float(config.labeling.stop_loss_atr), 0.25,
            key="de_sl",
        )

    try:
        labels = cached_labels(
            symbol,
            timeframe,
            selection["start"],
            selection["end"],
            blocks,
            config.data.warmup_bars,
            method,
            int(horizon),
            float(take_profit),
            float(stop_loss),
        )
    except Exception as exc:
        st.error(f"Labelling failed: {exc}")
        return

    distribution = labels["label"].value_counts(normalize=True)
    names = {1: "Long", 0: "Flat", -1: "Short"}

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.plotly_chart(
            theme.bar_chart(
                [names[k] for k in (1, 0, -1)],
                [float(distribution.get(k, 0.0) * 100) for k in (1, 0, -1)],
                title="Label balance",
                axis_title="Share of bars (%)",
                value_format=".1f",
                height=300,
            ),
            use_container_width=True,
        )
    with right:
        if method == "triple_barrier":
            hits = labels["barrier_hit"].value_counts(normalize=True) * 100
            readable = {
                "upper": "Upper barrier first",
                "lower": "Lower barrier first",
                "time": "Time limit reached",
                "both": "Both in one bar",
            }
            st.plotly_chart(
                theme.bar_chart(
                    [readable.get(k, k) for k in hits.index],
                    [float(v) for v in hits],
                    title="Which barrier resolved the position",
                    axis_title="Share of samples (%)",
                    value_format=".1f",
                    orientation="h",
                    height=300,
                ),
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                theme.histogram(
                    labels["forward_return"] * 1e4,
                    title=f"Forward return over {int(horizon)} bars",
                    axis_title="Return (bps)",
                    height=300,
                ),
                use_container_width=True,
            )

    st.markdown(
        theme.stat_tiles(
            [
                ("Samples", f"{len(labels):,}", "labelled bars"),
                ("Long", f"{distribution.get(1, 0) * 100:.1f}%", "upper barrier first"),
                ("Short", f"{distribution.get(-1, 0) * 100:.1f}%", "lower barrier first"),
                ("Flat", f"{distribution.get(0, 0) * 100:.1f}%", "neither, within the horizon"),
                (
                    "Mean holding",
                    f"{labels['holding_bars'].mean():.1f} bars",
                    "how long a position actually lasts",
                ),
            ]
        ),
        unsafe_allow_html=True,
    )

    imbalance = abs(distribution.get(1, 0) - distribution.get(-1, 0))
    if imbalance > 0.05:
        st.markdown(
            theme.note(
                f"Long and short labels differ by {imbalance * 100:.1f} percentage "
                "points. Asymmetric barriers bias the label set towards whichever "
                "side is closer, and a model trained on it will inherit that bias "
                "as a directional tilt that has nothing to do with the market. "
                "Set both barriers equal unless the asymmetry is deliberate."
            ),
            unsafe_allow_html=True,
        )
