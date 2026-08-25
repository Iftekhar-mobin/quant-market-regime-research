"""Overview: the research question, the method, and what is loaded."""

from __future__ import annotations

import streamlit as st
import theme
from state import cached_catalog, cached_describe

PIPELINE = """
```
      market history            <-  broker CSV / MetaTrader 5 export
            |
      causal features           <-  ~80 scale-invariant, past-only indicators
            |
      regime detection          <-  k-means / Gaussian mixture / rule, refit per fold
            |
      triple-barrier labels     <-  path-aware targets in ATR units
            |
      walk-forward validation   <-  expanding folds, embargoed
            |
      directional model         <-  logit / forest / XGBoost / LightGBM / LSTM
            |
      backtest with costs       <-  next-bar-open fills, spread, minimum hold
            |
      risk + significance       <-  drawdown, bootstrap CI, deflated Sharpe
```
"""


def render(selection: dict, config) -> None:
    st.markdown(
        theme.lede(
            "This framework tests one question end to end: <b>do market regimes "
            "identified from price, volatility and technical structure make "
            "systematic strategies more robust out of sample?</b> Every stage "
            "below is a separate, inspectable module, and every result the "
            "console reports is produced by walk-forward validation with an "
            "embargo — never by a fit on the data it is scored against."
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown("#### The pipeline")
        st.markdown(PIPELINE)

    with right:
        st.markdown("#### Design decisions that determine the answer")
        st.markdown(
            """
**Causal features only.** Swing points are reported on the bar that *confirms*
them, not the bar they occurred on. The usual `argrelextrema` pivot leaks the
future into the feature set and is the fastest way to a backtest that cannot be
reproduced live.

**Regimes are refitted inside every fold.** Clustering the full sample and then
evaluating "out of sample" within those clusters leaks the entire future into
the regime assignment.

**Path-aware labels.** A fixed-horizon return sign rewards a model for calling a
move that a real position would have been stopped out of. The triple barrier
prices the path in.

**Costs and a minimum holding period.** A twelve-bar-ahead forecast is one
opinion, not twelve. Acting on it every bar pays the spread twelve times, and on
intraday FX that alone turns a real edge into a loss.

**The benchmark is not handicapped.** Buy-and-hold and four rule-based
strategies are priced through the same execution model and the same costs.
"""
        )

    st.divider()

    # -- what is currently loaded -----------------------------------------
    st.markdown("#### Loaded dataset")
    try:
        stats = cached_describe(
            selection["symbol"], selection["timeframe"], selection["start"], selection["end"]
        )
    except Exception as exc:  # a bad date range should explain itself, not crash
        st.error(f"Could not load {selection['symbol']} {selection['timeframe']}: {exc}")
        return

    st.markdown(
        theme.stat_tiles(
            [
                (
                    "Instrument",
                    f"{selection['symbol']} {selection['timeframe']}",
                    "selected in the sidebar",
                ),
                ("Bars", f"{stats['bars']:,}", "after the study window is applied"),
                (
                    "Period",
                    f"{stats['start']:%Y-%m}",
                    f"through {stats['end']:%Y-%m}",
                ),
                (
                    "Volatility",
                    f"{stats['volatility_bps']:.1f} bps",
                    "standard deviation per bar",
                ),
                (
                    "Excess kurtosis",
                    f"{stats['kurtosis']:.1f}",
                    "fat tails: 0 would be normal",
                ),
                (
                    "Gaps",
                    f"{stats['missing_bars_pct']:.1f}%",
                    "bars wider than the median spacing",
                ),
            ]
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        theme.note(
            "Excess kurtosis well above zero is the reason this framework reports "
            "a bootstrap confidence interval and a probabilistic Sharpe ratio "
            "rather than a Sharpe ratio alone: the usual standard error assumes a "
            "normal distribution that these returns plainly do not have."
        ),
        unsafe_allow_html=True,
    )

    st.divider()

    with st.expander("Full data catalogue", expanded=False):
        catalog = cached_catalog()
        st.dataframe(catalog, use_container_width=True, hide_index=True)
        st.caption(
            "Files in `data/samples` ship with the repository so a fresh clone "
            "runs. Files in `data/raw` are the full local history and are not "
            "version controlled."
        )

    with st.expander("How to reproduce any result from the command line", expanded=False):
        st.markdown(
            """
The console writes every study to `experiments/<run_id>/` with the exact
configuration that produced it. The same runs are available headlessly:

```bash
qmr datasets                                     # what history is available
qmr run --set model.name=xgboost --set regime.method=kmeans
qmr compare --models logistic,xgboost,lightgbm --regimes none,kmeans
qmr results                                      # every stored run
qmr show <run_id>                                # one run in full
```

Nothing in the console is a shortcut around the library: both entry points call
the same `run_experiment`.
"""
        )
