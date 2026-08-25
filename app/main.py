"""Research console.

    streamlit run app/main.py      (or: qmr console)

The console is a front end to the same library the CLI drives — it configures a
study, runs it, and reads back the artefacts in ``experiments/``. Nothing is
computed here that cannot be reproduced from a config file, which is the point:
the interface is for exploring results, not for producing ones that only exist
inside a browser session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Market Regime Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import theme  # noqa: E402
from state import base_config, cached_datasets  # noqa: E402
from tabs import (  # noqa: E402
    comparison,
    data_explorer,
    overview,
    regime_lab,
    results,
    signals,
    training,
)

theme.register_template()
st.markdown(theme.APP_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: the dataset every tab works against
# ---------------------------------------------------------------------------
def render_sidebar() -> dict:
    st.sidebar.markdown("### Market Regime Research")
    st.sidebar.caption(
        "Do machine-learned market regimes make systematic strategies more robust "
        "out of sample?"
    )
    st.sidebar.divider()

    datasets = cached_datasets()
    if not datasets:
        st.sidebar.error("No market history found.")
        st.sidebar.caption(
            "Add CSV files named `SYMBOL_TIMEFRAME_YYYYMMDD_YYYYMMDD.csv` to "
            "`data/raw`, or run `python scripts/prepare_sample_data.py`."
        )
        st.stop()

    symbols = sorted({symbol for symbol, _, _ in datasets})
    default_symbol = "EURUSD" if "EURUSD" in symbols else symbols[0]

    st.sidebar.markdown("#### Dataset")
    symbol = st.sidebar.selectbox(
        "Instrument", symbols, index=symbols.index(default_symbol), key="sb_symbol"
    )

    timeframes = [tf for sym, tf, _ in datasets if sym == symbol]
    default_timeframe = "H1" if "H1" in timeframes else timeframes[0]
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        timeframes,
        index=timeframes.index(default_timeframe),
        key="sb_timeframe",
        help="Slower timeframes carry less noise per bar and cost far less to trade.",
    )

    coverage = next(
        (cov for sym, tf, cov in datasets if sym == symbol and tf == timeframe), "unknown"
    )
    st.sidebar.caption(f"Available: {coverage}")

    st.sidebar.markdown("#### Study window")
    start = st.sidebar.text_input(
        "From (YYYY-MM-DD)",
        value=st.session_state.get("sb_start", "2018-01-01"),
        key="sb_start",
        help="Leave blank to use the full history. A longer window gives the "
        "walk-forward more folds and a more credible result.",
    )
    end = st.sidebar.text_input("To (YYYY-MM-DD)", value=st.session_state.get("sb_end", ""), key="sb_end")

    selection = {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": start.strip() or None,
        "end": end.strip() or None,
    }
    st.session_state["selection"] = selection

    st.sidebar.divider()
    st.sidebar.caption(
        "Every number in this console is out of sample: models, scalers and "
        "regime detectors are refitted inside each walk-forward fold, with an "
        "embargo between training and test."
    )
    return selection


def main() -> None:
    selection = render_sidebar()
    config = base_config()

    st.title("Quantitative Market Regime Research")

    tab_overview, tab_data, tab_regimes, tab_train, tab_results, tab_compare, tab_signals = st.tabs(
        [
            "Overview",
            "Data",
            "Regimes",
            "Run a study",
            "Results",
            "Model comparison",
            "Signals",
        ]
    )

    with tab_overview:
        overview.render(selection, config)
    with tab_data:
        data_explorer.render(selection, config)
    with tab_regimes:
        regime_lab.render(selection, config)
    with tab_train:
        training.render(selection, config)
    with tab_results:
        results.render(selection, config)
    with tab_compare:
        comparison.render(selection, config)
    with tab_signals:
        signals.render(selection, config)


main()
