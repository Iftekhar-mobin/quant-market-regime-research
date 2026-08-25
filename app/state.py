"""Shared, cached access to data and study artefacts for the console.

Streamlit re-executes the whole script on every interaction, so anything that
touches the disk or spends real CPU is cached here rather than in the tabs. The
cache keys are plain values (symbol, timeframe, dates) so a config change
invalidates exactly what it should and nothing more.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# The console runs from app/, so put the package on the path before importing it.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qmr.config import Config, FeatureConfig, LabelConfig  # noqa: E402
from qmr.data.catalog import catalog_frame, scan_catalog  # noqa: E402
from qmr.data.loader import describe, load_ohlcv  # noqa: E402
from qmr.features import build_features  # noqa: E402
from qmr.labeling import build_labels  # noqa: E402
from qmr.paths import DEFAULT_CONFIG_PATH  # noqa: E402

CACHE_TTL = 3600


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_catalog() -> pd.DataFrame:
    return catalog_frame()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_datasets() -> list[tuple[str, str, str]]:
    """(symbol, timeframe, coverage) for every discovered dataset."""
    return [(d.symbol, d.timeframe, d.coverage) for d in scan_catalog()]


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading market history...")
def cached_price(symbol: str, timeframe: str, start: str | None, end: str | None) -> pd.DataFrame:
    return load_ohlcv(symbol, timeframe, start=start, end=end)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_describe(symbol: str, timeframe: str, start: str | None, end: str | None) -> dict:
    return describe(cached_price(symbol, timeframe, start, end))


@st.cache_data(ttl=CACHE_TTL, show_spinner="Building features...")
def cached_features(
    symbol: str,
    timeframe: str,
    start: str | None,
    end: str | None,
    blocks: tuple[str, ...],
    warmup: int,
) -> pd.DataFrame:
    price = cached_price(symbol, timeframe, start, end)
    return build_features(price, FeatureConfig(blocks=list(blocks)), warmup_bars=warmup)


@st.cache_data(ttl=CACHE_TTL, show_spinner="Labelling targets...")
def cached_labels(
    symbol: str,
    timeframe: str,
    start: str | None,
    end: str | None,
    blocks: tuple[str, ...],
    warmup: int,
    method: str,
    horizon: int,
    take_profit: float,
    stop_loss: float,
) -> pd.DataFrame:
    features = cached_features(symbol, timeframe, start, end, blocks, warmup)
    result = build_labels(
        features,
        LabelConfig(
            method=method,
            horizon=horizon,
            take_profit_atr=take_profit,
            stop_loss_atr=stop_loss,
        ),
    )
    return pd.DataFrame(
        {
            "label": result.labels,
            "forward_return": result.forward_return,
            "holding_bars": result.holding_bars,
            "barrier_hit": result.barrier_hit,
        }
    )


def base_config() -> Config:
    """The on-disk default configuration, loaded once per session."""
    if "base_config" not in st.session_state:
        st.session_state["base_config"] = Config.load(DEFAULT_CONFIG_PATH)
    return st.session_state["base_config"]


def selection() -> dict:
    """The dataset selection made in the sidebar."""
    return st.session_state.get(
        "selection",
        {"symbol": "EURUSD", "timeframe": "H1", "start": None, "end": None},
    )


def empty_state(message: str, hint: str = "") -> None:
    """Consistent placeholder for a view with nothing to show yet."""
    st.info(message)
    if hint:
        st.caption(hint)
