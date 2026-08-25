"""Export market history from a MetaTrader 5 terminal.

Optional. The research pipeline runs entirely from CSV, and the repository ships
samples so nothing here is needed to reproduce a study. This module exists so
the local history can be refreshed without leaving the project, and it is
imported lazily: the ``MetaTrader5`` package is Windows-only and absent from
most environments.

    qmr export-mt5 --symbols EURUSD,GBPUSD,GOLD --timeframes H1,H4,D1
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from qmr.logging_utils import get_logger
from qmr.paths import RAW_DATA_DIR

log = get_logger(__name__)

_TIMEFRAME_ATTRIBUTES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
    "MN1": "TIMEFRAME_MN1",
}


def _import_mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise ImportError(
            "MetaTrader5 is not installed. It is Windows only; install it with:\n"
            "    pip install MetaTrader5\n"
            "Alternatively, drop broker CSV exports into data/raw named "
            "<SYMBOL>_<TIMEFRAME>_<YYYYMMDD>_<YYYYMMDD>.csv."
        ) from exc
    return mt5


def _initialise(mt5) -> None:
    """Connect to the terminal, using .env credentials when they are present."""
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")
    terminal = os.environ.get("MT5_TERMINAL_PATH")

    kwargs: dict[str, object] = {}
    if terminal:
        kwargs["path"] = terminal
    if login and password and server:
        kwargs.update({"login": int(login), "password": password, "server": server})

    if not mt5.initialize(**kwargs):
        raise RuntimeError(
            f"Could not connect to the MetaTrader 5 terminal: {mt5.last_error()}. "
            f"Make sure the terminal is running and, if the account needs it, that "
            f"MT5_LOGIN / MT5_PASSWORD / MT5_SERVER are set in .env."
        )


def export_history(
    symbols: list[str],
    timeframes: list[str],
    bars: int = 200_000,
    destination: Path | None = None,
) -> list[Path]:
    """Pull history for each symbol and timeframe into ``data/raw``."""
    mt5 = _import_mt5()
    destination = destination or RAW_DATA_DIR
    destination.mkdir(parents=True, exist_ok=True)

    _initialise(mt5)
    written: list[Path] = []

    try:
        for symbol in symbols:
            if not mt5.symbol_select(symbol, True):
                log.warning("Symbol %s is not available in this terminal, skipping", symbol)
                continue

            for timeframe in timeframes:
                attribute = _TIMEFRAME_ATTRIBUTES.get(timeframe.upper())
                if attribute is None:
                    log.warning("Unknown timeframe %s, skipping", timeframe)
                    continue

                rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, attribute), 0, bars)
                if rates is None or len(rates) == 0:
                    log.warning("No %s %s bars returned: %s", symbol, timeframe, mt5.last_error())
                    continue

                frame = pd.DataFrame(rates)
                frame["time"] = pd.to_datetime(frame["time"], unit="s")
                frame = frame.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "tick_volume": "Volume",
                    }
                )[["time", "Open", "High", "Low", "Close", "Volume"]]

                name = (
                    f"{symbol.upper()}_{timeframe.upper()}_"
                    f"{frame['time'].iloc[0]:%Y%m%d}_{frame['time'].iloc[-1]:%Y%m%d}.csv"
                )
                path = destination / name

                # The date range is part of the file name, so an updated pull
                # would otherwise accumulate alongside the previous one.
                for stale in destination.glob(f"{symbol.upper()}_{timeframe.upper()}_*.csv"):
                    if stale != path:
                        stale.unlink()

                frame.to_csv(path, index=False)
                written.append(path)
                log.info("Exported %s %s: %d bars -> %s", symbol, timeframe, len(frame), path.name)
    finally:
        mt5.shutdown()

    return written


def terminal_symbols() -> list[str]:
    """Every symbol the connected terminal offers."""
    mt5 = _import_mt5()
    _initialise(mt5)
    try:
        symbols = mt5.symbols_get()
        return sorted(s.name for s in symbols) if symbols else []
    finally:
        mt5.shutdown()
