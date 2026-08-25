"""Discovery of the local market-history files.

History is stored as one CSV per symbol and timeframe, named

    <SYMBOL>_<TIMEFRAME>_<YYYYMMDD>_<YYYYMMDD>.csv

which lets the catalogue answer "what do I have, and over what period" without
opening a single file. Anything that does not match the pattern is still picked
up as long as the leading ``SYMBOL_TIMEFRAME`` prefix is present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from qmr.logging_utils import get_logger
from qmr.paths import data_search_paths

log = get_logger(__name__)

# Ordered from fastest to slowest so UI dropdowns read naturally.
TIMEFRAME_ORDER = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]

TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10_080,
    "MN1": 43_200,
}

# Bars a year, used to annualise performance metrics. FX trades ~24h a day,
# five days a week, ~260 weekdays a year.
TIMEFRAME_BARS_PER_YEAR = {
    tf: max(1, int(round(260 * 24 * 60 / minutes))) if minutes < 1440 else 260
    for tf, minutes in TIMEFRAME_MINUTES.items()
}
TIMEFRAME_BARS_PER_YEAR["W1"] = 52
TIMEFRAME_BARS_PER_YEAR["MN1"] = 12

_FILENAME_RE = re.compile(
    r"^(?P<symbol>[A-Za-z0-9]+)_(?P<timeframe>[A-Za-z0-9]+)"
    r"(?:_(?P<start>\d{8})_(?P<end>\d{8}))?.*\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DatasetInfo:
    """One discovered history file."""

    symbol: str
    timeframe: str
    path: Path
    start: date | None
    end: date | None
    size_bytes: int

    @property
    def label(self) -> str:
        return f"{self.symbol} {self.timeframe}"

    @property
    def coverage(self) -> str:
        if self.start and self.end:
            return f"{self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}"
        return "unknown"

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


def _parse_date(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y%m%d").date()
    except ValueError:
        return None


def _timeframe_rank(timeframe: str) -> int:
    try:
        return TIMEFRAME_ORDER.index(timeframe.upper())
    except ValueError:
        return len(TIMEFRAME_ORDER)


def scan_catalog(directories: list[Path] | None = None) -> list[DatasetInfo]:
    """Return every history file found, newest coverage first within a symbol.

    When the same symbol/timeframe exists in several directories the first
    match in ``directories`` order wins, so a full local history in
    ``data/raw`` shadows the trimmed sample shipped in ``data/samples``.
    """
    directories = directories or data_search_paths()
    found: dict[tuple[str, str], DatasetInfo] = {}

    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.csv")):
            match = _FILENAME_RE.match(path.name)
            if not match:
                log.debug("Skipping unrecognised file name: %s", path.name)
                continue
            symbol = match.group("symbol").upper()
            timeframe = match.group("timeframe").upper()
            if timeframe not in TIMEFRAME_MINUTES:
                log.debug("Skipping unknown timeframe %s in %s", timeframe, path.name)
                continue
            key = (symbol, timeframe)
            if key in found:
                continue
            found[key] = DatasetInfo(
                symbol=symbol,
                timeframe=timeframe,
                path=path,
                start=_parse_date(match.group("start")),
                end=_parse_date(match.group("end")),
                size_bytes=path.stat().st_size,
            )

    return sorted(found.values(), key=lambda d: (d.symbol, _timeframe_rank(d.timeframe)))


@lru_cache(maxsize=1)
def _cached_catalog_key() -> None:  # pragma: no cover - cache invalidation hook
    return None


def find_dataset(symbol: str, timeframe: str) -> DatasetInfo:
    """Locate one dataset, raising a helpful error when it is missing."""
    symbol, timeframe = symbol.upper(), timeframe.upper()
    catalog = scan_catalog()
    for item in catalog:
        if item.symbol == symbol and item.timeframe == timeframe:
            return item

    known = ", ".join(sorted({f"{d.symbol} {d.timeframe}" for d in catalog})) or "nothing"
    raise FileNotFoundError(
        f"No history found for {symbol} {timeframe}. Available datasets: {known}. "
        f"Place CSV files named <SYMBOL>_<TIMEFRAME>_<YYYYMMDD>_<YYYYMMDD>.csv "
        f"under data/raw, or run `qmr export-mt5` to pull them from MetaTrader 5."
    )


def available_symbols() -> list[str]:
    return sorted({item.symbol for item in scan_catalog()})


def available_timeframes(symbol: str | None = None) -> list[str]:
    catalog = scan_catalog()
    if symbol:
        catalog = [item for item in catalog if item.symbol == symbol.upper()]
    return sorted({item.timeframe for item in catalog}, key=_timeframe_rank)


def catalog_frame(directories: list[Path] | None = None) -> pd.DataFrame:
    """The catalogue as a table, for display in the research console."""
    rows = [
        {
            "Symbol": item.symbol,
            "Timeframe": item.timeframe,
            "Coverage": item.coverage,
            "Size (MB)": round(item.size_mb, 1),
            "File": item.path.name,
            "Source": item.path.parent.name,
        }
        for item in scan_catalog(directories)
    ]
    return pd.DataFrame(rows)


def bars_per_year(timeframe: str) -> int:
    return TIMEFRAME_BARS_PER_YEAR.get(timeframe.upper(), 6240)
