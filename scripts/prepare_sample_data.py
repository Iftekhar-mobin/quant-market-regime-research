"""Build the trimmed sample datasets that ship with the repository.

The full history (hundreds of megabytes of broker CSV) is deliberately kept out
of version control. This script cuts the most recent N bars from the local
history in ``data/raw`` and writes them to ``data/samples`` so that a fresh
clone can run the whole pipeline and the research console end to end.

    python scripts/prepare_sample_data.py --bars 15000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qmr.data.loader import read_ohlcv_csv  # noqa: E402
from qmr.logging_utils import configure_logging, get_logger  # noqa: E402
from qmr.paths import RAW_DATA_DIR, SAMPLE_DATA_DIR  # noqa: E402

log = get_logger("prepare_sample_data")

# One major, one cross-heavy major and one commodity: enough variety for the
# regime analysis to show genuinely different market structure.
DEFAULT_DATASETS = [("EURUSD", "H1"), ("GBPUSD", "H1"), ("GOLD", "H1")]


def build_sample(symbol: str, timeframe: str, bars: int) -> Path | None:
    matches = sorted(RAW_DATA_DIR.glob(f"{symbol}_{timeframe}_*.csv"))
    if not matches:
        log.warning("No local history for %s %s, skipping", symbol, timeframe)
        return None

    frame = read_ohlcv_csv(matches[-1]).iloc[-bars:]
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    name = f"{symbol}_{timeframe}_{frame.index[0]:%Y%m%d}_{frame.index[-1]:%Y%m%d}.csv"
    destination = SAMPLE_DATA_DIR / name

    # Remove stale samples for the same symbol/timeframe: the date range, and
    # therefore the file name, changes every time the sample is rebuilt.
    for stale in SAMPLE_DATA_DIR.glob(f"{symbol}_{timeframe}_*.csv"):
        if stale != destination:
            stale.unlink()

    out = frame.reset_index().rename(columns={"timestamp": "time"})
    out.to_csv(destination, index=False, float_format="%.6f")

    log.info(
        "%s %s -> %s (%d bars, %.1f MB)",
        symbol,
        timeframe,
        destination.name,
        len(out),
        destination.stat().st_size / 1024 / 1024,
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=15_000, help="bars to keep per dataset")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="SYMBOL:TIMEFRAME",
        help="dataset to sample, repeatable (default: EURUSD:H1 GBPUSD:H1 GOLD:H1)",
    )
    args = parser.parse_args()
    configure_logging()

    datasets = DEFAULT_DATASETS
    if args.dataset:
        datasets = [tuple(item.split(":", 1)) for item in args.dataset]

    written = [build_sample(symbol, timeframe, args.bars) for symbol, timeframe in datasets]
    return 0 if any(written) else 1


if __name__ == "__main__":
    raise SystemExit(main())
