"""Capture high-resolution screenshots of the research console.

The physical screen is smaller than the console wants to be, so a normal browser
capture comes out short and cropped. Playwright renders to an off-screen
viewport of any size at any device scale factor, which is what makes these
crisp enough to publish.

    python deploy/huggingface/capture_screenshots.py --out ~/Downloads/shots

Assumes the console is already running:  qmr console
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

# Tab label -> (file name, extra scroll in px, capture full page)
SHOTS = [
    ("Overview", "ui_1_overview", 0, False),
    ("Results", "ui_2_results_metrics", 0, False),
    ("Results", "ui_3_results_equity", 900, False),
    ("Regimes", "ui_4_regimes", 700, False),
    ("Signals", "ui_5_signals", 650, False),
    ("Run a study", "ui_6_run_study", 0, False),
]


def capture(url: str, out: Path, width: int, height: int, scale: float) -> int:
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": width, "height": height},
            device_scale_factor=scale,
        )
        print(f"Opening {url} at {width}x{height} @{scale}x ...")
        page.goto(url, wait_until="networkidle", timeout=120_000)
        page.wait_for_timeout(9000)

        written = 0
        for label, name, scroll, full in SHOTS:
            try:
                tab = page.locator(f'button[data-baseweb="tab"]:has-text("{label}")').first
                tab.click(timeout=20_000)
            except Exception as exc:
                print(f"  ! could not open tab {label!r}: {str(exc)[:80]}")
                continue

            # Streamlit re-renders and the charts draw asynchronously.
            page.wait_for_timeout(7000)

            if scroll:
                # Streamlit scrolls an inner container, not the window.
                page.evaluate(
                    """(y) => {
                        const main = document.querySelector('section.main')
                                  || document.querySelector('[data-testid="stAppViewContainer"]')
                                  || document.scrollingElement;
                        main.scrollTop = y;
                        window.scrollTo(0, y);
                    }""",
                    scroll,
                )
                page.wait_for_timeout(2500)

            path = out / f"{name}.png"
            page.screenshot(path=str(path), full_page=full)
            size_kb = path.stat().st_size / 1024
            print(f"  + {path.name}  ({size_kb:.0f} KB)")
            written += 1

        browser.close()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", default=str(Path.home() / "Downloads" / "qmr_screenshots"))
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1150)
    parser.add_argument("--scale", type=float, default=2.0, help="device scale factor")
    args = parser.parse_args()

    n = capture(args.url, Path(args.out), args.width, args.height, args.scale)
    print(f"\n{n} screenshot(s) written to {args.out}")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
