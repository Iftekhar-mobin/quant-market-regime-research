"""Visual system for the research console.

One palette, one Plotly template, one set of chart builders. Every figure in the
console is produced by a function in this module, which is what keeps forty
charts across seven tabs reading as one instrument rather than seven.

Palette rules in force
----------------------
* Categorical hues are assigned in a fixed order and never cycled; a series
  keeps its colour when a filter removes its neighbours.
* Magnitude uses a single hue, light to dark. Polarity (a fold that made money
  against one that lost) uses two opposing hues with a neutral midpoint.
* Status colours are reserved for state and never stand in for a series.
* Marks are thin, grids are hairlines, and no chart carries two y-axes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BORDER = "rgba(11,11,11,0.10)"

# Fixed categorical order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Single-hue magnitude ramp (blue, light to dark).
SEQUENTIAL = [
    [0.0, "#f4f8fe"],
    [0.15, "#cde2fb"],
    [0.3, "#9ec5f4"],
    [0.45, "#6da7ec"],
    [0.6, "#3987e5"],
    [0.75, "#256abf"],
    [0.9, "#184f95"],
    [1.0, "#0d366b"],
]

# Polarity: two opposing hues with a neutral midpoint.
DIVERGING = [
    [0.0, "#8f2020"],
    [0.2, "#d03b3b"],
    [0.4, "#f0b8b8"],
    [0.5, "#f0efec"],
    [0.6, "#9ec5f4"],
    [0.8, "#2a78d6"],
    [1.0, "#0d366b"],
]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

POSITIVE = "#2a78d6"
NEGATIVE = "#d03b3b"

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Regime colours come from the categorical order, so regime 0 is always the same
# hue in every chart in the console.
REGIME_COLOURS = SERIES


def register_template() -> str:
    """Install the console Plotly template and return its name."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT_FAMILY, size=13, color=INK_SECONDARY),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", pad=dict(b=12)),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=SERIES,
        margin=dict(l=56, r=24, t=48, b=44),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor=AXIS,
            font=dict(family=FONT_FAMILY, size=12, color=INK),
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, color=INK_SECONDARY),
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor=AXIS,
            linewidth=1,
            ticks="outside",
            ticklen=4,
            tickcolor=AXIS,
            tickfont=dict(size=11, color=INK_MUTED),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID,
            gridwidth=1,
            zeroline=False,
            showline=False,
            ticks="",
            tickfont=dict(size=11, color=INK_MUTED),
        ),
    )
    pio.templates["qmr"] = template
    pio.templates.default = "qmr"
    return "qmr"


APP_CSS = f"""
<style>
  .stApp {{ background: {PLANE}; }}
  .block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
  h1, h2, h3, h4 {{ color: {INK}; font-family: {FONT_FAMILY}; letter-spacing: -0.01em; }}
  h1 {{ font-size: 1.65rem; font-weight: 620; }}
  h2 {{ font-size: 1.2rem; font-weight: 600; margin-top: 0.4rem; }}
  h3 {{ font-size: 1.02rem; font-weight: 600; }}

  /* Lede paragraph that opens each tab: states what the view is for. */
  .qmr-lede {{
    color: {INK_SECONDARY}; font-size: 0.94rem; line-height: 1.55;
    max-width: 68ch; margin: 0.1rem 0 1.2rem 0;
  }}
  .qmr-note {{
    color: {INK_MUTED}; font-size: 0.83rem; line-height: 1.5;
    border-left: 2px solid {GRID}; padding-left: 0.75rem; margin: 0.6rem 0 1rem 0;
  }}

  /* Stat tiles: a headline number is not a one-bar bar chart. */
  .qmr-tiles {{ display: flex; flex-wrap: wrap; gap: 0.7rem; margin: 0.4rem 0 1.3rem 0; }}
  .qmr-tile {{
    flex: 1 1 168px; background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 0.85rem 1rem;
  }}
  .qmr-tile-label {{
    color: {INK_MUTED}; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.055em;
  }}
  .qmr-tile-value {{
    color: {INK}; font-size: 1.5rem; font-weight: 600; line-height: 1.25; margin-top: 0.2rem;
  }}
  .qmr-tile-help {{ color: {INK_MUTED}; font-size: 0.76rem; margin-top: 0.15rem; }}
  .qmr-pos {{ color: #006300; }}
  .qmr-neg {{ color: {NEGATIVE}; }}

  .stTabs [data-baseweb="tab-list"] {{ gap: 0.35rem; border-bottom: 1px solid {GRID}; }}
  .stTabs [data-baseweb="tab"] {{
    height: 2.5rem; padding: 0 0.95rem; font-size: 0.9rem; color: {INK_SECONDARY};
  }}
  .stTabs [aria-selected="true"] {{ color: {INK}; font-weight: 600; }}

  [data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
  [data-testid="stMetricValue"] {{ font-size: 1.35rem; }}
  hr {{ border-color: {GRID}; }}
  code {{ color: {INK}; background: #f0efec; }}
</style>
"""


# ---------------------------------------------------------------------------
# Small display helpers
# ---------------------------------------------------------------------------
def format_metric(key: str, value: float | None) -> str:
    """Format one metric for a stat tile."""
    if value is None or not np.isfinite(value):
        return "n/a"

    percent = {
        "total_return",
        "cagr",
        "annualised_volatility",
        "max_drawdown",
        "hit_rate",
        "exposure",
        "trade_win_rate",
        "directional_precision",
        "accuracy",
        "signal_rate",
        "balanced_accuracy",
    }
    if key in percent:
        return f"{value * 100:.1f}%"
    if key in {"trades", "bars", "signals_taken", "oos_bars", "longest_drawdown_bars"}:
        return f"{value:,.0f}"
    return f"{value:.2f}"


def stat_tiles(tiles: list[tuple[str, str, str]]) -> str:
    """Render a row of stat tiles as HTML.

    ``tiles`` is a list of ``(label, value, help_text)``. A value prefixed with
    ``+`` or ``-`` is tinted, so direction is legible without reading the digits.
    """
    cells = []
    for label, value, help_text in tiles:
        tone = ""
        if value.startswith("+"):
            tone = " qmr-pos"
        elif value.startswith("-") and value not in {"-", "-n/a"}:
            tone = " qmr-neg"
        cells.append(
            f'<div class="qmr-tile">'
            f'<div class="qmr-tile-label">{label}</div>'
            f'<div class="qmr-tile-value{tone}">{value}</div>'
            f'<div class="qmr-tile-help">{help_text}</div>'
            f"</div>"
        )
    return f'<div class="qmr-tiles">{"".join(cells)}</div>'


def lede(text: str) -> str:
    return f'<p class="qmr-lede">{text}</p>'


def note(text: str) -> str:
    return f'<div class="qmr-note">{text}</div>'


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def _finish(figure: go.Figure, height: int, title: str | None = None) -> go.Figure:
    figure.update_layout(height=height, title=title)
    return figure


def price_chart(
    price: pd.DataFrame,
    regimes: pd.Series | None = None,
    regime_names: dict[int, str] | None = None,
    signals: pd.Series | None = None,
    overlays: dict[str, pd.Series] | None = None,
    title: str | None = None,
    height: int = 460,
) -> go.Figure:
    """Price with optional regime shading, moving-average overlays and signals."""
    figure = go.Figure()

    # Regime bands sit behind everything, at low opacity: they are context, not
    # a series, so they must never compete with the price line.
    if regimes is not None and not regimes.empty:
        names = regime_names or {}
        changed = regimes.ne(regimes.shift())
        block_id = changed.cumsum()
        for _, block in regimes.groupby(block_id):
            state = int(block.iloc[0])
            figure.add_vrect(
                x0=block.index[0],
                x1=block.index[-1],
                fillcolor=REGIME_COLOURS[state % len(REGIME_COLOURS)],
                opacity=0.10,
                line_width=0,
                layer="below",
            )
        # One invisible trace per regime so the legend explains the shading.
        for state in sorted(regimes.unique()):
            figure.add_trace(
                go.Scatter(
                    x=[price.index[0]],
                    y=[None],
                    mode="markers",
                    marker=dict(size=9, color=REGIME_COLOURS[int(state) % len(REGIME_COLOURS)]),
                    name=names.get(int(state), f"Regime {int(state)}"),
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    figure.add_trace(
        go.Scatter(
            x=price.index,
            y=price["close"],
            mode="lines",
            name="Close",
            line=dict(color=INK_SECONDARY, width=1.4),
            hovertemplate="%{y:.5f}<extra>Close</extra>",
        )
    )

    for i, (label, series) in enumerate((overlays or {}).items()):
        figure.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode="lines",
                name=label,
                line=dict(color=SERIES[i % len(SERIES)], width=1.6),
                hovertemplate="%{y:.5f}<extra>" + label + "</extra>",
            )
        )

    if signals is not None and not signals.empty:
        entries = signals[signals.ne(signals.shift()) & signals.ne(0)]
        for value, colour, symbol, label in (
            (1, POSITIVE, "triangle-up", "Long entry"),
            (-1, NEGATIVE, "triangle-down", "Short entry"),
        ):
            marks = entries[entries == value]
            if marks.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=marks.index,
                    y=price["close"].reindex(marks.index),
                    mode="markers",
                    name=label,
                    marker=dict(
                        color=colour,
                        size=10,
                        symbol=symbol,
                        # A 2px surface ring, not a border: it separates
                        # overlapping markers without adding a stroke colour.
                        line=dict(color=SURFACE, width=2),
                    ),
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.5f}<extra>" + label + "</extra>",
                )
            )

    figure.update_layout(hovermode="x unified", yaxis_title=None)
    return _finish(figure, height, title)


def equity_chart(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    title: str | None = None,
    height: int = 380,
) -> go.Figure:
    """Strategy equity against buy-and-hold, indexed to a common base of 100.

    Indexing to a common base is what allows both series to share one y-axis;
    a second axis would invent a relationship between them that the data does
    not contain.
    """
    figure = go.Figure()

    normalised = equity / equity.iloc[0] * 100
    figure.add_trace(
        go.Scatter(
            x=normalised.index,
            y=normalised,
            mode="lines",
            name="Strategy",
            line=dict(color=SERIES[0], width=2),
            hovertemplate="%{y:.1f}<extra>Strategy</extra>",
        )
    )

    if benchmark is not None and not benchmark.empty:
        benchmark_normalised = benchmark / benchmark.iloc[0] * 100
        figure.add_trace(
            go.Scatter(
                x=benchmark_normalised.index,
                y=benchmark_normalised,
                mode="lines",
                name="Buy and hold",
                line=dict(color=INK_MUTED, width=1.6, dash="dot"),
                hovertemplate="%{y:.1f}<extra>Buy and hold</extra>",
            )
        )

    figure.add_hline(y=100, line=dict(color=AXIS, width=1))
    figure.update_layout(yaxis_title="Index (start = 100)")
    return _finish(figure, height, title)


def drawdown_chart(
    drawdown: pd.Series, title: str | None = None, height: int = 240
) -> go.Figure:
    """Underwater curve. One series, so no legend box is needed."""
    figure = go.Figure(
        go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            mode="lines",
            name="Drawdown",
            line=dict(color=NEGATIVE, width=1.4),
            fill="tozeroy",
            fillcolor="rgba(208,59,59,0.12)",
            hovertemplate="%{y:.2f}%<extra>Drawdown</extra>",
            showlegend=False,
        )
    )
    figure.update_layout(yaxis_title="Drawdown (%)")
    return _finish(figure, height, title)


def bar_chart(
    labels: list[str],
    values: list[float],
    title: str | None = None,
    height: int = 340,
    orientation: str = "v",
    diverging: bool = False,
    value_format: str = ".2f",
    axis_title: str | None = None,
) -> go.Figure:
    """Single-series bars.

    ``diverging=True`` colours by sign — the one case where colour on a single
    series carries information the bar length does not already make obvious.
    """
    if diverging:
        colours = [POSITIVE if v >= 0 else NEGATIVE for v in values]
    else:
        colours = [SERIES[0]] * len(values)

    if orientation == "h":
        figure = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(color=colours),
                hovertemplate="%{y}: %{x:" + value_format + "}<extra></extra>",
            )
        )
        figure.update_layout(
            xaxis=dict(showgrid=True, gridcolor=GRID, title=axis_title),
            yaxis=dict(showgrid=False, autorange="reversed"),
            bargap=0.28,
        )
    else:
        figure = go.Figure(
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color=colours),
                hovertemplate="%{x}: %{y:" + value_format + "}<extra></extra>",
            )
        )
        figure.update_layout(yaxis_title=axis_title, bargap=0.32)

    if diverging:
        figure.add_hline(y=0, line=dict(color=AXIS, width=1)) if orientation == "v" else figure.add_vline(
            x=0, line=dict(color=AXIS, width=1)
        )
    figure.update_traces(marker_line_width=0)
    return _finish(figure, height, title)


def grouped_bar_chart(
    frame: pd.DataFrame,
    title: str | None = None,
    height: int = 380,
    axis_title: str | None = None,
    value_format: str = ".2f",
) -> go.Figure:
    """Grouped bars: index on the category axis, one series per column."""
    figure = go.Figure()
    for i, column in enumerate(frame.columns):
        figure.add_trace(
            go.Bar(
                x=[str(v) for v in frame.index],
                y=frame[column],
                name=str(column),
                marker=dict(color=SERIES[i % len(SERIES)], line=dict(width=0)),
                hovertemplate="%{x}<br>" + str(column) + ": %{y:" + value_format + "}<extra></extra>",
            )
        )
    figure.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, yaxis_title=axis_title)
    figure.add_hline(y=0, line=dict(color=AXIS, width=1))
    return _finish(figure, height, title)


def heatmap(
    frame: pd.DataFrame,
    title: str | None = None,
    height: int = 380,
    colorscale: list | None = None,
    value_format: str = ".2f",
    zmid: float | None = None,
    colorbar_title: str = "",
) -> go.Figure:
    """Matrix view with the values printed in each cell."""
    figure = go.Figure(
        go.Heatmap(
            z=frame.to_numpy(dtype=float),
            x=[str(c) for c in frame.columns],
            y=[str(i) for i in frame.index],
            colorscale=colorscale or SEQUENTIAL,
            zmid=zmid,
            text=[[format(v, value_format) for v in row] for row in frame.to_numpy(dtype=float)],
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="%{y} -> %{x}: %{z:" + value_format + "}<extra></extra>",
            # A surface-coloured gap between cells, rather than a border.
            xgap=2,
            ygap=2,
            colorbar=dict(
                title=dict(text=colorbar_title, font=dict(size=11)),
                thickness=12,
                len=0.85,
                outlinewidth=0,
                tickfont=dict(size=10, color=INK_MUTED),
            ),
        )
    )
    figure.update_layout(
        xaxis=dict(showgrid=False, side="bottom"),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    return _finish(figure, height, title)


def line_chart(
    series_map: dict[str, pd.Series],
    title: str | None = None,
    height: int = 340,
    axis_title: str | None = None,
    x_title: str | None = None,
    value_format: str = ".3f",
) -> go.Figure:
    """Several series on one axis."""
    figure = go.Figure()
    for i, (label, series) in enumerate(series_map.items()):
        figure.add_trace(
            go.Scatter(
                x=series.index,
                y=series.to_numpy(),
                mode="lines",
                name=label,
                line=dict(color=SERIES[i % len(SERIES)], width=2),
                hovertemplate="%{y:" + value_format + "}<extra>" + label + "</extra>",
            )
        )
    figure.update_layout(
        yaxis_title=axis_title,
        xaxis_title=x_title,
        showlegend=len(series_map) > 1,
    )
    return _finish(figure, height, title)


def histogram(
    values: pd.Series,
    title: str | None = None,
    height: int = 300,
    bins: int = 60,
    axis_title: str | None = None,
) -> go.Figure:
    """Distribution with the mean marked. One series, so no legend."""
    figure = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=bins,
            marker=dict(color=SERIES[0], line=dict(width=0)),
            hovertemplate="%{x}<br>%{y} bars<extra></extra>",
            showlegend=False,
        )
    )
    mean = float(values.mean())
    figure.add_vline(
        x=mean,
        line=dict(color=INK_SECONDARY, width=1.5, dash="dot"),
        annotation_text=f"mean {mean:.4g}",
        annotation_position="top right",
        annotation_font=dict(size=11, color=INK_SECONDARY),
    )
    figure.update_layout(xaxis_title=axis_title, yaxis_title="Bars", bargap=0.02)
    return _finish(figure, height, title)


def regime_share_chart(
    shares: pd.Series, names: dict[int, str], title: str | None = None, height: int = 300
) -> go.Figure:
    """How much of the sample each regime occupies.

    Bars rather than a pie: the whole point is comparing close values, which is
    exactly what a pie cannot do.
    """
    labels = [names.get(int(i), f"Regime {int(i)}") for i in shares.index]
    colours = [REGIME_COLOURS[int(i) % len(REGIME_COLOURS)] for i in shares.index]

    figure = go.Figure(
        go.Bar(
            x=(shares * 100).to_numpy(),
            y=labels,
            orientation="h",
            marker=dict(color=colours, line=dict(width=0)),
            text=[f"{v * 100:.1f}%" for v in shares],
            textposition="outside",
            textfont=dict(size=11, color=INK_SECONDARY),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_layout(
        xaxis=dict(showgrid=True, gridcolor=GRID, title="Share of bars (%)"),
        yaxis=dict(showgrid=False, autorange="reversed"),
        bargap=0.3,
    )
    return _finish(figure, height, title)


def scatter_chart(
    x: pd.Series,
    y: pd.Series,
    colour_by: pd.Series | None = None,
    names: dict[int, str] | None = None,
    title: str | None = None,
    height: int = 420,
    x_title: str | None = None,
    y_title: str | None = None,
) -> go.Figure:
    """Two descriptors against each other, optionally coloured by regime.

    Capped at the categorical all-pairs limit: past three colours in a scatter,
    adjacent hues stop being separable for colour-vision-deficient readers, so
    the remaining states fold into a neutral "Other".
    """
    figure = go.Figure()

    if colour_by is None:
        figure.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=4, color=SERIES[0], opacity=0.5),
                showlegend=False,
                hovertemplate=f"{x_title}: %{{x:.3f}}<br>{y_title}: %{{y:.3f}}<extra></extra>",
            )
        )
    else:
        states = sorted(int(s) for s in colour_by.unique())
        leading = states[:3]
        other_shown = False

        for state in states:
            mask = (colour_by == state).to_numpy()
            is_leading = state in leading

            if is_leading:
                name = (names or {}).get(state, f"Regime {state}")
                colour = SERIES[leading.index(state)]
                show_in_legend = True
            else:
                name = "Other regimes"
                colour = INK_MUTED
                # One legend entry for the whole folded tail, not one per state.
                show_in_legend = not other_shown
                other_shown = True

            figure.add_trace(
                go.Scattergl(
                    x=x[mask],
                    y=y[mask],
                    mode="markers",
                    name=name,
                    legendgroup=str(state) if is_leading else "other",
                    showlegend=bool(show_in_legend),
                    marker=dict(size=4, color=colour, opacity=0.55),
                    hovertemplate="%{x:.3f}, %{y:.3f}<extra>" + name + "</extra>",
                )
            )

    figure.update_layout(
        xaxis=dict(title=x_title, showgrid=True, gridcolor=GRID),
        yaxis_title=y_title,
        hovermode="closest",
    )
    return _finish(figure, height, title)
