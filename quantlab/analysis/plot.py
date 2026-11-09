"""Candlestick chart of one session with the VWAP bands and the best support / resistance lines."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quantlab.features.levels import resistance_lines, support_lines
from quantlab.features.technical import VWAP

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _line_series(index: pd.Index, line: pd.Series) -> pd.Series:
    """NaN everywhere except on the bars covered by a support / resistance line."""
    series = pd.Series(index=index, data=np.nan)
    for idx in range(int(line["idxStart"]), min(int(line["idxEnd"]) + 1, len(index))):
        if idx in line["points"]:
            series.iloc[idx] = line["points"][idx]
    return series


def session_bars(data: pd.DataFrame, day: str | None = None) -> tuple[pd.DataFrame, date]:
    """The bars of one New York session (default: the last one), indexed in New York time."""
    index = pd.DatetimeIndex(data.index)
    index = (index.tz_localize("UTC") if index.tz is None else index).tz_convert("America/New_York")
    chosen = pd.Timestamp(day).date() if day else index[-1].date()
    mask = index.date == chosen
    if not mask.any():
        raise ValueError(f"No bar on {chosen}.")
    return data[mask].set_axis(index[mask]), chosen


def plot_levels(data: pd.DataFrame, title: str, n_lines: int = 3, output: str | None = None) -> None:
    """Show the chart (or save it to `output`)."""
    import matplotlib.pyplot as plt

    figure = levels_figure(data, title, n_lines)
    if output:
        figure.savefig(output, dpi=120)
    else:
        plt.show()


def levels_figure(data: pd.DataFrame, title: str, n_lines: int = 3) -> Figure:
    """Candles, volume, VWAP bands and support / resistance lines (these look ahead: charting only)."""
    import mplfinance as mpf

    vwap = VWAP().compute(data, {})
    plots: list[Any] = [mpf.make_addplot(vwap["vwap"], color="orange", width=2, label="VWAP")]
    for column, style, label in (
        ("vwap_upper_3", "--", "Upper band"),
        ("vwap_far_upper_3", ":", "Far upper band"),
        ("vwap_lower_3", "--", "Lower band"),
        ("vwap_far_lower_3", ":", "Far lower band"),
    ):
        plots.append(mpf.make_addplot(vwap[column], color="green", linestyle=style, label=label))
    for lines, color, name in ((resistance_lines(data), "red", "Resistance"), (support_lines(data), "blue", "Support")):
        for rank, (_, line) in enumerate(lines.head(n_lines).iterrows(), start=1):
            series = _line_series(data.index, line)
            plots.append(mpf.make_addplot(series, color=color, width=1.5, label=f"{name} {rank}"))

    figure, axes = mpf.plot(
        data.rename(columns=str.title),  # mplfinance expects Open/High/Low/Close/Volume
        type="candle",
        style="yahoo",
        figsize=(12, 8),
        title=title,
        ylabel="Price ($)",
        addplot=plots,
        volume=True,
        show_nontrading=False,
        tight_layout=True,
        returnfig=True,
        warn_too_much_data=100_000,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper left", fontsize=8)
    result: Figure = figure
    return result
