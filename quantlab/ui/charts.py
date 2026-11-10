"""
Altair charts of the UI. Colors follow fixed roles: categorical slots in a fixed order for identity
(runs, strategies), one blue hue for single-series magnitude, blue / red poles for gains / losses.
Times are shown in New York exchange time. Every chart has hover tooltips.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import cast

import altair as alt
import numpy as np
import pandas as pd

#: categorical slots, in this fixed order (validated adjacent-pair CVD order); never cycled
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SINGLE = SERIES[0]
GAIN, LOSS = "#2a78d6", "#e34948"  # diverging poles
NEUTRAL = "#888888"  # crosshair: readable on the light and the dark surface
MAX_POINTS = 4000  # thin long curves before sending them to the browser
Y_AXIS_WIDTH = 56  # pixels kept for the y labels of the charts stacked on the same bars


def exchange_time(index: pd.Index | pd.Series) -> pd.DatetimeIndex:
    """New York wall-clock times (naive) so the browser does not re-convert them to its own zone."""
    times = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    return times.tz_convert("America/New_York").tz_localize(None)


def _thin(frame: pd.DataFrame, max_points: int = MAX_POINTS) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    step = -(-len(frame) // max_points)
    return pd.concat([frame.iloc[::step], frame.iloc[[-1]]]).drop_duplicates()


def equity_chart(curves: Mapping[str, pd.Series], height: int = 280) -> alt.Chart:
    """Equity curves (one per run); a legend appears from two curves on."""
    frames = []
    for label, curve in curves.items():
        frame = pd.DataFrame({"time": exchange_time(curve.index), "equity": curve.to_numpy(), "run": label})
        frames.append(_thin(frame))
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["time", "equity", "run"])
    labels = list(curves)
    color = alt.Color(
        "run:N",
        title=None,
        scale=alt.Scale(domain=labels, range=list(SERIES[: len(labels)])),
        legend=alt.Legend(orient="top") if len(labels) > 1 else None,
    )
    return (
        alt.Chart(data)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time:T", title="New York time"),
            y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(zero=False)),
            color=color,
            tooltip=[
                alt.Tooltip("run:N", title="Run"),
                alt.Tooltip("time:T", title="Time", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("equity:Q", title="Equity", format=",.2f"),
            ],
        )
        .properties(height=height)
        .interactive(bind_y=False)
    )


def drawdown_chart(equity: pd.Series, height: int = 160) -> alt.Chart:
    """Decline from the running peak, in percent (single series)."""
    drawdown = (equity / equity.cummax() - 1) * 100
    data = _thin(pd.DataFrame({"time": exchange_time(equity.index), "drawdown": drawdown.to_numpy()}))
    return (
        alt.Chart(data)
        .mark_area(color=SINGLE, opacity=0.35, line={"color": SINGLE, "strokeWidth": 2})
        .encode(
            x=alt.X("time:T", title="New York time"),
            y=alt.Y("drawdown:Q", title="Drawdown (%)"),
            tooltip=[
                alt.Tooltip("time:T", title="Time", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("drawdown:Q", title="Drawdown (%)", format=".3f"),
            ],
        )
        .properties(height=height)
    )


def trade_pnl_chart(trades: pd.DataFrame, height: int = 220) -> alt.Chart:
    """Net P&L of each trade in exit order: blue gains, red losses."""
    data = trades.reset_index(drop=True).copy()
    data["trade"] = data.index + 1
    data["result"] = ["Gain" if v > 0 else "Loss" for v in data["pnl_net"]]
    data["exit"] = exchange_time(data["exit_time"]).strftime("%Y-%m-%d %H:%M")
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("trade:O", title="Trade #", axis=alt.Axis(labelOverlap=True)),
            y=alt.Y("pnl_net:Q", title="P&L net ($)"),
            color=alt.Color(
                "result:N",
                title=None,
                scale=alt.Scale(domain=["Gain", "Loss"], range=[GAIN, LOSS]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("trade:O", title="Trade"),
                alt.Tooltip("strategy_id:N", title="Strategy"),
                alt.Tooltip("symbol:N", title="Symbol"),
                alt.Tooltip("side:N", title="Side"),
                alt.Tooltip("exit:N", title="Exit"),
                alt.Tooltip("reason:N", title="Reason"),
                alt.Tooltip("pnl_net:Q", title="P&L net", format=",.2f"),
            ],
        )
        .properties(height=height)
    )


def _label_axis(ticks: Sequence[str] | None) -> alt.Axis:
    """Axis of a bar label: every label, or only `ticks` written flat when the bars are many."""
    return alt.Axis(labelLimit=220) if ticks is None else alt.Axis(values=list(ticks), labelAngle=0, labelOverlap=True)


def value_bars(
    values: pd.Series,
    title: str,
    label: str,
    height: int = 220,
    fmt: str = ",.4f",
    ticks: Sequence[str] | None = None,
) -> alt.Chart:
    """Single-series bars (fold scores, parameter counts, time-of-day profiles…); `ticks`: the labels to write."""
    data = pd.DataFrame({"label": [str(i) for i in values.index], "value": values.to_numpy(dtype=float)})
    return (
        alt.Chart(data)
        .mark_bar(color=SINGLE, cornerRadiusEnd=4)
        .encode(
            x=alt.X("label:N", title=label, sort=None, axis=_label_axis(ticks)),
            y=alt.Y("value:Q", title=title),
            tooltip=[alt.Tooltip("label:N", title=label), alt.Tooltip("value:Q", title=title, format=fmt)],
        )
        .properties(height=height)
    )


def signed_bars(
    values: pd.Series,
    title: str,
    label: str,
    height: int = 220,
    fmt: str = "+,.2f",
    names: tuple[str, str] = ("Up", "Down"),
) -> alt.Chart:
    """Bars of a signed value (session returns…): blue above zero, red below."""
    data = pd.DataFrame({"label": [str(i) for i in values.index], "value": values.to_numpy(dtype=float)})
    data["sign"] = [names[0] if v > 0 else names[1] for v in data["value"]]
    ticks = list(data["label"]) if len(data) > 12 else None
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("label:N", title=label, sort=None, axis=_label_axis(ticks)),
            y=alt.Y("value:Q", title=title),
            color=alt.Color(
                "sign:N",
                title=None,
                scale=alt.Scale(domain=list(names), range=[GAIN, LOSS]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[alt.Tooltip("label:N", title=label), alt.Tooltip("value:Q", title=title, format=fmt)],
        )
        .properties(height=height)
    )


def histogram(values: pd.Series, title: str, bins: int = 50, height: int = 260, fmt: str = ",.1f") -> alt.Chart:
    """Distribution of a single series: number of observations per bin."""
    counts, edges = np.histogram(values.to_numpy(dtype=float), bins=bins)
    data = pd.DataFrame({"low": edges[:-1], "high": edges[1:], "count": counts})
    return (
        alt.Chart(data)
        .mark_bar(color=SINGLE, cornerRadiusEnd=3)
        .encode(
            x=alt.X("low:Q", title=title, bin="binned"),
            x2="high:Q",
            y=alt.Y("count:Q", title="Bars", stack=None),  # stacking would drop the x range
            tooltip=[
                alt.Tooltip("low:Q", title="From", format=fmt),
                alt.Tooltip("high:Q", title="To", format=fmt),
                alt.Tooltip("count:Q", title="Bars", format=","),
            ],
        )
        .properties(height=height)
    )


# --- bars on a trading-time axis ---------------------------------------------------------------------------
# The x scale is the position of the bar in the data, labeled with its time: nights and weekends take no room.


def _bar_ticks(times: pd.DatetimeIndex, max_ticks: int = 10) -> dict[int, str]:
    """Tick positions and labels: the first bar of the sessions, or of the hours when there are few sessions."""
    days = times.normalize()
    starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    if len(starts) > 2:
        day_format = "%Y-%m-%d" if times[0].year != times[-1].year else "%b %d"
        return {int(i): times[i].strftime(day_format) for i in starts[:: -(-len(starts) // max_ticks)]}
    hours = np.flatnonzero(np.r_[True, (times.hour[1:] != times.hour[:-1]) | (days[1:] != days[:-1])])
    first = set(starts.tolist())
    picked = sorted(first | set(hours[:: -(-len(hours) // max_ticks)].tolist()))
    return {i: times[i].strftime("%b %d %H:%M" if i in first else "%H:%M") for i in picked}


def _bar_x(times: pd.DatetimeIndex, field: str, title: str | None) -> alt.X:
    ticks = _bar_ticks(times)
    labels = json.dumps({str(position): label for position, label in ticks.items()})
    return alt.X(
        f"{field}:Q",
        title=title,
        scale=alt.Scale(domain=[-0.5, len(times) - 0.5], nice=False, zero=False),
        axis=alt.Axis(values=list(ticks), labelExpr=f"{labels}[datum.value]", labelOverlap=True),
    )


def _value_axis(fmt: str | None = None) -> alt.Axis:
    """A y axis of fixed width, so charts stacked on the same bars stay aligned."""
    return alt.Axis(minExtent=Y_AXIS_WIDTH, maxExtent=Y_AXIS_WIDTH, format=alt.Undefined if fmt is None else fmt)


def price_chart(
    series: Mapping[str, pd.Series], title: str, x_title: str | None = "New York time", height: int = 340
) -> alt.LayerChart:
    """
    Lines of one value per symbol (same New York index) on the trading-time axis, with a crosshair that
    reads every line at the hovered bar. A legend appears from two lines on.
    """
    wide = pd.DataFrame(dict(series))
    times = pd.DatetimeIndex(wide.index)
    labels = [str(c) for c in wide.columns]
    fields = [f"s{i}" for i in range(len(labels))]  # symbols may hold characters that field names cannot
    data = wide.set_axis(fields, axis=1).reset_index(drop=True)
    data["bar"] = np.arange(len(data))
    data["time"] = times.strftime("%Y-%m-%d %H:%M")
    data = _thin(data)
    names = json.dumps(dict(zip(fields, labels, strict=True)))
    lines = (
        alt.Chart(data)
        .transform_fold([*fields], as_=["field", "value"])
        .transform_calculate(symbol=f"{names}[datum.field]")
        .mark_line(strokeWidth=2)
        .encode(
            x=_bar_x(times, "bar", x_title),
            y=alt.Y("value:Q", title=title, scale=alt.Scale(zero=False), axis=_value_axis()),
            color=alt.Color(
                "symbol:N",
                title=None,
                scale=alt.Scale(domain=labels, range=list(SERIES[: len(labels)])),
                legend=alt.Legend(orient="top") if len(labels) > 1 else None,
            ),
        )
    )
    hover = alt.selection_point(fields=["bar"], nearest=True, on="pointerover", clear="pointerout", empty=False)
    crosshair = (
        alt.Chart(data)
        .mark_rule(color=NEUTRAL, strokeWidth=1)
        .encode(
            x=alt.X("bar:Q"),
            opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
            tooltip=[
                alt.Tooltip("time:N", title="Time"),
                *[alt.Tooltip(f"{f}:Q", title=name, format=",.2f") for f, name in zip(fields, labels, strict=True)],
            ],
        )
        .add_params(hover)
    )
    return cast(alt.LayerChart, alt.layer(lines, crosshair, height=height))


def volume_chart(volume: pd.Series, max_bars: int = 120, height: int = 170) -> alt.Chart:
    """
    Volume per bar under a price chart, on the same axis. Long periods are averaged into about `max_bars`
    bars, which split each session (or span several sessions when there are more sessions than bars).
    """
    times = pd.DatetimeIndex(volume.index)
    position = np.arange(len(volume))
    opens = np.r_[True, times.normalize()[1:] != times.normalize()[:-1]]  # first bar of each session
    session = np.cumsum(opens) - 1
    session_bars = float(np.median(np.bincount(session)))
    step = -(-len(volume) // max_bars)
    if step < session_bars:
        step = int(-(-session_bars // -(-session_bars // step)))
        offset = position - np.maximum.accumulate(np.where(opens, position, 0))
        bucket = session * len(volume) + offset // step
    else:
        bucket = position // step
    frame = pd.DataFrame({"bar": position, "time": times.strftime("%Y-%m-%d %H:%M"), "volume": volume.to_numpy()})
    data = frame.groupby(bucket).agg(
        first=("bar", "first"),
        last=("bar", "last"),
        since=("time", "first"),
        until=("time", "last"),
        volume=("volume", "mean"),
    )
    gap = 0.1 * (data["last"] - data["first"] + 1)  # a thin surface gap between neighbors
    data["x0"], data["x1"] = data["first"] - 0.5 + gap, data["last"] + 0.5 - gap
    return (
        alt.Chart(data)
        .mark_bar(color=SINGLE, cornerRadiusEnd=2, orient="vertical")
        .encode(
            x=_bar_x(times, "x0", "New York time"),
            x2="x1:Q",
            y=alt.Y("volume:Q", title="Volume", axis=_value_axis("~s"), stack=None),
            tooltip=[
                alt.Tooltip("since:N", title="From"),
                alt.Tooltip("until:N", title="To"),
                alt.Tooltip("volume:Q", title="Volume per bar", format=",.0f"),
            ],
        )
        .properties(height=height)
    )


def time_line(times: pd.Series, values: pd.Series, title: str, height: int = 160, fmt: str = ",.2f") -> alt.Chart:
    """Single-series line over time (latency, live equity…)."""
    data = _thin(pd.DataFrame({"time": exchange_time(pd.Index(times)), "value": values.to_numpy(dtype=float)}))
    return (
        alt.Chart(data)
        .mark_line(color=SINGLE, strokeWidth=2)
        .encode(
            x=alt.X("time:T", title="New York time"),
            y=alt.Y("value:Q", title=title, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("time:T", title="Time", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("value:Q", title=title, format=fmt),
            ],
        )
        .properties(height=height)
    )
