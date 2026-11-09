"""
Altair charts of the UI. Colors follow fixed roles: categorical slots in a fixed order for identity
(runs, strategies), one blue hue for single-series magnitude, blue / red poles for gains / losses.
Times are shown in New York exchange time. Every chart has hover tooltips.
"""

from __future__ import annotations

from collections.abc import Mapping

import altair as alt
import pandas as pd

#: categorical slots, in this fixed order (validated adjacent-pair CVD order); never cycled
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SINGLE = SERIES[0]
GAIN, LOSS = "#2a78d6", "#e34948"  # diverging poles
MAX_POINTS = 4000  # thin long curves before sending them to the browser


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
            x=alt.X("time:T", title="Heure de New York"),
            y=alt.Y("equity:Q", title="Capital ($)", scale=alt.Scale(zero=False)),
            color=color,
            tooltip=[
                alt.Tooltip("run:N", title="Run"),
                alt.Tooltip("time:T", title="Heure", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("equity:Q", title="Capital", format=",.2f"),
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
            x=alt.X("time:T", title="Heure de New York"),
            y=alt.Y("drawdown:Q", title="Drawdown (%)"),
            tooltip=[
                alt.Tooltip("time:T", title="Heure", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("drawdown:Q", title="Drawdown (%)", format=".3f"),
            ],
        )
        .properties(height=height)
    )


def trade_pnl_chart(trades: pd.DataFrame, height: int = 220) -> alt.Chart:
    """Net P&L of each trade in exit order: blue gains, red losses."""
    data = trades.reset_index(drop=True).copy()
    data["trade"] = data.index + 1
    data["result"] = ["Gain" if v > 0 else "Perte" for v in data["pnl_net"]]
    data["exit"] = exchange_time(data["exit_time"]).strftime("%Y-%m-%d %H:%M")
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("trade:O", title="Trade n°", axis=alt.Axis(labelOverlap=True)),
            y=alt.Y("pnl_net:Q", title="P&L net ($)"),
            color=alt.Color(
                "result:N",
                title=None,
                scale=alt.Scale(domain=["Gain", "Perte"], range=[GAIN, LOSS]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("trade:O", title="Trade"),
                alt.Tooltip("strategy_id:N", title="Stratégie"),
                alt.Tooltip("symbol:N", title="Symbole"),
                alt.Tooltip("side:N", title="Sens"),
                alt.Tooltip("exit:N", title="Sortie"),
                alt.Tooltip("reason:N", title="Raison"),
                alt.Tooltip("pnl_net:Q", title="P&L net", format=",.2f"),
            ],
        )
        .properties(height=height)
    )


def value_bars(values: pd.Series, title: str, label: str, height: int = 220, fmt: str = ",.4f") -> alt.Chart:
    """Single-series bars (fold scores, parameter counts, feature differences…)."""
    data = pd.DataFrame({"label": [str(i) for i in values.index], "value": values.to_numpy(dtype=float)})
    return (
        alt.Chart(data)
        .mark_bar(color=SINGLE, cornerRadiusEnd=4)
        .encode(
            x=alt.X("label:N", title=label, sort=None, axis=alt.Axis(labelLimit=220)),
            y=alt.Y("value:Q", title=title),
            tooltip=[alt.Tooltip("label:N", title=label), alt.Tooltip("value:Q", title=title, format=fmt)],
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
            x=alt.X("time:T", title="Heure de New York"),
            y=alt.Y("value:Q", title=title, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("time:T", title="Heure", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("value:Q", title=title, format=fmt),
            ],
        )
        .properties(height=height)
    )
