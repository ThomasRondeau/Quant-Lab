"""
quantlab graphical interface (Streamlit): everything the CLI does, plus charts.

    python -m quantlab ui            (or: python -m streamlit run quantlab/ui/app.py)

Pages: Data (analysis of a set of bars), Backtest, Optimization (walk-forward), Live trading (simulated).
The pages call the same library functions as the CLI. With QUANTLAB_HOSTED=1 (the public demo), the inputs and the
computations are restricted (`ui/hosting.py`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # `streamlit run` puts only this file's folder on the path
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from quantlab.cli import format_value, parse_value  # noqa: E402
from quantlab.data import loading  # noqa: E402
from quantlab.execution.risk import RiskLimits  # noqa: E402
from quantlab.strategies.params import (  # noqa: E402
    GROUPS,
    Param,
    ParamError,
    defaults,
    load_preset,
    param_range,
    save_preset,
)
from quantlab.strategies.registry import STRATEGIES, build_strategy  # noqa: E402
from quantlab.ui import charts  # noqa: E402
from quantlab.ui.hosting import ComputeSlots, LimitError, Limits, hosted, preset_slug  # noqa: E402

logger = logging.getLogger("quantlab.ui")

SOURCE_LABELS = {
    "sample": "Bundled sample (2 min, offline)",
    "yfinance": "yfinance (2 min, ~60 days)",
    "file": "File (parquet / csv)",
}
STRATEGY_LABELS = {
    "trend": "Trend following",
    "mean_reversion": "Mean reversion",
    "arima": "ARIMA",
    "ml_hybrid": "ML Hybrid",
    "cascade": "Cascade",
    "pair": "Pair trading (2 symbols)",
}
OBJECTIVES = ("total_return_pct", "sharpe_ratio", "sortino_ratio", "profit_factor", "total_pnl_net", "win_rate")
#: versioned parameter presets, `<strategy>-<name>.json`
PRESETS_DIR = _ROOT / "presets"
LIMITS = Limits()
REPOSITORY_URL = "https://github.com/ThomasRondeau/Quant-Lab"
#: simulated live sessions run from the interface (local runs; the hosted demo uses a temporary folder)
SESSIONS_DIR = _ROOT / "sessions"


@st.cache_resource
def compute_slots() -> ComputeSlots:
    """Shared by every visitor of the server process."""
    return ComputeSlots(size=2, timeout=90.0)


def session_presets(name: str) -> dict[str, dict[str, Any]]:
    """Presets saved by this visitor in the hosted demo (kept in the session, never written on the server)."""
    store: dict[str, dict[str, dict[str, Any]]] = st.session_state.setdefault("user_presets", {})
    return store.setdefault(name, {})


# --- shared widgets ----------------------------------------------------------------------------------------


#: display names of the parameter groups
SECTION_LABELS = {
    "signal": "Signal",
    "indicators": "Indicators",
    "model": "Model",
    "target": "Target",
    "sizing": "Sizing",
    "exits": "Exits",
}
NUMERIC = ("int", "float")


def sheet(name: str) -> tuple[Param, ...]:
    return STRATEGIES[name].params


def group_label(group: str) -> str:
    section, _, model = group.partition(":")
    return SECTION_LABELS.get(section, section) + (f": {model}" if model else "")


def grouped(params: Sequence[Param]) -> dict[str, list[Param]]:
    """Parameters by group, in the GROUPS order (declaration order inside a group)."""
    order = {section: i for i, section in enumerate(GROUPS)}
    groups: dict[str, list[Param]] = {}
    for param in sorted(params, key=lambda p: order.get(p.section, len(order))):
        groups.setdefault(param.group, []).append(param)
    return groups


def to_widget(param: Param, value: Any) -> Any:
    """A parameter value as its widget holds it (numbers: 0 for "none"; tuples as text)."""
    if param.kind == "int":
        return int(value or 0)
    if param.kind == "float":
        return float(value or 0.0)
    if param.kind == "tuple":
        return ", ".join(map(str, value or ()))
    return value


def from_widget(param: Param, raw: Any) -> Any:
    """The widget value as a parameter value (0 means "none" for an optional number); raises ParamError."""
    if param.optional and param.kind in NUMERIC and raw == 0:
        return None
    return param.convert(raw)


def param_widget(param: Param, key: str) -> Any:
    """The widget of one parameter; its value is kept in `st.session_state[key]` (presets write it there)."""
    if key not in st.session_state:
        st.session_state[key] = to_widget(param, param.default)
    label = param.name
    help_text = param.help
    if param.kind == "bool":
        return st.checkbox(label, key=key, help=help_text)
    if param.kind == "int":
        low = None if param.min is None else int(param.min)
        high = None if param.max is None else int(param.max)
        return st.number_input(label, min_value=low, max_value=high, step=1, key=key, help=help_text)
    if param.kind == "float":
        flow = None if param.min is None or (param.optional and param.min > 0) else float(param.min)
        fhigh = None if param.max is None else float(param.max)
        return st.number_input(label, min_value=flow, max_value=fhigh, format="%g", key=key, help=help_text)
    if param.kind == "choice":
        options = [*param.choices, *([None] if param.optional else [])]
        return st.selectbox(label, options, key=key, help=help_text, format_func=lambda v: "none" if v is None else v)
    return st.text_input(label, key=key, help=help_text + " Comma-separated values.")


def widget_key(key: str, name: str, param: Param) -> str:
    return f"{key}_{name}_{param.name}"


def set_widgets(key: str, name: str, values: Mapping[str, Any]) -> None:
    """Write parameter values into the widgets (before they are drawn: presets, reset)."""
    for param in sheet(name):
        st.session_state[widget_key(key, name, param)] = to_widget(param, values.get(param.name, param.default))


def preset_paths(name: str) -> list[Path]:
    return sorted(PRESETS_DIR.glob(f"{name}-*.json"))


def preset_controls(name: str, key: str) -> None:
    """Load a preset (of `presets/`, or saved in this session) or the defaults into the widgets (drawn above them)."""
    paths = {p.stem: p for p in preset_paths(name)}
    own = {f"{name}-{slug}": values for slug, values in session_presets(name).items()}
    c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
    chosen = c1.selectbox("Preset", [*paths, *own], index=None, placeholder="none", key=f"{key}_{name}_preset")
    if c2.button("Load", key=f"{key}_{name}_load", disabled=chosen is None) and chosen is not None:
        try:
            values = own[chosen] if chosen in own else load_preset(paths[chosen])[1]
            set_widgets(key, name, STRATEGIES[name].validate(values))
            st.toast(f"Preset {chosen} loaded")
        except ParamError as exc:
            st.error(f"Invalid preset {chosen}: {exc}")
    if c3.button("Defaults", key=f"{key}_{name}_reset", help="Reset every value to its default"):
        set_widgets(key, name, {})


def param_inputs(name: str, key: str, presets: bool = True) -> dict[str, Any]:
    """
    Form of every parameter of a strategy, generated from its sheet (one tab per group); returns the values
    that differ from the defaults. Errors (bounds, rules linking several values) are shown under the form.
    """
    if presets:
        preset_controls(name, key)
    chosen: dict[str, Any] = {}
    errors: list[str] = []
    groups = grouped(sheet(name))
    for tab, params in zip(st.tabs([group_label(g) for g in groups]), groups.values(), strict=True):
        with tab:
            columns = st.columns(3)
            for i, param in enumerate(params):
                with columns[i % 3]:
                    raw = param_widget(param, widget_key(key, name, param))
                try:
                    value = from_widget(param, raw)
                except ParamError as exc:
                    errors.append(str(exc))
                    continue
                if value != param.default:
                    chosen[param.name] = value
    if STRATEGIES[name].factory.is_group:
        st.caption(
            "Stop and target: on the combined P&L of both legs, which hold the same notional and close together."
        )
    if chosen:
        st.caption("Changed from the defaults: " + ", ".join(f"{k}={format_value(v)}" for k, v in chosen.items()))
    try:
        values = STRATEGIES[name].validate(chosen)
        full = {**defaults(sheet(name)), **values}
        for warning in STRATEGIES[name].factory.warnings(full):
            st.warning(warning)
    except ParamError as exc:
        errors.append(str(exc))
    for error in errors:
        st.error(error)
    if presets:
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        label = c1.text_input("Save as preset", key=f"{key}_{name}_save_as", placeholder="e.g. wide-stops")
        slug = preset_slug(label)
        if c2.button("Save", key=f"{key}_{name}_save", disabled=not slug or bool(errors)):
            if hosted():
                session_presets(name)[slug] = dict(chosen)
                st.toast(f"Preset saved for this session: {name}-{slug}")
            else:
                path = PRESETS_DIR / f"{name}-{slug}.json"
                save_preset(path, name, chosen)
                st.toast(f"Preset saved: {path.name}")
    return chosen


def parse_extra(text: str) -> dict[str, Any]:
    """Free `key=value` lines (same syntax as `--param`)."""
    extra: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"Invalid parameter (expected key=value): {line!r}")
        extra[key.strip()] = parse_value(value.strip())
    return extra


def grid_inputs(name: str) -> dict[str, list[Any]]:
    """Ranges (numbers) or value lists (booleans, choices) of the parameters to optimize."""
    tunable = [p for p in sheet(name) if p.kind in (*NUMERIC, "bool", "choice")]
    by_name = {p.name: p for p in tunable}
    selected = st.multiselect(
        "Parameters to optimize", list(by_name), key=f"wf_grid_{name}", help="The others keep the value set above."
    )
    grid: dict[str, list[Any]] = {}
    for param_name in selected:
        param = by_name[param_name]
        key = f"wf_range_{name}_{param_name}"
        if param.kind in NUMERIC:
            c0, c1, c2, c3 = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
            c0.markdown(f"`{param_name}` · {param.help}")
            default = float(param.default or (param.min or 0) or 1)
            start = c1.number_input("min", value=default, format="%g", key=f"{key}_min")
            stop = c2.number_input("max", value=default * 2, format="%g", key=f"{key}_max")
            step = c3.number_input("step", value=1.0 if param.kind == "int" else default / 2 or 0.1, format="%g",
                                   key=f"{key}_step")  # fmt: skip
            try:
                grid[param_name] = param_range(param, start, stop, step)
            except ParamError as exc:
                st.error(str(exc))
        else:
            options = [True, False] if param.kind == "bool" else [*param.choices, *([None] if param.optional else [])]
            grid[param_name] = st.multiselect(param_name, options, default=options, key=key, help=param.help)
    if grid:
        st.caption(" · ".join(f"{k}: {', '.join(map(format_value, v))}" for k, v in grid.items()))
    return grid


def data_inputs(key: str) -> dict[str, Any]:
    columns = st.columns([2, 2, 1])
    with columns[0]:
        symbols_text = st.text_input(
            "Symbol(s)",
            value="AAPL",
            key=f"{key}_symbols",
            help="Comma-separated; two symbols for pair trading (e.g. AAPL, MSFT). Bundled samples: "
            + ", ".join(loading.sample_symbols())
            + ".",
        )
    with columns[1]:
        source = st.selectbox(
            "Source", list(loading.SOURCES), format_func=SOURCE_LABELS.__getitem__, key=f"{key}_source"
        )
    with columns[2]:
        days = int(st.number_input("Days of history", min_value=1, value=55, key=f"{key}_days"))
    file_path: str | None = None
    if source == "file":
        uploaded = st.file_uploader("Bars file", type=["parquet", "csv"], key=f"{key}_upload")
        typed = (
            ""
            if hosted()
            else st.text_input("…or file path", key=f"{key}_path", placeholder="tests/fixtures/aapl_1m.parquet")
        )
        if uploaded is not None:
            file_path = save_upload(uploaded.getvalue(), uploaded.name)
        elif typed.strip():
            file_path = typed.strip()
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
    return {"symbols": symbols, "source": source, "days": days, "file": file_path}


def save_upload(content: bytes, name: str) -> str:
    """Uploaded files are written once (by content hash) to a temporary folder."""
    digest = hashlib.sha256(content).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / "quantlab_uploads" / f"{digest}{Path(name).suffix}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return str(path)


def execution_inputs(key: str) -> dict[str, Any]:
    """Capital, simulated costs and risk limits."""
    with st.expander("Execution and risk", expanded=False):
        c1, c2, c3 = st.columns(3)
        capital = c1.number_input("Capital ($)", min_value=1.0, value=100_000.0, step=10_000.0, key=f"{key}_capital")
        fee_bps = c2.number_input("Fees (bp per side)", min_value=0.0, value=1.0, key=f"{key}_fee")
        slippage_bps = c3.number_input("Simulated slippage (bp)", min_value=0.0, value=1.0, key=f"{key}_slip")
        st.caption("Stop, target and holding period are parameters of each strategy (Exits tab).")
        c1, c2, c3 = st.columns(3)
        max_order = c1.number_input("Max shares per order (0 = unlimited)", min_value=0, value=0, key=f"{key}_maxo")
        max_position = c2.number_input("Max shares per symbol (0 = unlimited)", min_value=0, value=0, key=f"{key}_maxp")
        max_loss = c3.number_input("Daily-loss kill switch (%, 0 = none)", min_value=0.0, value=0.0, key=f"{key}_dl")
    return {
        "capital": float(capital),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "risk": RiskLimits(
            max_order_quantity=int(max_order) or None,
            max_position_quantity=int(max_position) or None,
            max_daily_loss_pct=max_loss / 100 or None,
        ),
    }


@st.cache_data(show_spinner=False, ttl=3600)
def cached_bars(symbols: tuple[str, ...], source: str, days: int, file: str | None) -> pd.DataFrame:
    return loading.load_bars(list(symbols), source, days, file)


def get_bars(data: dict[str, Any]) -> pd.DataFrame:
    if not data["symbols"]:
        raise ValueError("Enter at least one symbol.")
    with st.spinner(f"Loading {data['source']} data…"):
        return cached_bars(tuple(data["symbols"]), data["source"], data["days"], data["file"])


def metric_tiles(metrics: dict[str, float]) -> None:
    tiles = [
        ("Return", f"{metrics['total_return_pct']:+.3f} %"),
        ("Net P&L", f"{metrics['total_pnl_net']:+,.2f} $"),
        ("Trades", f"{metrics['total_trades']:.0f}"),
        ("Win rate", f"{metrics['win_rate']:.1f} %"),
        ("Profit factor", f"{metrics['profit_factor']:.2f}"),
        ("Sharpe", f"{metrics['sharpe_ratio']:.2f}"),
        ("Max drawdown", f"{metrics['max_drawdown_pct']:.3f} %"),
    ]
    show_tiles(tiles)


def show_tiles(tiles: list[tuple[str, str]], per_row: int = 4) -> None:
    """Metric tiles, at most `per_row` per row so the values are never truncated."""
    for start in range(0, len(tiles), per_row):
        row = tiles[start : start + per_row]
        for column, (label, value) in zip(st.columns(per_row), row, strict=False):
            column.metric(label, value)


def show_error(exc: Exception) -> None:
    if isinstance(exc, (LimitError, ParamError)):
        st.warning(str(exc))  # an expected refusal, not a failure
        return
    logger.exception("UI action failed")
    st.error(f"{type(exc).__name__}: {exc}")


# --- pages -------------------------------------------------------------------------------------------------


def page_data() -> None:
    from quantlab.analysis import dataset

    st.header("Data")
    st.caption(
        "Explore a set of bars before trading it: the price over the period you choose, then the returns, the "
        "intraday profile, the sessions and the quality of the data. Times are New York time."
    )
    data = data_inputs("data")
    symbols: list[str] = data["symbols"]
    if data["source"] == "file" and not data["file"]:
        st.info("Upload a bars file" + ("." if hosted() else ", or type its path."))
        return
    try:
        if len(symbols) > len(charts.SERIES):
            raise LimitError(f"At most {len(charts.SERIES)} symbols at once.")
        frames = dataset.symbol_bars(get_bars(data), symbols)
    except Exception as exc:
        show_error(exc)
        return
    index = pd.DatetimeIndex(frames[symbols[0]].index)
    start, end = period_slider(index, dataset.bar_interval(index))
    frames = {symbol: frame.loc[start:end] for symbol, frame in frames.items()}
    if min(len(frame) for frame in frames.values()) < 2:
        st.warning("This period holds fewer than two bars: widen it.")
        return

    if len(symbols) == 1:
        symbol = symbols[0]
        st.altair_chart(charts.price_chart({symbol: frames[symbol]["close"]}, "Close ($)", x_title=None),
                        width="stretch")  # fmt: skip
        st.altair_chart(charts.volume_chart(frames[symbol]["volume"]), width="stretch")
    else:
        rebased = {s: frame["close"] / frame["close"].iloc[0] * 100 for s, frame in frames.items()}
        st.altair_chart(charts.price_chart(rebased, "Close (start of the period = 100)"), width="stretch")
        symbol = st.radio("Symbol analyzed below", symbols, horizontal=True, key="data_symbol")
    frame = frames[symbol]
    figures, checks = dataset.summary(frame), dataset.quality(frame)
    show_tiles(
        [
            ("Bars", f"{figures['bars']:,}"),
            ("Sessions", f"{figures['sessions']:,}"),
            ("Bar interval", format_duration(figures["interval"])),
            ("Missing bars", f"{checks['missing_bars']:,}"),
            ("Return", f"{figures['return_pct']:+.2f} %"),
            ("Low - high", f"{figures['low']:,.2f} - {figures['high']:,.2f}"),
            ("Volatility (annualized)", "-" if pd.isna(figures["volatility_pct"]) else
             f"{figures['volatility_pct']:.1f} %"),
            ("Volume per session", compact(figures["session_volume"])),
        ]
    )  # fmt: skip

    returns_tab, profile_tab, sessions_tab, quality_tab, bars_tab = st.tabs(
        ["Returns", "Intraday profile", "Sessions", "Quality", "Bars"]
    )
    with returns_tab:
        data_returns(frames, symbol)
    with profile_tab:
        profile = dataset.intraday_profile(frame)
        if len(profile) < 2:
            st.info("The bars all share one time of day: no intraday profile.")
        else:
            hours = [label for label in profile.index if label.endswith(":00")] or None
            c1, c2 = st.columns(2)
            c1.markdown("**Average volume per bar**, by time of day")
            c1.altair_chart(charts.value_bars(profile["volume"], "Volume", "New York time", fmt=",.0f", ticks=hours),
                            width="stretch")  # fmt: skip
            c2.markdown("**Average size of a bar move** (absolute return, bp), by time of day")
            c2.altair_chart(charts.value_bars(profile["abs_return_bp"].dropna(), "Absolute return (bp)",
                                              "New York time", fmt=",.2f", ticks=hours), width="stretch")  # fmt: skip
    with sessions_tab:
        table = dataset.sessions(frame)
        if len(table) > 1:
            labels = pd.Index([f"{day:%b %d}" for day in table.index])
            st.altair_chart(charts.signed_bars(table["return_pct"].set_axis(labels), "Return (%)", "Session"),
                            width="stretch")  # fmt: skip
        st.caption("Return: on the previous close (first session: on its open). Range: high over low.")
        st.dataframe(table.round({"return_pct": 3, "range_pct": 3}), width="stretch", height=300)
        st.download_button("Sessions (CSV)", table.to_csv(), f"{symbol}_sessions.csv", "text/csv")
    with quality_tab:
        show_tiles(
            [
                ("Missing bars", f"{checks['missing_bars']:,}"),
                ("Incomplete sessions", f"{checks['incomplete_sessions']:,}"),
                ("Largest gap in a session", format_duration(checks["largest_gap"])),
                ("Bars without volume", f"{checks['zero_volume_bars']:,}"),
                ("Inconsistent bars", f"{checks['inconsistent_bars']:,}"),
                ("Bars with a missing value", f"{checks['missing_values']:,}"),
            ],
            per_row=3,
        )
        st.caption(
            "Missing bars: holes inside a session, between its first and its last bar. Incomplete sessions: fewer "
            "bars than the typical session (a half day, or the period cutting a session). Inconsistent bars: a high "
            "under the open or the close, a low above them, or a price at or under zero."
        )
    with bars_tab:
        st.dataframe(frame.rename_axis("New York time"), width="stretch", height=360)
        st.download_button("Bars of the period (CSV)", frame.to_csv(), f"{symbol}_bars.csv", "text/csv")


def period_slider(index: pd.DatetimeIndex, interval: pd.Timedelta) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Range slider over the time span of the bars, one bar per step; a new span starts from the full range."""
    step = max(interval, pd.Timedelta(minutes=1))
    first = index[0]
    last = first + -(-(index[-1] - first) // step) * step  # the last bar, on the grid of the steps
    if last <= first:
        return first, first
    start, end = st.slider(
        "Period",
        min_value=first.to_pydatetime(),
        max_value=last.to_pydatetime(),
        value=(first.to_pydatetime(), last.to_pydatetime()),
        step=step.to_pytimedelta(),
        format="YYYY-MM-DD HH:mm",
        key=f"data_period_{first:%Y%m%d%H%M}_{last:%Y%m%d%H%M}",
        help="Drag both ends: the chart and every figure below follow the period.",
    )
    return pd.Timestamp(start), pd.Timestamp(end)


#: share of the bar returns left out of the histogram on each side
TAIL = 0.005


def data_returns(frames: Mapping[str, pd.DataFrame], symbol: str) -> None:
    from quantlab.analysis import dataset

    returns = dataset.intraday_returns(frames[symbol])
    if len(returns) < 2:
        st.info("Not enough bars for a distribution.")
        return
    stats = dataset.return_stats(returns)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Bar returns** (close to close inside a session, overnight gaps left out)")
        low, high = returns.quantile([TAIL, 1 - TAIL])
        body = returns[(returns >= low) & (returns <= high)]
        st.altair_chart(charts.histogram(body * 1e4, "Return (bp)"), width="stretch")
        st.caption(f"The {len(returns) - len(body)} most extreme returns are not drawn, to keep the body readable: "
                   "see Worst and Best.")  # fmt: skip
    with c2:
        st.markdown(f"**{len(returns):,} returns**")
        show_tiles(
            [
                ("Mean", f"{stats['mean_bp']:+.3f} bp"),
                ("Standard deviation", f"{stats['std_bp']:.2f} bp"),
                ("Worst", f"{stats['min_bp']:+.1f} bp"),
                ("Best", f"{stats['max_bp']:+.1f} bp"),
                ("Skewness", f"{stats['skew']:+.2f}"),
                ("Excess kurtosis", f"{stats['excess_kurtosis']:+.2f}"),
                ("Bars up", f"{stats['positive_pct']:.1f} %"),
            ],
            per_row=2,
        )
    if len(frames) > 1:
        st.markdown("**Correlation of the bar returns**")
        st.dataframe(dataset.return_correlation(frames).round(3), width="stretch")


def format_duration(duration: pd.Timedelta) -> str:
    """A bar interval or a gap, in its largest whole unit (2 min, 1 h, 1 d)."""
    seconds = duration.total_seconds()
    for unit, size in (("d", 86400), ("h", 3600), ("min", 60)):
        if seconds >= size and seconds % size == 0:
            return f"{seconds / size:.0f} {unit}"
    return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.0f} s"


def compact(value: float) -> str:
    """A large count on a tile: 45.2M rather than 45,213,870."""
    for unit, size in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(value) >= size:
            return f"{value / size:.1f}{unit}"
    return f"{value:.0f}"


def page_backtest() -> None:
    st.header("Backtest")
    st.caption(
        "Trains the strategies on the start of the data, then trades them bar by bar on the end, with the same "
        "engine as live trading (fills at the next open, fees, slippage, a stop / target per position)."
    )
    names = st.multiselect(
        "Strategies",
        list(STRATEGIES),
        default=["trend"],
        format_func=STRATEGY_LABELS.__getitem__,
        help="Several strategies = one portfolio.",
    )
    data = data_inputs("bt")
    c1, c2, c3 = st.columns(3)
    split_mode = c1.radio("Test block", ["Last days", "Last fraction"], key="bt_split", horizontal=True)
    if split_mode == "Last days":
        test_days, test_fraction = c2.number_input("Test days", min_value=0.1, value=5.0), None
    else:
        test_days, test_fraction = 5.0, c2.slider("Test fraction", 0.05, 0.9, 0.2)
    each = c3.checkbox("Also each strategy alone", value=True, disabled=len(names) < 2)
    bar_by_bar = c3.checkbox("Bar-by-bar live path (slower)")
    params: dict[str, dict[str, Any]] = {}
    for name in names:
        with st.expander(f"Parameters: {STRATEGY_LABELS[name]}"):
            params[name] = param_inputs(name, "bt")
    execution = execution_inputs("bt")

    if st.button("Run the backtest", type="primary", disabled=not names):
        try:
            st.session_state["backtest"] = run_backtests(names, params, data, test_days, test_fraction, each,
                                                         bar_by_bar, execution)  # fmt: skip
        except Exception as exc:
            show_error(exc)
    results = st.session_state.get("backtest")
    if results:
        show_backtest(results)


def run_backtests(
    names: list[str],
    params: dict[str, dict[str, Any]],
    data: dict[str, Any],
    test_days: float,
    test_fraction: float | None,
    each: bool,
    bar_by_bar: bool,
    execution: dict[str, Any],
) -> dict[str, Any]:
    bars = get_bars(data)
    train, test = loading.split(bars, test_days, test_fraction)
    if train.empty or test.empty:
        raise ValueError(f"Empty split: {len(train)} training bars, {len(test)} test bars.")
    if hosted():
        LIMITS.check_backtest(names, len(bars), len(test))
    for name in names:  # invalid parameters fail here, before any computation
        STRATEGIES[name].validate(params[name])
    runs = [[n] for n in names] if each and len(names) > 1 else []
    runs.append(list(names))
    results = {}
    progress = st.progress(0.0, text="Backtest…")
    for i, run_names in enumerate(runs):
        label = " + ".join(STRATEGY_LABELS[n].split(" (")[0] for n in run_names)
        progress.progress(i / len(runs), text=f"Backtest: {label}")
        spec = {
            "names": run_names,
            "params": {n: params[n] for n in run_names},
            "data": data,
            "split": [test_days, test_fraction],
            "bar_by_bar": bar_by_bar,
            "execution": {k: v for k, v in execution.items() if k != "risk"},
            "risk": dataclasses.asdict(execution["risk"]),
        }
        results[label] = cached_backtest(json.dumps(spec, sort_keys=True, default=str))
    progress.empty()
    return {"results": results, "train": (train.index[0], train.index[-1], len(train)),
            "test": (test.index[0], test.index[-1], len(test))}  # fmt: skip


@st.cache_data(max_entries=64, show_spinner=False)
def cached_backtest(spec_json: str) -> Any:
    """One backtest run, cached for every visitor by its full specification (instant for the demo defaults)."""
    from quantlab.backtest.run import backtest

    spec = json.loads(spec_json)
    data = spec["data"]
    bars = cached_bars(tuple(data["symbols"]), data["source"], data["days"], data["file"])
    train, test = loading.split(bars, *spec["split"])
    strategies = [build_strategy(n, data["symbols"], spec["params"][n]) for n in spec["names"]]
    with compute_slots().acquire():
        return backtest(
            strategies,
            train,
            test,
            capital=spec["execution"]["capital"],
            fee_bps=spec["execution"]["fee_bps"],
            slippage_bps=spec["execution"]["slippage_bps"],
            risk_limits=RiskLimits(**spec["risk"]),
            precompute_features=not spec["bar_by_bar"],
        )


def show_backtest(state: dict[str, Any]) -> None:
    results = state["results"]
    t0, t1, n_train = state["train"]
    s0, s1, n_test = state["test"]
    st.caption(f"Training: {t0} → {t1} ({n_train} bars) · Test: {s0} → {s1} ({n_test} bars)")
    labels = list(results)
    main = labels[-1]
    st.subheader(main)
    metric_tiles(results[main].metrics)
    st.altair_chart(charts.equity_chart({label: results[label].equity for label in labels}), width="stretch")
    st.altair_chart(charts.drawdown_chart(results[main].equity), width="stretch")
    if len(labels) > 1:
        st.dataframe(pd.DataFrame({k: v.metrics for k, v in results.items()}).T, width="stretch")

    trades = results[main].trades_frame()
    st.subheader(f"Trades ({len(trades)})")
    if trades.empty:
        st.info("No trade.")
        return
    st.altair_chart(charts.trade_pnl_chart(trades), width="stretch")
    st.dataframe(trades, width="stretch", height=260, hide_index=True)
    c1, c2 = st.columns(2)
    c1.download_button("Trades (CSV)", trades.to_csv(index=False), "trades.csv", "text/csv")
    c2.download_button("Equity curve (CSV)", results[main].equity.to_csv(), "equity.csv", "text/csv")


def page_optimize() -> None:
    st.header("Walk-forward optimization")
    st.caption(
        "Each fold: [train][embargo][validation][embargo][test]. The best combination on the validation block is "
        "refitted, then scored once on the test block."
    )
    name = st.selectbox("Strategy", list(STRATEGIES), format_func=STRATEGY_LABELS.__getitem__, key="wf_strategy")
    data = data_inputs("wf")
    with st.expander("Fixed parameters", expanded=False):
        fixed = param_inputs(name, "wf")
    st.markdown("**Grid**: a min / max / step range for numbers, a list of values otherwise.")
    grid = grid_inputs(name)
    c1, c2, c3, c4 = st.columns(4)
    train_size = int(c1.number_input("Training bars", min_value=10, value=3000, step=100))
    validation_size = int(c2.number_input("Validation bars", min_value=10, value=1000, step=100))
    test_size = int(c3.number_input("Test bars", min_value=10, value=1000, step=100))
    embargo = int(c4.number_input("Embargo (bars)", min_value=0, value=30))
    c1, c2, c3, c4 = st.columns(4)
    step = int(c1.number_input("Step between folds (0 = test size)", min_value=0, value=0))
    objective = c2.selectbox("Objective", OBJECTIVES)
    min_trades = int(c3.number_input("Minimum trades", min_value=0, value=5))
    anchored = c4.checkbox("Anchored training (growing window)")

    if st.button("Run the optimization", type="primary", disabled=not grid):
        try:
            from quantlab.backtest.walk_forward import WalkForwardOptimizer, make_folds

            bars = get_bars(data)
            optimizer = WalkForwardOptimizer(
                strategy_factory=lambda **p: build_strategy(name, data["symbols"], {**fixed, **p}),
                param_grid=grid,
                train_size=train_size,
                validation_size=validation_size,
                test_size=test_size,
                step=step or None,
                embargo=embargo,
                anchored=anchored,
                objective=objective,
                min_trades=min_trades,
            )
            n = len(optimizer.candidates)
            if hosted():
                folds = make_folds(len(bars), train_size, validation_size, test_size, step or None, embargo, anchored)
                LIMITS.check_optimize(name, n, len(folds), len(bars))
            with st.spinner(f"{n} combinations per fold…"), compute_slots().acquire():
                st.session_state["optimize"] = optimizer.run(bars)
        except Exception as exc:
            show_error(exc)
    if not grid:
        st.info("Choose at least one parameter for the grid.")
    result = st.session_state.get("optimize")
    if result is not None:
        summary = result.summary()
        st.metric(f"Mean out-of-sample score ({result.objective})", f"{result.mean_test_score:,.4f}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Test score per fold**")
            st.altair_chart(charts.value_bars(summary["test_score"], result.objective, "Fold"), width="stretch")
        with c2:
            st.markdown("**Stability of the chosen parameters** (number of folds)")
            stability = result.param_stability()
            st.altair_chart(charts.value_bars(stability, "Folds", "Parameters", fmt="d"), width="stretch")
        st.dataframe(summary, width="stretch")


# --- simulated live trading ------------------------------------------------------------------------------

#: hosted demo: sessions in a temporary folder, a few at once, short and bounded
HOSTED_SESSIONS = {"max_running": 3, "per_owner": 1, "max_bars": 975, "max_seconds": 900.0, "keep_seconds": 3600.0}
SPEEDS = (1.0, 2.0, 5.0, 10.0, 20.0)


@st.cache_resource
def simulated_sessions(is_hosted: bool) -> Any:
    from quantlab.live.session import SimulatedSessions

    if is_hosted:
        return SimulatedSessions(
            Path(tempfile.gettempdir()) / "quantlab_sessions",
            max_running=int(HOSTED_SESSIONS["max_running"]),
            per_owner=int(HOSTED_SESSIONS["per_owner"]),
            prepare_guard=compute_slots().acquire,
        )
    return SimulatedSessions(SESSIONS_DIR)


def visitor_id() -> str:
    """The visitor's id in the hosted demo (kept in the URL so a reload finds its sessions); "local" otherwise."""
    if not hosted():
        return "local"
    if "visitor" not in st.session_state:
        candidate = str(st.query_params.get("visitor", ""))
        st.session_state["visitor"] = candidate if re.fullmatch(r"[0-9a-f]{12}", candidate) else uuid.uuid4().hex[:12]
    visitor: str = st.session_state["visitor"]
    st.query_params["visitor"] = visitor
    return visitor


def page_live() -> None:
    from quantlab.live.session import SessionLimitError, SimulatedConfig

    st.header("Live trading (simulated)")
    st.caption(
        "A strategy trades a real-time stream of bars, with the same engine as a live session: orders, fills, "
        "protective stops and targets, positions, journal and persisted state. The bars replay the end of the "
        "bundled data, or come from a synthetic market; the broker is simulated. Once it ends, replay the session "
        "in the backtester to check that both agree."
    )
    c1, c2 = st.columns([1, 1])
    name = c1.selectbox("Strategy", list(STRATEGIES), format_func=STRATEGY_LABELS.__getitem__, key="live_strategy")
    default_symbols = "SPY, QQQ" if STRATEGIES[name].n_symbols == 2 else "AAPL"
    symbols_text = c2.text_input("Symbol(s)", value=default_symbols, key=f"live_symbols_{name}",
                                 help="Bundled samples: " + ", ".join(loading.sample_symbols()) + ".")  # fmt: skip
    with st.expander(f"Parameters: {STRATEGY_LABELS[name]}"):
        params = param_inputs(name, "live")
    c1, c2, c3, c4 = st.columns(4)
    source = c1.radio("Market", ["replay", "synthetic"], key="live_source", horizontal=True,
                      format_func={"replay": "Replay", "synthetic": "Synthetic"}.__getitem__,
                      help="Replay: the end of the bundled data. Synthetic: a random market after it.")  # fmt: skip
    max_bars = HOSTED_SESSIONS["max_bars"] if hosted() else 5000
    live_bars = int(c2.number_input("Bars to stream", min_value=50, max_value=int(max_bars), value=390, step=65,
                                    key="live_bars", help="195 two-minute bars = one session."))  # fmt: skip
    speed = c3.select_slider("Speed (bars per second)", SPEEDS, value=2.0, key="live_speed")
    seed = int(
        c4.number_input("Seed (synthetic)", min_value=0, value=7, key="live_seed", disabled=source != "synthetic")
    )
    execution = execution_inputs("live")
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
    registry = simulated_sessions(hosted())
    owner = visitor_id()

    if st.button("Start the session", type="primary"):
        try:
            STRATEGIES[name].validate(params)
            if len(symbols) != STRATEGIES[name].n_symbols:
                raise ValueError(f"{STRATEGY_LABELS[name]} needs {STRATEGIES[name].n_symbols} symbol(s).")
            if hosted():
                registry.cleanup(HOSTED_SESSIONS["keep_seconds"])
            config = SimulatedConfig(
                strategy=name,
                symbols=tuple(symbols),
                params=params,
                source="synthetic" if source == "synthetic" else "replay",
                live_bars=live_bars,
                bars_per_second=float(speed),
                max_seconds=HOSTED_SESSIONS["max_seconds"] if hosted() else None,
                seed=seed,
                capital=execution["capital"],
                risk=execution["risk"],
                fee_bps=execution["fee_bps"],
                slippage_bps=execution["slippage_bps"],
            )
            session_name = f"{name}_{'_'.join(symbols)}_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
            registry.start(session_name, config, owner=owner)
            st.toast(f"Session {session_name} started")
        except (SessionLimitError, ValueError) as exc:
            show_error(LimitError(str(exc)))
    entries = sorted(registry.entries(owner), key=lambda e: e.started, reverse=True)
    if not entries:
        st.info("No session yet: choose a strategy and start one.")
        return
    if any(e.alive for e in entries):
        live_monitor_running(owner)
    else:
        live_monitor(owner)


def live_monitor(owner: str) -> None:
    entries = sorted(simulated_sessions(hosted()).entries(owner), key=lambda e: e.started, reverse=True)
    for i, entry in enumerate(entries):
        with st.container(border=True):
            session_panel(entry, expanded=i == 0)


@st.fragment(run_every=2)
def live_monitor_running(owner: str) -> None:
    live_monitor(owner)
    if not any(e.alive for e in simulated_sessions(hosted()).entries(owner)):
        st.rerun()  # the last session ended: stop refreshing


def session_panel(entry: Any, expanded: bool) -> None:
    from quantlab.ui.live_view import read_session

    delivered, total = entry.session.progress
    config = entry.session.config
    c1, c2 = st.columns([5, 1])
    c1.markdown(
        f"**{entry.name}** · {STRATEGY_LABELS[config.strategy]} on {', '.join(config.symbols)} · "
        f"{config.source} · {config.bars_per_second:g} bars/s · status **{entry.status}**"
    )
    if entry.alive and c2.button("Stop", key=f"stop_{entry.name}"):
        entry.stop()
        st.toast(f"Stopping {entry.name}")
    if entry.error:
        st.error(entry.error)
        return
    st.progress(min(delivered / total, 1.0) if total else 0.0, text=f"{delivered} / {total} bars")
    if not expanded:
        return
    view = read_session(entry.path)
    if not view.bars:
        st.caption("Training the strategy on the history…" if entry.status == "training" else "Waiting for bars…")
        return
    unrealized = float(view.lots["unrealized P&L"].fillna(0).sum()) if not view.lots.empty else 0.0
    show_tiles(
        [
            ("Equity", f"{view.equity:,.2f} $" if view.equity is not None else "-"),
            ("Realized P&L", f"{view.realized_pnl:+,.2f} $"),
            ("Unrealized P&L", f"{unrealized:+,.2f} $"),
            ("Trades (won)", f"{view.trades} ({view.wins})"),
            ("Positions", ", ".join(f"{k} {v:+d}" for k, v in view.positions.items()) or "flat"),
            ("Last bar", f"{charts.exchange_time(pd.DatetimeIndex([view.last_bar]))[0]:%m-%d %H:%M}"),
            ("Latency p50", f"{view.latency_p50_ms:.1f} ms" if view.latency_p50_ms is not None else "-"),
            ("Kill switch", "ON" if view.halted else "off"),
        ]
    )
    if len(view.equity_curve) > 1:
        st.altair_chart(charts.equity_chart({"Equity": view.equity_curve}, height=220), width="stretch")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Open positions**")
        if view.lots.empty:
            st.caption("None.")
        else:
            st.dataframe(view.lots, width="stretch", hide_index=True)
    with c2:
        st.markdown("**Activity** (newest first)")
        activity = view.activity.assign(bar=charts.exchange_time(pd.DatetimeIndex(view.activity["bar"])))
        st.dataframe(activity, width="stretch", hide_index=True, height=300)
    if not view.closed_trades.empty and len(view.closed_trades) > 1:
        st.altair_chart(charts.trade_pnl_chart(view.closed_trades), width="stretch")
    if not entry.alive:
        session_replay(entry)


def session_replay(entry: Any) -> None:
    key = f"replay_{entry.owner}_{entry.name}"
    if st.button("Replay in the backtester", key=f"btn_{key}"):
        try:
            from quantlab.analysis.replay import replay_session

            with st.spinner("Replay…"), compute_slots().acquire():
                st.session_state[key] = replay_session(entry.path)
        except Exception as exc:
            show_error(exc)
    report = st.session_state.get(key)
    if report is None:
        return
    s = report.summary
    st.markdown(
        "**Replay**: the session's strategies, as they were before the first live bar, rerun on the journaled "
        "bars with the simulated broker."
    )
    show_tiles(
        [
            ("Signal agreement", f"{s['signal_agreement_pct']:.1f} %"),
            ("Signals live / replay", f"{s['signals_live']:.0f} / {s['signals_replay']:.0f}"),
            ("Entry slippage", f"{s['entry_slippage_bps_mean']:+.2f} bp"),
            ("Net P&L live / replay", f"{s['pnl_net_live']:+,.2f} / {s['pnl_net_replay']:+,.2f} $"),
        ]
    )


def main() -> None:
    st.set_page_config(page_title="Quant Lab", page_icon=":chart_with_upwards_trend:", layout="wide")
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    navigation = st.navigation(
        [
            st.Page(page_data, title="Data", icon=":material/database:", default=True),
            st.Page(page_backtest, title="Backtest", icon=":material/query_stats:", url_path="backtest"),
            st.Page(page_optimize, title="Optimization", icon=":material/tune:", url_path="optimization"),
            st.Page(page_live, title="Live trading", icon=":material/sensors:", url_path="live"),
        ]
    )
    if hosted():
        st.sidebar.caption(
            "Online demo: bundled market data and capped computations. "
            f"[Source code and full version]({REPOSITORY_URL})."
        )
    navigation.run()


if __name__ == "__main__":
    main()
