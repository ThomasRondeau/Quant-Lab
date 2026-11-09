"""
quantlab graphical interface (Streamlit): everything the CLI does, plus charts.

    python -m quantlab ui            (or: python -m streamlit run quantlab/ui/app.py)

Pages: Backtest, Optimization (walk-forward), Chart (VWAP + support / resistance).
The pages call the same library functions as the CLI. With QUANTLAB_HOSTED=1 (the public demo), the inputs and the
computations are restricted (`ui/hosting.py`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import sys
import tempfile
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


def data_inputs(key: str, multi_symbol: bool = True, default_days: int = 55) -> dict[str, Any]:
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
        days = int(st.number_input("Days of history", min_value=1, value=default_days, key=f"{key}_days"))
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
    if not multi_symbol:
        symbols = symbols[:1]
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


def page_chart() -> None:
    from quantlab.analysis.plot import levels_figure, session_bars

    st.header("Session chart")
    st.caption(
        "Candles, volume, the VWAP and its bands, and the best support / resistance lines. These lines use "
        "future bars: they are for visual analysis only, never a signal."
    )
    data = data_inputs("chart", multi_symbol=False, default_days=3)
    c1, c2 = st.columns(2)
    day = c1.date_input("Session (empty = the last one)", value=None)
    n_lines = int(c2.number_input("Support / resistance lines", min_value=0, max_value=10, value=3))
    if st.button("Draw", type="primary"):
        try:
            bars, chosen = session_bars(get_bars(data), day.isoformat() if day else None)
            with st.spinner("Computing the levels…"):
                figure = levels_figure(bars, f"{data['symbols'][0]} {chosen} (New York time)", n_lines)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=120)
            st.session_state["chart"] = (figure, buffer.getvalue(), f"{data['symbols'][0]}_{chosen}.png")
        except Exception as exc:
            show_error(exc)
    if "chart" in st.session_state:
        figure, png, filename = st.session_state["chart"]
        st.pyplot(figure)
        st.download_button("Download (PNG)", png, filename, "image/png")


def main() -> None:
    st.set_page_config(page_title="Quant Lab", page_icon=":chart_with_upwards_trend:", layout="wide")
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    navigation = st.navigation(
        [
            st.Page(page_backtest, title="Backtest", icon=":material/query_stats:", default=True),
            st.Page(page_optimize, title="Optimization", icon=":material/tune:", url_path="optimization"),
            st.Page(page_chart, title="Chart", icon=":material/candlestick_chart:", url_path="chart"),
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
