"""
quantlab command line: one entry point for strategy research.

    quantlab backtest --strategy trend --symbol AAPL --test-days 5          (offline, on the bundled sample)
    quantlab backtest --strategy trend --symbol AAPL --source yfinance --days 55 --test-days 5
    quantlab backtest --strategy cascade --strategy trend --symbol SPY --days 55 --each
    quantlab backtest --strategy trend --symbol AAPL --source file --file tests/fixtures/aapl_1m.parquet \\
                      --test-fraction 0.3
    quantlab backtest --strategy pair --symbol AAPL --symbol MSFT --param zscore_threshold=2.0
    quantlab backtest --strategy trend --strategy mean_reversion --symbol AAPL --param trend.stop_atr=1.5 \\
                      --param mean_reversion.target_middle_band=true --param holding_bars=10
    quantlab backtest --strategy trend --symbol AAPL --params-file presets/trend-atr-exits.json
    quantlab params --strategy cascade                (every parameter: group, default, bounds, help)
    quantlab optimize --strategy trend --symbol AAPL --range sma_fast=5:20:5 --grid sma_slow=30,50,100 \\
                      --train-size 3000 --validation-size 1000 --test-size 1000 --embargo 30
    quantlab live --strategy trend --symbol AAPL --source synthetic --speed 5      (simulated live trading)
    quantlab replay --session NAME
    quantlab plot --symbol SPY --days 3
    quantlab ui                                     (graphical interface in the browser)

Data sources: sample (the default: bundled 2-minute bars of AAPL, MSFT, NVDA, SPY and QQQ, offline), yfinance
(2-minute bars, ~60 days, free, no API key) and file (--file bars.parquet|csv).

Parameters: `quantlab params --strategy NAME` lists them. `--param key=value` applies to every selected strategy
that declares the key, `--param name.key=value` to one; `--params-file` loads a preset (JSON, see presets/),
which --param overrides. Unknown keys and out-of-bounds values are refused. Exits are strategy parameters too:
stop_loss_pct (2), take_profit_pct (5), stop_atr / take_profit_atr (ATR multiples, replace the percentages),
atr_window (14), holding_bars (5); 0 or none disables one. An entry without a stop is refused.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data import loading
from quantlab.data.loading import SOURCES, split
from quantlab.execution.risk import RiskLimits
from quantlab.strategies.base import Strategy
from quantlab.strategies.params import GROUPS, ParamError, load_preset, param_range
from quantlab.strategies.registry import STRATEGIES, build_strategy

logger = logging.getLogger("quantlab")

__all__ = ["SOURCES", "main", "parse_grid", "parse_params", "parse_value", "split"]


# --- argument helpers --------------------------------------------------------------------------------------


def parse_value(text: str) -> Any:
    """'10' -> 10, '2.5' -> 2.5, 'true' -> True, 'none' -> None, anything else stays a string."""
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def format_value(value: Any) -> str:
    """A parameter value as typed on the command line (`none`, `true`, `2,1,0`)."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, tuple):
        return ",".join(map(str, value))
    return str(value)


def declares(strategy: str, key: str) -> bool:
    return strategy in STRATEGIES and any(p.name == key for p in STRATEGIES[strategy].params)


def parse_params(
    items: Sequence[str], strategies: Sequence[str], files: Sequence[str | Path] = ()
) -> dict[str, dict[str, Any]]:
    """
    Parameters per strategy: the presets `files` first, then `--param` items. `key=value` applies to every
    strategy declaring `key` (an error if none does); `name.key=value` to one.
    """
    params: dict[str, dict[str, Any]] = {name: {} for name in strategies}
    for path in files:
        try:
            target, values = load_preset(path)
        except ParamError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
        if target not in params:
            raise argparse.ArgumentTypeError(f"--params-file {path}: {target!r} is not a selected strategy")
        params[target].update(values)
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"--param expects key=value, got {item!r}")
        target, dot, name = key.rpartition(".")
        if dot and target not in params:
            raise argparse.ArgumentTypeError(f"--param {item!r}: {target!r} is not a selected strategy")
        targets = [target] if dot else [s for s in params if declares(s, name)]
        if not dot and not targets:
            raise argparse.ArgumentTypeError(
                f"--param {item!r}: no selected strategy has a parameter {name!r} (see quantlab params)"
            )
        for strategy in targets:
            params[strategy][name] = parse_value(value)
    return params


def params_table(name: str) -> pd.DataFrame:
    """The parameter sheet of a strategy, grouped (for `quantlab params`)."""
    rows = [p.describe() for p in STRATEGIES[name].params]
    order = {group: i for i, group in enumerate(GROUPS)}
    table = pd.DataFrame(rows)
    table["default"] = [format_value(row["default"]) for row in rows]  # no float upcast of the ints
    table["_order"] = [order.get(g.split(":", 1)[0], len(order)) for g in table["group"]]
    return table.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)


def parse_grid(items: Sequence[str]) -> dict[str, list[Any]]:
    grid: dict[str, list[Any]] = {}
    for item in items:
        key, sep, values = item.partition("=")
        if not sep or not values:
            raise argparse.ArgumentTypeError(f"--grid expects key=v1,v2,..., got {item!r}")
        grid[key] = [parse_value(v) for v in values.split(",")]
    return grid


def parse_ranges(items: Sequence[str], strategy: str) -> dict[str, list[Any]]:
    """`--range key=min:max:step` -> the values of a numeric parameter, checked by the strategy's sheet."""
    sheet = {p.name: p for p in STRATEGIES[strategy].params}
    grid: dict[str, list[Any]] = {}
    for item in items:
        key, sep, spec = item.partition("=")
        parts = spec.split(":")
        if not sep or len(parts) != 3:
            raise argparse.ArgumentTypeError(f"--range expects key=min:max:step, got {item!r}")
        if key not in sheet:
            raise argparse.ArgumentTypeError(f"--range {item!r}: {strategy} has no parameter {key!r}")
        try:
            start, stop, step = (float(v) for v in parts)
            grid[key] = param_range(sheet[key], start, stop, step)
        except (ValueError, ParamError) as exc:
            raise argparse.ArgumentTypeError(f"--range {item!r}: {exc}") from exc
    return grid


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    try:
        return loading.load_bars(args.symbol, args.source, args.days, args.file)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def strategy_params(args: argparse.Namespace, strategies: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Parameters per strategy, validated by their sheets (a clear exit on an invalid value)."""
    params = parse_params(args.param, strategies, args.params_file)
    for name, values in params.items():
        try:
            STRATEGIES[name].validate(values)
        except ParamError as exc:
            raise SystemExit(f"{name}: {exc}") from exc
    return params


def build_strategies(args: argparse.Namespace) -> list[Strategy]:
    params = strategy_params(args, args.strategy)
    return [build_strategy(name, args.symbol, params[name]) for name in args.strategy]


def risk_limits(args: argparse.Namespace) -> RiskLimits:
    def pct(value: float | None) -> float | None:
        return None if value is None or value <= 0 else value / 100

    return RiskLimits(
        max_order_quantity=args.max_order,
        max_position_quantity=args.max_position,
        max_daily_loss_pct=pct(args.max_daily_loss),
    )


# --- commands ----------------------------------------------------------------------------------------------


def cmd_backtest(args: argparse.Namespace) -> int:
    from quantlab.backtest.run import backtest

    data = load_bars(args)
    train, test = split(data, args.test_days, args.test_fraction)
    logger.info("Training %s to %s (%d bars)", train.index[0], train.index[-1], len(train))
    logger.info("Test     %s to %s (%d bars)", test.index[0], test.index[-1], len(test))

    runs: list[list[str]] = [[name] for name in args.strategy] if args.each and len(args.strategy) > 1 else []
    runs.append(list(args.strategy))
    rows = {}
    for names in runs:
        sub = argparse.Namespace(**{**vars(args), "strategy": names})
        result = backtest(
            build_strategies(sub),
            train,
            test,
            capital=args.capital,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            risk_limits=risk_limits(args),
            precompute_features=not args.bar_by_bar,
        )
        label = "+".join(names)
        rows[label] = result.metrics
        if len(names) == len(args.strategy):
            if args.trades:
                result.trades_frame().to_csv(args.trades, index=False)
                logger.info("Trades written to %s", args.trades)
            if args.equity:
                result.equity.to_csv(args.equity)
                logger.info("Equity curve written to %s", args.equity)
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:,.4f}"))
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    from quantlab.backtest.walk_forward import WalkForwardOptimizer

    if len(args.strategy) != 1:
        raise SystemExit("optimize takes exactly one --strategy")
    name = args.strategy[0]
    fixed = strategy_params(args, [name])[name]
    grid = {**parse_grid(args.grid), **parse_ranges(args.range, name)}
    if not grid:
        raise SystemExit("optimize needs at least one --grid or --range")
    data = load_bars(args)
    optimizer = WalkForwardOptimizer(
        strategy_factory=lambda **p: build_strategy(name, args.symbol, {**fixed, **p}),
        param_grid=grid,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        step=args.step,
        embargo=args.embargo,
        anchored=args.anchored,
        objective=args.objective,
        min_trades=args.min_trades,
    )
    result = optimizer.run(data)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(result.summary().to_string())
    print("\nStability of the chosen parameters:")
    print(result.param_stability().to_string())
    print(f"\nMean out-of-sample {result.objective}: {result.mean_test_score:.4f}")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from datetime import datetime

    from quantlab.live.session import SimulatedConfig, SimulatedSession

    if len(args.strategy) != 1:
        raise SystemExit("live takes exactly one --strategy")
    name = args.strategy[0]
    config = SimulatedConfig(
        strategy=name,
        symbols=tuple(args.symbol),
        params=strategy_params(args, [name])[name],
        source=args.source,
        live_bars=args.bars,
        bars_per_second=args.speed,
        seed=args.seed,
        capital=args.capital,
        risk=risk_limits(args),
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        data_source=args.data,
        days=args.days,
    )
    session_name = args.session or f"{name}_{'_'.join(args.symbol)}_{datetime.now():%Y%m%d_%H%M%S}"
    session = SimulatedSession(args.sessions_dir / session_name, config)
    session.prepare()
    assert session.runner is not None
    session.runner.router.fill_listeners.append(
        lambda fill: logger.info(
            "%s fill %s %d %s @ %.2f", fill.timestamp, fill.order.side, fill.quantity, fill.order.symbol, fill.price
        )
    )
    logger.info("Streaming %d bars at %g bars/s (Ctrl+C to stop)", config.live_bars, config.bars_per_second)
    try:
        runner = session.run()
    except KeyboardInterrupt:
        session.stop()
        raise SystemExit(f"Interrupted; session saved in {session.files.root}") from None
    trades = runner.router.trades
    print(f"Session {session.files.root} ({session.status}): {len(trades)} trades, "
          f"net P&L {sum(t.pnl_net for t in trades):+,.2f}, equity {runner.portfolio.equity:,.2f}")  # fmt: skip
    print(f"Compare it with the backtester: quantlab replay --session {session_name}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from quantlab.analysis.replay import replay_session

    report = replay_session(args.sessions_dir / args.session, slippage_bps=args.slippage_bps)
    print(pd.Series(report.summary).to_string(float_format=lambda v: f"{v:,.4f}"))
    mismatched = report.signals[report.signals["match"] != "both"]
    if not mismatched.empty:
        print("\nSignals not matched:\n" + mismatched.to_string())
    if not report.entry_fills.empty:
        print("\nEntry fills:\n" + report.entry_fills.to_string())
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from quantlab.analysis.plot import plot_levels, session_bars

    try:
        session, day = session_bars(load_bars(args), args.date)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    plot_levels(session, f"{args.symbol[0]} {day} (New York time)", output=args.output)
    return 0


def cmd_params(args: argparse.Namespace) -> int:
    with pd.option_context("display.max_colwidth", 90, "display.width", 250):
        for name in args.strategy:
            print(f"{name} ({STRATEGIES[name].factory.__name__})")
            print(params_table(name).to_string(index=False) + "\n")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    import subprocess

    app = Path(__file__).parent / "ui" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port)]
    return subprocess.call(command)


# --- parser ------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantlab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logs")
    commands = parser.add_subparsers(dest="command", required=True)

    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--symbol", action="append", required=True, help="repeat for multi-symbol strategies")
    data.add_argument("--source", choices=SOURCES, default="sample", help="sample: bundled data, offline (default)")
    data.add_argument("--file", help="bars file for --source file (parquet or csv)")
    data.add_argument("--days", type=int, default=55, help="history window in calendar days")

    strategy = argparse.ArgumentParser(add_help=False)
    strategy.add_argument("--strategy", action="append", required=True, choices=sorted(STRATEGIES))
    strategy.add_argument("--param", action="append", default=[], help="key=value or strategy.key=value")
    strategy.add_argument(
        "--params-file", action="append", default=[], help="preset JSON {strategy, params}; --param overrides it"
    )

    def execution_parser(live: bool) -> argparse.ArgumentParser:
        # One parser per command: argparse parents share their actions, so defaults set on one command
        # (the conservative live limits) would leak into the others.
        execution = argparse.ArgumentParser(add_help=False)
        execution.add_argument("--capital", type=float, default=100_000.0)
        execution.add_argument("--fee-bps", type=float, default=1.0)
        execution.add_argument("--slippage-bps", type=float, default=1.0, help="simulated slippage")
        execution.add_argument("--max-order", type=int, default=10 if live else None, help="max shares per entry order")
        execution.add_argument(
            "--max-position", type=int, default=50 if live else None, help="max absolute shares per symbol"
        )
        execution.add_argument(
            "--max-daily-loss",
            type=float,
            default=1.0 if live else None,
            help="kill switch, percent of the day's equity",
        )
        return execution

    sessions = argparse.ArgumentParser(add_help=False)
    sessions.add_argument("--sessions-dir", type=Path, default=Path("sessions"))

    p = commands.add_parser(
        "backtest", parents=[data, strategy, execution_parser(live=False)], help="backtest strategies"
    )
    p.add_argument("--test-days", type=float, default=5, help="last N calendar days are the test block")
    p.add_argument("--test-fraction", type=float, help="or: the last fraction of the bars")
    p.add_argument("--each", action="store_true", help="with several strategies: also run each alone")
    p.add_argument("--bar-by-bar", action="store_true", help="live feature path instead of the batch mode")
    p.add_argument("--trades", help="write the trades to this CSV")
    p.add_argument("--equity", help="write the equity curve to this CSV")
    p.set_defaults(func=cmd_backtest)

    p = commands.add_parser("optimize", parents=[data, strategy], help="walk-forward hyperparameter search")
    p.add_argument("--grid", action="append", default=[], help="key=v1,v2,...")
    p.add_argument("--range", action="append", default=[], help="key=min:max:step (numeric parameters)")
    p.add_argument("--train-size", type=int, required=True)
    p.add_argument("--validation-size", type=int, required=True)
    p.add_argument("--test-size", type=int, required=True)
    p.add_argument("--step", type=int)
    p.add_argument("--embargo", type=int, default=30)
    p.add_argument("--anchored", action="store_true")
    p.add_argument("--objective", default="total_return_pct")
    p.add_argument("--min-trades", type=int, default=5)
    p.set_defaults(func=cmd_optimize)

    p = commands.add_parser(
        "live",
        parents=[strategy, execution_parser(live=False), sessions],
        help="simulated live trading: a strategy trades a real-time stream of replayed or synthetic bars",
    )
    p.add_argument("--symbol", action="append", required=True, help="repeat for multi-symbol strategies")
    p.add_argument(
        "--source",
        choices=("replay", "synthetic"),
        default="replay",
        help="replay: the last bars of the data; synthetic: a random market continuing the data",
    )
    p.add_argument("--bars", type=int, default=975, help="bars to stream (975 = five sessions of 2-minute bars)")
    p.add_argument("--speed", type=float, default=2.0, help="bars per second (0: as fast as possible)")
    p.add_argument("--seed", type=int, help="seed of the synthetic market")
    p.add_argument("--data", choices=SOURCES, default="sample", help="where the bars come from")
    p.add_argument("--days", type=int, default=60, help="history window in calendar days")
    p.add_argument("--session", help="name (default: <strategy>_<symbols>_<time>)")
    p.set_defaults(func=cmd_live)

    p = commands.add_parser("replay", parents=[sessions], help="replay a session in the backtester")
    p.add_argument("--session", required=True)
    p.add_argument("--slippage-bps", type=float, help="simulated slippage (default: the session's)")
    p.set_defaults(func=cmd_replay)

    p = commands.add_parser("plot", parents=[data], help="chart one session with VWAP and support/resistance")
    p.add_argument("--date", help="session date (default: the last one)")
    p.add_argument("--output", help="save to this image file instead of showing")
    p.set_defaults(func=cmd_plot)

    p = commands.add_parser("params", help="list the parameters of strategies")
    p.add_argument("--strategy", action="append", required=True, choices=sorted(STRATEGIES))
    p.set_defaults(func=cmd_params)

    p = commands.add_parser("ui", help="graphical interface in the browser (Streamlit)")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=cmd_ui)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("ibapi").setLevel(logging.WARNING)
    try:
        code: int = args.func(args)
    except ConnectionError as exc:
        logger.error("%s", exc)
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
