"""Command line: argument parsing and the offline commands on the fixtures (no network)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from quantlab.cli import build_parser, main, parse_grid, parse_params, parse_value, split
from quantlab.strategies.pair_trading import PairTradingStrategy
from quantlab.strategies.registry import STRATEGIES, build_strategy
from quantlab.strategies.trend_following import TrendFollowingStrategy

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_value() -> None:
    assert [parse_value(v) for v in ("10", "2.5", "true", "False", "AAPL")] == [10, 2.5, True, False, "AAPL"]
    assert parse_value("none") is None


def test_parse_params_global_and_targeted() -> None:
    params = parse_params(["holding_bars=3", "trend.sma_fast=8"], ["trend", "mean_reversion"])
    assert params == {"trend": {"holding_bars": 3, "sma_fast": 8}, "mean_reversion": {"holding_bars": 3}}
    # a global key goes only to the strategies that declare it, and must be declared by one
    params = parse_params(["bb_window=30"], ["trend", "mean_reversion"])
    assert params == {"trend": {}, "mean_reversion": {"bb_window": 30}}
    with pytest.raises(argparse.ArgumentTypeError, match="no selected strategy"):
        parse_params(["holding=3"], ["trend"])
    with pytest.raises(argparse.ArgumentTypeError):
        parse_params(["cascade.x=1"], ["trend"])
    with pytest.raises(argparse.ArgumentTypeError):
        parse_params(["no_value"], ["trend"])


def test_parse_grid() -> None:
    assert parse_grid(["sma_fast=5,10", "sma_slow=30"]) == {"sma_fast": [5, 10], "sma_slow": [30]}


def test_parse_ranges_are_checked_by_the_sheet() -> None:
    from quantlab.cli import parse_ranges

    assert parse_ranges(["sma_fast=5:20:5", "stop_atr=1:2:0.5"], "trend") == {
        "sma_fast": [5, 10, 15, 20],
        "stop_atr": [1.0, 1.5, 2.0],
    }
    for bad in ("sma_fast=5:20", "nope=1:2:1", "sma_fast=0:4:1", "size_mode=1:2:1", "sma_fast=1:5000:1"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_ranges([bad], "trend")


def test_split_by_days_and_fraction() -> None:
    index = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    data = pd.DataFrame({"close": range(10)}, index=index)
    train, test = split(data, test_days=3, test_fraction=None)
    assert (len(train), len(test)) == (7, 3)
    train, test = split(data, test_days=3, test_fraction=0.2)
    assert (len(train), len(test)) == (8, 2)


def test_registry_builds_every_strategy() -> None:
    assert isinstance(build_strategy("trend", ["SPY"], {"sma_fast": 3}), TrendFollowingStrategy)
    pair = build_strategy("pair", ["AAPL", "MSFT"])
    assert isinstance(pair, PairTradingStrategy) and (pair.symbol_a, pair.symbol_b) == ("AAPL", "MSFT")
    with pytest.raises(ValueError):
        build_strategy("pair", ["AAPL"])
    for name, spec in STRATEGIES.items():
        assert build_strategy(name, ["A", "B"][: spec.n_symbols]).strategy_id


def test_backtest_has_no_risk_limits_by_default() -> None:
    args = build_parser().parse_args(["backtest", "--strategy", "trend", "--symbol", "AAPL"])
    assert (args.max_order, args.max_position, args.max_daily_loss) == (None, None, None)


def test_exits_are_strategy_parameters() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["backtest", "--strategy", "trend", "--symbol", "AAPL", "--stop-loss", "1"])
    params = parse_params(["trend.stop_atr=1.5", "holding_bars=none"], ["trend", "mean_reversion"])
    trend = build_strategy("trend", ["AAPL"], params["trend"])
    reversion = build_strategy("mean_reversion", ["AAPL"], params["mean_reversion"])
    assert (trend.stop_atr, trend.holding_bars, reversion.stop_atr, reversion.holding_bars) == (1.5, None, None, None)


def test_backtest_command_writes_trades_and_equity(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    trades, equity = tmp_path / "trades.csv", tmp_path / "equity.csv"
    code = main(
        ["backtest", "--strategy", "trend", "--strategy", "mean_reversion", "--each", "--symbol", "AAPL",
         "--source", "file", "--file", str(FIXTURES / "aapl_1m.parquet"), "--test-fraction", "0.2",
         "--trades", str(trades), "--equity", str(equity)]
    )  # fmt: skip
    assert code == 0
    out = capsys.readouterr().out
    assert "trend+mean_reversion" in out and "total_return_pct" in out
    frame = pd.read_csv(trades)
    assert set(frame["strategy_id"]) == {"TrendFollowingStrategy", "MeanReversionStrategy"}
    assert len(pd.read_csv(equity)) == 1000


def test_optimize_command(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["optimize", "--strategy", "trend", "--symbol", "AAPL", "--source", "file",
         "--file", str(FIXTURES / "aapl_1m.parquet"), "--grid", "sma_fast=5,10", "--grid", "sma_slow=30",
         "--train-size", "2000", "--validation-size", "1000", "--test-size", "1000", "--embargo", "10",
         "--min-trades", "1"]
    )  # fmt: skip
    assert code == 0
    assert "Mean out-of-sample total_return_pct" in capsys.readouterr().out
