"""Parameter sheets: they match the constructors, reach the strategy, validate values, and presets round trip."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import pytest

from quantlab.cli import main, parse_params
from quantlab.features.base import Feature
from quantlab.features.pipeline import FeaturePipeline
from quantlab.strategies.base import Strategy
from quantlab.strategies.params import Param, ParamError, defaults, diff, load_preset, save_preset, validate
from quantlab.strategies.registry import STRATEGIES, build_strategy

PRESETS = Path(__file__).parent.parent / "presets"
NOT_PARAMS = {"self", "symbol", "symbol_a", "symbol_b", "strategy_id", "kwargs"}
SYMBOLS = {1: ["AAPL"], 2: ["AAPL", "MSFT"]}


def constructor_defaults(cls: type) -> dict[str, Any]:
    return {
        name: p.default
        for name, p in inspect.signature(cls.__init__).parameters.items()  # type: ignore[misc]
        if name not in NOT_PARAMS
    }


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_sheet_matches_the_constructor(name: str) -> None:
    spec = STRATEGIES[name]
    sheet = {p.name: p for p in spec.params}
    own, base = constructor_defaults(spec.factory), constructor_defaults(Strategy)
    for key, default in own.items():  # every constructor argument is declared, with the same default
        assert key in sheet, f"{name}: {key} is not declared in PARAMS"
        expected = None if default is None else sheet[key].convert(default)
        assert sheet[key].default == expected, f"{name}.{key}: default {sheet[key].default!r} != {default!r}"
    for key, param in sheet.items():  # and every declared parameter reaches a constructor
        assert key in own or key in base, f"{name}: {key} is declared but not a constructor argument"
        if key not in own:
            assert param.default == base[key] or (param.default is None and base[key] is None)
    missing = set(base) - set(sheet)
    assert not missing or (spec.factory.is_group and missing == {"stop_atr", "take_profit_atr", "atr_window"})


def changed(param: Param) -> Any:
    """A valid value different from the default."""
    if param.kind == "bool":
        return not param.default
    if param.kind == "tuple":
        return (param.default[0] + 1, *param.default[1:])
    if param.kind == "choice":
        return next(c for c in param.choices if c != param.default)
    if param.default in (None, 0):
        return 1 if param.kind == "int" else 1.0
    value = param.default + 1 if param.kind == "int" else param.default * 0.75
    return min(value, param.max) if param.max is not None else value


def describe(value: Any, depth: int = 0) -> Any:
    """A deterministic description of an object graph (no memory addresses): features by name and inputs,
    estimators by their parameters, other objects by their attributes."""
    if isinstance(value, Feature):
        return ("feature", value.name, value.outputs, [describe(f, depth + 1) for f in value.inputs()])
    if hasattr(value, "get_params"):
        return ("estimator", type(value).__name__, describe(value.get_params(deep=False), depth + 1))
    if isinstance(value, dict):
        return sorted((str(k), describe(v, depth + 1)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return [describe(v, depth + 1) for v in value]
    if callable(value):
        return getattr(value, "__name__", type(value).__name__)
    if hasattr(value, "__dict__") and depth < 5:
        return (type(value).__name__, describe(vars(value), depth + 1))
    return repr(value)


def fingerprint(strategy: Strategy) -> str:
    """Everything a parameter can change: attributes, models and the resolved feature columns."""
    pipeline = FeaturePipeline([*strategy.features(), *strategy._exit_features()])
    return repr((describe(vars(strategy)), pipeline.columns, pipeline.warmup))


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_every_parameter_reaches_the_strategy(name: str) -> None:
    spec = STRATEGIES[name]
    reference = fingerprint(build_strategy(name, SYMBOLS[spec.n_symbols]))
    assert fingerprint(build_strategy(name, SYMBOLS[spec.n_symbols])) == reference  # deterministic
    for param in spec.params:
        extra = {"size_value": 1000.0} if param.name == "size_mode" else {}  # a non-fixed mode needs a value
        strategy = build_strategy(name, SYMBOLS[spec.n_symbols], {param.name: changed(param), **extra})
        assert fingerprint(strategy) != reference, f"{name}.{param.name} has no effect"


def test_values_are_converted_and_checked() -> None:
    sheet = STRATEGIES["mean_reversion"].params
    values = validate(sheet, {"bb_window": "30", "target_middle_band": "true", "stop_atr": "none", "bb_std": 2})
    assert values == {"bb_window": 30, "target_middle_band": True, "stop_atr": None, "bb_std": 2.0}
    assert validate(STRATEGIES["arima"].params, {"order": "(1, 1, 2)"}) == {"order": (1, 1, 2)}
    with pytest.raises(ParamError, match=r"unknown parameter 'bb_windw'.*Valid parameters: .*bb_window"):
        validate(sheet, {"bb_windw": 30})
    with pytest.raises(ParamError, match="below the minimum"):
        validate(sheet, {"bb_window": 1})
    with pytest.raises(ParamError, match="not a valid int"):
        validate(sheet, {"bb_window": 2.5})
    with pytest.raises(ParamError, match="a value is required"):
        validate(sheet, {"bb_window": None})
    with pytest.raises(ParamError):  # the registry validates too
        build_strategy("trend", ["AAPL"], {"sma_fats": 3})


@pytest.mark.parametrize(
    ("name", "params", "message"),
    [
        ("trend", {"sma_fast": 20, "sma_slow": 20}, "sma_fast must be smaller"),
        ("pair", {"exit_zscore": 2.5}, "exit_zscore must be smaller"),
        ("pair", {"stop_loss_pct": 0}, "stop_loss_pct: 0.0 is below the minimum 0.01. Valid"),
        ("pair", {"leg_stop_pct": None}, "leg_stop_pct: a value is required"),
    ],
)
def test_rules_linking_parameters(name: str, params: dict[str, Any], message: str) -> None:
    with pytest.raises(ParamError, match=message):
        build_strategy(name, SYMBOLS[STRATEGIES[name].n_symbols], params)


def test_pair_leg_stop_inside_the_combined_stop_is_a_warning() -> None:
    from quantlab.strategies.pair_trading import PairTradingStrategy

    values = {**defaults(STRATEGIES["pair"].params), "leg_stop_pct": 1.5}
    assert PairTradingStrategy.check(values) == []
    assert "a leg stop can fire" in PairTradingStrategy.warnings(values)[0]


def test_group_strategies_have_no_atr_exits() -> None:
    with pytest.raises(ParamError, match="unknown parameter 'stop_atr'"):
        build_strategy("pair", ["AAPL", "MSFT"], {"stop_atr": 1.5})


def test_presets_round_trip(tmp_path: Path) -> None:
    sheet = STRATEGIES["arima"].params
    values = validate(sheet, {"order": (1, 1, 1), "window": 80, "holding_bars": None})
    save_preset(tmp_path / "p.json", "arima", values)
    strategy, loaded = load_preset(tmp_path / "p.json")
    assert strategy == "arima" and validate(sheet, loaded) == values
    assert diff(sheet, {**defaults(sheet), **values}) == values
    (tmp_path / "bad.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ParamError):
        load_preset(tmp_path / "bad.json")


@pytest.mark.parametrize("path", sorted(PRESETS.glob("*.json")), ids=lambda p: p.name)
def test_versioned_presets_are_valid(path: Path) -> None:
    name, values = load_preset(path)
    assert path.name.startswith(f"{name}-")
    build_strategy(name, SYMBOLS[STRATEGIES[name].n_symbols], values)


def test_params_file_then_param_overrides(tmp_path: Path) -> None:
    save_preset(tmp_path / "t.json", "trend", {"sma_fast": 8, "holding_bars": 12})
    params = parse_params(["trend.sma_fast=9"], ["trend", "arima"], [tmp_path / "t.json"])
    assert params == {"trend": {"sma_fast": 9, "holding_bars": 12}, "arima": {}}
    with pytest.raises(argparse.ArgumentTypeError, match="not a selected strategy"):
        parse_params([], ["arima"], [tmp_path / "t.json"])


def test_params_command_lists_every_parameter(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["params", "--strategy", "arima"]) == 0
    out = capsys.readouterr().out
    assert all(p.name in out for p in STRATEGIES["arima"].params) and "2,1,0" in out


def test_invalid_cli_parameters_exit_clearly() -> None:
    fixture = str(Path(__file__).parent / "fixtures" / "aapl_1m.parquet")
    argv = ["backtest", "--strategy", "trend", "--symbol", "AAPL", "--source", "file", "--file", fixture]
    with pytest.raises(SystemExit, match="above the maximum"):
        main([*argv, "--param", "stop_loss_pct=500"])
