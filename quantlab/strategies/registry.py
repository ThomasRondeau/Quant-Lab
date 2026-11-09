"""Strategies by name, for the CLI, the UI and the live sessions; parameters are validated by their sheet."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from quantlab.strategies.arima import ArimaStrategy
from quantlab.strategies.base import Strategy, check_size
from quantlab.strategies.cascade import CascadeStrategy
from quantlab.strategies.mean_reversion import MeanReversionStrategy
from quantlab.strategies.ml_hybrid import MLHybridStrategy
from quantlab.strategies.pair_trading import PairTradingStrategy
from quantlab.strategies.params import Param, ParamError, defaults, validate
from quantlab.strategies.trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategySpec:
    factory: type[Strategy]
    n_symbols: int = 1

    @property
    def params(self) -> tuple[Param, ...]:
        """The strategy's parameter sheet."""
        return self.factory.PARAMS

    def validate(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """`params` converted and checked (types, bounds, rules linking several values); raises ParamError."""
        values = validate(self.params, params)
        full = {**defaults(self.params), **values}
        errors = check_size(full) + self.factory.check(full)
        if errors:
            raise ParamError(f"{self.factory.__name__}: " + " ".join(errors))
        for warning in self.factory.warnings(full):
            logger.warning("%s: %s", self.factory.__name__, warning)
        return values

    def build(self, symbols: Sequence[str], params: Mapping[str, Any]) -> Strategy:
        if len(symbols) != self.n_symbols:
            raise ValueError(f"{self.factory.__name__} needs {self.n_symbols} symbol(s), got {list(symbols)}.")
        values = self.validate(params)
        if self.n_symbols == 2:
            strategy = self.factory(symbol_a=symbols[0], symbol_b=symbols[1], **values)  # type: ignore[call-arg]
        else:
            strategy = self.factory(symbol=symbols[0], **values)
        strategy.params = {**defaults(self.params), **values}
        return strategy


STRATEGIES: dict[str, StrategySpec] = {
    "trend": StrategySpec(TrendFollowingStrategy),
    "mean_reversion": StrategySpec(MeanReversionStrategy),
    "arima": StrategySpec(ArimaStrategy),
    "ml_hybrid": StrategySpec(MLHybridStrategy),
    "cascade": StrategySpec(CascadeStrategy),
    "pair": StrategySpec(PairTradingStrategy, n_symbols=2),
}


def build_strategy(name: str, symbols: Sequence[str], params: Mapping[str, Any] | None = None) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy {name!r}; available: {', '.join(STRATEGIES)}.")
    return STRATEGIES[name].build(symbols, params or {})
