from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from quantlab.execution.orders import Order, Side
from quantlab.features.base import Feature
from quantlab.features.technical import SMA
from quantlab.strategies.base import Strategy
from quantlab.strategies.params import Param


class TrendFollowingStrategy(Strategy):
    """Fast/slow SMA crossover."""

    PARAMS = (
        Param("sma_fast", "int", 5, "Fenêtre de la moyenne mobile rapide (barres).", "indicateurs", min=1, max=1000),
        Param("sma_slow", "int", 20, "Fenêtre de la moyenne mobile lente (barres).", "indicateurs", min=2, max=5000),
        *Strategy.PARAMS,
    )

    @classmethod
    def check(cls, values: Mapping[str, Any]) -> list[str]:
        if values["sma_fast"] >= values["sma_slow"]:
            return ["sma_fast must be smaller than sma_slow."]
        return []

    def __init__(self, sma_fast: int = 5, sma_slow: int = 20, symbol: str = "AAPL", **kwargs: Any) -> None:
        super().__init__(symbol=symbol, **kwargs)
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.fast, self.slow = SMA(sma_fast), SMA(sma_slow)

    def features(self) -> Sequence[Feature]:
        return (self.fast, self.slow)

    def decide(self, bar: pd.Series) -> list[Order]:
        rows = self.feed.latest(2)
        if len(rows) < 2:
            return []
        fast, slow = rows[self.fast.name], rows[self.slow.name]

        if fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]:
            return [self.order(Side.BUY)]
        if fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]:
            return [self.order(Side.SELL)]
        return []
