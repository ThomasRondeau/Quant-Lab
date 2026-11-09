from __future__ import annotations

from typing import Any

import pandas as pd

from quantlab.execution.orders import Order, Side
from quantlab.models.arima import ArimaForecaster
from quantlab.strategies.base import Strategy
from quantlab.strategies.params import Param, with_defaults


class ArimaStrategy(Strategy):
    """Trade the direction of the ARIMA one-bar forecast when it beats trading costs."""

    PARAMS = (
        Param("min_edge_bps", "float", 2.0, "Mouvement prévu minimal pour trader, en points de base du prix.",
              "signal", min=0, max=10_000),
        Param("order", "tuple", (2, 1, 0), "Ordre ARIMA (p, d, q).", "modèle:arima", min=0, max=10),
        Param("window", "int", 50, "Clôtures sur lesquelles l'ARIMA est réestimé à chaque barre.", "modèle:arima",
              min=10, max=5000),
        *with_defaults(Strategy.PARAMS, quantity=5),
    )  # fmt: skip
    quantity = 5  # class value: the default of strategies pickled before it was a parameter

    def __init__(
        self,
        order: tuple[int, int, int] = (2, 1, 0),
        min_edge_bps: float = 2.0,
        window: int = 50,
        symbol: str = "AAPL",
        quantity: int = 5,
        **kwargs: Any,
    ) -> None:
        """
        :param order: ARIMA (p, d, q) order.
        :param min_edge_bps: minimum predicted move (in basis points of the price) to trade;
                             defaults to the round-trip fees (2 x 1 bp).
        :param window: closes the ARIMA is refitted on at each bar.
        """
        super().__init__(symbol=symbol, quantity=quantity, **kwargs)
        self.forecaster = ArimaForecaster(order=order, window=window)
        self.min_edge_bps = min_edge_bps
        self.min_history = window

    def decide(self, bar: pd.Series) -> list[Order]:
        closes = self.feed.bars(self.forecaster.window)["close"].to_numpy()
        prediction = self.forecaster.forecast_change(closes)
        self.diagnostics = {"forecast_change": float(prediction)}
        if abs(prediction) / float(bar["close"]) * 1e4 < self.min_edge_bps:
            return []
        return [self.order(Side.BUY if prediction > 0 else Side.SELL)]
